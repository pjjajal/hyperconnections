"""Where does compiled CGHC lose to compiled mHC? Stage-level breakdown.

Isolated, same-shape comparison of the two residual ops with an identity
sublayer (so attention/MLP cost is excluded) at the LM benchmark's working
point: n=4, m=1, embed_dim=768 -> input_dim=3072, tokens=8*1024, bf16
autocast, per the minipile cghc_124m / mhc_124m configs.

Module-level variants (eager + torch.compile mode=default, the repo standard):
  mhc              full ManifoldHyperConnections (sinkhorn_iters=20)
  cghc             full ContinuousGenHyperConnections, use_triton=True
  cghc_eagerops    use_triton=False (pure-torch T18 expm + einsum mix)
  cghc_linphi      cghc with the matrix exponential ablated (Phi = A, i.e.
                   _matrix_exp = identity): the ONLY change is that exp
                   disappears -- its cost is cghc minus cghc_linphi.
  cghc_einsum_mix  cghc with triton expm but einsum stream mix (isolates the
                   stream_mix kernel's contribution).
  cghc_linphi_einsum  both ablations: the everything-else floor of cghc.

Stage timings (each compiled separately at the exact shapes the modules use):
  cghc: generator A | expm variants (triton T18 / eager T18 / linalg) |
        read-write weights | stream_mix (triton / einsum)
  mhc:  mixing weights incl. sinkhorn | sinkhorn alone | mixing einsum

Everything is REPORTED, never raised -- the run always completes.

Usage (CUDA GPU; A100 for representative numbers):
    python experiments/hc_stage_breakdown.py [--tokens 8192] [--no-autocast]
        [--warmup 10] [--iters 50] [--output report.json]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
import traceback

import torch
import torch.nn.functional as F
from torch import nn

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.mhc import ManifoldHyperConnections
from hyperconnections.ops import HAS_TRITON
from hyperconnections.ops.expm import expm_t18
from hyperconnections.ops.expm_triton import expm_t18_triton
from hyperconnections.ops.stream_mix import stream_mix_add

N, M, EMBED_DIM = 4, 1, 768                # minipile 124m configs
INPUT_DIM = (N // M) * EMBED_DIM           # 3072
BLOCK_SIZE = EMBED_DIM // M                # 768

CGHC_ARGS = dict(                          # cghc_124m.yaml init_args
    n=N, m=M, input_dim=INPUT_DIM, embed_dim=EMBED_DIM,
    bias=False, elementwise_affine=True,
    generator_type="conservative_diag_diss", projection="v",
    learn_dt=True, dt=0.1, dt_min=0.0001, dt_max=0.20, vec_dt=True,
)
MHC_ARGS = dict(                           # mhc_124m.yaml init_args
    n=N, m=M, input_dim=INPUT_DIM, embed_dim=EMBED_DIM,
    bias=False, elementwise_affine=True,
    sinkhorn_iters=20, sinkhorn_bias_init=5.0,
)


def make_sublayer(kind):
    """identity: pure HC-op cost. mlp: SwiGLU-sized MLP so the harness exposes
    fusion at the HC<->sublayer seam like a real per-block-compiled Block."""
    if kind == "identity":
        return nn.Identity()
    return nn.Sequential(
        nn.Linear(EMBED_DIM, 4 * EMBED_DIM, bias=False),
        nn.SiLU(),
        nn.Linear(4 * EMBED_DIM, EMBED_DIM, bias=False),
    )


def build_variants(sublayer):
    dev = "cuda"
    variants = {}
    variants["mhc"] = ManifoldHyperConnections(module=make_sublayer(sublayer), **MHC_ARGS).to(dev)
    variants["cghc"] = ContinuousGenHyperConnections(
        module=make_sublayer(sublayer), use_triton=True, **CGHC_ARGS).to(dev)
    variants["cghc_eagerops"] = ContinuousGenHyperConnections(
        module=make_sublayer(sublayer), use_triton=False, **CGHC_ARGS).to(dev)

    hc = ContinuousGenHyperConnections(module=make_sublayer(sublayer), use_triton=True, **CGHC_ARGS).to(dev)
    hc._matrix_exp = lambda A: A           # Phi = A: exp ablated, grads intact
    variants["cghc_linphi"] = hc

    hc = ContinuousGenHyperConnections(module=make_sublayer(sublayer), use_triton=True, **CGHC_ARGS).to(dev)
    hc._stream_mix = hc._stream_mix_eager  # triton expm, einsum mix
    variants["cghc_einsum_mix"] = hc

    hc = ContinuousGenHyperConnections(module=make_sublayer(sublayer), use_triton=True, **CGHC_ARGS).to(dev)
    hc._matrix_exp = lambda A: A
    hc._stream_mix = hc._stream_mix_eager
    variants["cghc_linphi_einsum"] = hc
    return variants


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def _time(fn, inputs, *, warmup, iters, backward, autocast):
    ins = tuple(t.detach().clone().requires_grad_(backward) for t in inputs)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
           else contextlib.nullcontext())

    def run_once():
        if backward:
            with ctx:
                out = fn(*ins)
            out = out[0] if isinstance(out, tuple) else out
            out.float().square().mean().backward()
            for t in ins:
                t.grad = None
        else:
            with torch.no_grad(), ctx:
                fn(*ins)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times),
            "min_ms": min(times)}


def bench(name, fn, inputs, args, results, *, compile_fn=True):
    row = {"name": name, "timings": {}}
    for backend in ("eager", "compile:default") if compile_fn else ("eager",):
        torch._dynamo.reset()
        run = torch.compile(fn, mode="default") if backend != "eager" else fn
        cell = {}
        for backward in (False, True):
            key = "fwd_bwd" if backward else "fwd"
            try:
                cell[key] = _time(run, inputs, warmup=args.warmup, iters=args.iters,
                                  backward=backward, autocast=args.autocast)
            except Exception as exc:  # report, never abort
                cell[key] = {"error": f"{type(exc).__name__}: {exc}"}
        row["timings"][backend] = cell
        fwd, bwd = cell["fwd"].get("mean_ms"), cell["fwd_bwd"].get("mean_ms")
        fwd_s = f"{fwd:7.3f}" if fwd is not None else f"FAIL({cell['fwd']['error'][:48]})"
        bwd_s = f"{bwd:7.3f}" if bwd is not None else f"FAIL({cell['fwd_bwd']['error'][:48]})"
        print(f"  {name:22s} {backend:16s} fwd {fwd_s} ms   fwd+bwd {bwd_s} ms", flush=True)
    results.append(row)
    return row


def get(results, name, backend="compile:default", key="fwd"):
    for r in results:
        if r["name"] == name:
            return r["timings"].get(backend, {}).get(key, {}).get("mean_ms")
    return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokens", type=int, default=8192,
                   help="Tokens per forward (sweep bs8 x seq1024 = 8192).")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--no-autocast", dest="autocast", action="store_false",
                   help="Disable bf16 autocast (default on, matching the LM benchmark).")
    p.add_argument("--sublayer", default="identity", choices=["identity", "mlp"],
                   help="Wrapped module: identity isolates the HC op; mlp exposes "
                        "fusion at the HC<->sublayer seam.")
    p.add_argument("--module-only", action="store_true",
                   help="Skip the stage-level sections (module variants + accounting only).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")
    assert HAS_TRITON, "Triton kernels unavailable; this breakdown needs them."
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64

    B = args.tokens
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
          f"tokens={B} n={N} m={M} input_dim={INPUT_DIM} autocast={args.autocast}")

    results = []

    print(f"\n==== module-level (sublayer={args.sublayer}) ====")
    x = torch.randn(B, INPUT_DIM, device="cuda")
    for name, mod in build_variants(args.sublayer).items():
        bench(name, mod, (x,), args, results)

    if args.module_only:
        _accounting(results)
        _write(args, results)
        return

    print("\n==== cghc stages ====")
    hc = ContinuousGenHyperConnections(module=nn.Identity(), use_triton=True, **CGHC_ARGS).cuda()
    x_norm = torch.randn(B, INPUT_DIM, device="cuda")
    A = hc.compute_generator(torch.randn(B, INPUT_DIM, device="cuda")).detach()
    xs = torch.randn(B, N, BLOCK_SIZE, device="cuda")
    Y = torch.randn(B, N, BLOCK_SIZE, device="cuda")
    v = F.normalize(torch.randn(B, N, device="cuda"), dim=-1)

    bench("gen:A(x_norm)", hc.compute_generator, (x_norm,), args, results)
    bench("expm:triton", lambda A: expm_t18_triton(A), (A,), args, results)
    bench("expm:t18_eager", lambda A: expm_t18(A.float()), (A,), args, results)
    bench("expm:linalg", torch.linalg.matrix_exp, (A,), args, results)
    bench("weights:read_write", hc.compute_read_write_weights, (x_norm,), args, results)
    bench("mix:triton", lambda P, xx, YY, vv: stream_mix_add(P, xx, YY, vv),
          (A, xs, Y, v), args, results)
    bench("mix:einsum", lambda P, xx, YY, vv: hc._stream_mix_eager(xx, P, YY, vv),
          (A, xs, Y, v), args, results)

    print("\n==== mhc stages ====")
    mhc = ManifoldHyperConnections(module=nn.Identity(), **MHC_ARGS).cuda()
    src = torch.randn(B, N, BLOCK_SIZE, device="cuda")
    H = torch.randn(B, N, N, device="cuda")
    bench("mhc:weights+sinkhorn", mhc.compute_mixing_weights, (src,), args, results)
    bench("mhc:sinkhorn_only", mhc._sinkhorn_knopp, (H,), args, results)
    bench("mhc:mix_einsum",
          lambda S, xx: torch.einsum("bij,bjd->bid", S, xx), (H, xs), args, results)

    _accounting(results)
    _write(args, results)


def _accounting(results):
    print("\n==== gap accounting (compiled, mode=default) ====")
    for key, label in (("fwd", "fwd (inference, matches latency sweep)"),
                       ("fwd_bwd", "fwd+bwd (training)")):
        c = get(results, "cghc", key=key)
        m_ = get(results, "mhc", key=key)
        lp = get(results, "cghc_linphi", key=key)
        em = get(results, "cghc_einsum_mix", key=key)
        floor = get(results, "cghc_linphi_einsum", key=key)
        if None in (c, m_, lp):
            print(f"  [{label}] incomplete (a variant failed); see rows above")
            continue
        print(f"  [{label}]")
        print(f"    cghc - mhc gap per sublayer : {c - m_:+7.3f} ms  (x24 sublayers = {(c - m_) * 24:+7.2f} ms/model)")
        print(f"    expm cost (cghc - linphi)   : {c - lp:+7.3f} ms")
        if em is not None:
            print(f"    stream_mix kernel benefit   : {em - c:+7.3f} ms  (einsum_mix - cghc)")
        if floor is not None:
            print(f"    non-expm gap (linphi - mhc) : {lp - m_:+7.3f} ms   everything-else floor - mhc: {floor - m_:+7.3f} ms")


def _write(args, results):
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"experiment": "hc_stage_breakdown", "torch": torch.__version__,
                       "device": torch.cuda.get_device_name(), "config": vars(args),
                       "results": results}, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
