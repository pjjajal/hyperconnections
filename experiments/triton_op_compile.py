"""Standalone experiment: torch.library wrapping of the existing Triton ops.

Question: the ops in hyperconnections/ops are plain torch.autograd.Functions,
which Dynamo cannot trace -> every call is a graph break under torch.compile.
Does re-exposing the SAME kernels through torch.library remove the breaks, and
does that translate into end-to-end latency?

Variants per op (nothing under hyperconnections/ops is modified; all wrappers
live in this file):

  eager_ref   pure-PyTorch reference (the "mHC-like" path: fully traceable,
              Inductor fuses it wholesale).
  autograd_fn the existing autograd.Function path, imported as-is (status quo).
  triton_op   torch.library.triton_op + wrap_triton around the same raw
              kernels: Dynamo traces through; Inductor may fuse around the
              kernel and include it in cudagraph regions.
  custom_op   torch.library.custom_op + register_fake: the op stays opaque but
              graph-break-free (no fusion into the kernel; cudagraph-eligible).

Both torch.library variants register the SAME backward (register_autograd); its
kernel launches are wrapped as opaque custom ops because AOTAutograd traces the
registered backward with fake tensors -- a raw launch there fails with "Cannot
access data pointer of Tensor".

Each (op, variant) runs eager and under torch.compile (--modes), inside a small
harness with fusable elementwise ops on both sides of the op, at CGHC-forward
shapes: Phi/A [B, N, N], x/Y [B, N, D]. Reported per cell: graph breaks
(torch._dynamo.explain), fwd and fwd+bwd latency, max abs error of outputs and
input grads vs the pure-PyTorch reference.

Correctness failures are REPORTED, never raised -- the run always completes.

Usage (needs a CUDA GPU; A100 for representative numbers):
    python experiments/triton_op_compile.py [--ops stream_mix expm expm_block]
        [--modes default max-autotune] [--batch 4096] [--streams 4] [--dim 768]
        [--dtype fp32] [--warmup 10] [--iters 50] [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback

import torch
import torch.nn.functional as F
from torch import nn

import triton

from hyperconnections.ops.expm_triton import (
    _MAX_S,
    _TORCH_TO_TL,
    _expm_t18_augmented_fwd,
    _expm_t18_augmented_no_grad,
    _expm_t18_fwd,
    _expm_t18_no_grad,
    _expm_t18_structure_no_grad,
    _next_pow2,
    expm_t18_block_triton,
    expm_t18_triton,
)
from hyperconnections.ops.stream_mix_small_nb import (
    _launch_bwd_dPhi,
    _launch_bwd_dx,
    _launch_fwd,
    _stream_mix_fwd,
    stream_mix_add_small_nb,
)

try:
    from torch.library import custom_op, triton_op, wrap_triton
except ImportError as exc:  # torch too old for triton_op/wrap_triton
    raise SystemExit(f"torch.library.triton_op unavailable ({exc}); need torch >= 2.6")


# --------------------------------------------------------------------------
# torch.library wrappers -- stream_mix (small-NB backend, chosen explicitly so
# every variant runs the identical kernel; the ops.stream_mix dispatch
# heuristic is deliberately bypassed)
# --------------------------------------------------------------------------
@triton_op("hc_exp::stream_mix_b", mutates_args={})
def stream_mix_b(Phi: torch.Tensor, x: torch.Tensor, Y: torch.Tensor,
                 v: torch.Tensor | None = None) -> torch.Tensor:
    Phi_c, x_c, Y_c = Phi.contiguous(), x.contiguous(), Y.contiguous()
    B, N, D = x_c.shape
    use_proj = v is not None
    v_arg = v.contiguous() if v is not None else torch.zeros(B, N, dtype=x_c.dtype, device=x_c.device)
    out = torch.empty_like(x_c)
    grid = lambda meta: (B * N, triton.cdiv(D, meta["BLOCK_D"]))
    wrap_triton(_stream_mix_fwd)[grid](
        Phi_c, x_c, Y_c, out, v_arg,
        D,
        *Phi_c.stride(), *x_c.stride(), *Y_c.stride(), *out.stride(), *v_arg.stride(),
        N_STREAMS=N, USE_PROJ=use_proj,
    )
    return out


@custom_op("hc_exp::stream_mix_c", mutates_args=())
def stream_mix_c(Phi: torch.Tensor, x: torch.Tensor, Y: torch.Tensor,
                 v: torch.Tensor | None = None) -> torch.Tensor:
    Phi_c, x_c, Y_c = Phi.contiguous(), x.contiguous(), Y.contiguous()
    out = torch.empty_like(x_c)
    _launch_fwd(Phi_c, x_c, Y_c, v.contiguous() if v is not None else None, out)
    return out


@stream_mix_c.register_fake
def _(Phi, x, Y, v=None):
    return torch.empty_like(x)


# The registered backwards are traced by AOTAutograd with fake tensors, so any
# raw kernel launch inside them fails ("Cannot access data pointer of Tensor");
# the backward kernel launches must themselves be opaque custom ops. Shared by
# the triton_op and custom_op forward variants.
@custom_op("hc_exp::stream_mix_bwd_dx_", mutates_args={"grad_x"})
def stream_mix_bwd_dx_(G: torch.Tensor, Phi: torch.Tensor, v: torch.Tensor | None,
                       beta: torch.Tensor | None, grad_x: torch.Tensor, N: int) -> None:
    _launch_bwd_dx(G, Phi, v, beta, grad_x, N)


@stream_mix_bwd_dx_.register_fake
def _(G, Phi, v, beta, grad_x, N):
    return None


@custom_op("hc_exp::stream_mix_bwd_dphi_", mutates_args={"grad_Phi"})
def stream_mix_bwd_dphi_(G: torch.Tensor, x: torch.Tensor, v: torch.Tensor | None,
                         alpha: torch.Tensor | None, grad_Phi: torch.Tensor, N: int) -> None:
    _launch_bwd_dPhi(G, x, v, alpha, grad_Phi, N)


@stream_mix_bwd_dphi_.register_fake
def _(G, x, v, alpha, grad_Phi, N):
    return None


def _stream_mix_setup(ctx, inputs, output):
    Phi, x, Y, v = inputs
    ctx.save_for_backward(Phi.contiguous(), x.contiguous(),
                          v.contiguous() if v is not None else None)


def _stream_mix_backward(ctx, grad_out):
    # Same math as ops.stream_mix_small_nb._StreamMixFn.backward (grads computed
    # unconditionally; the harness differentiates every input anyway).
    Phi, x, v = ctx.saved_tensors
    B, N, D = x.shape
    use_proj = v is not None
    G = grad_out.float().contiguous()

    alpha = beta = None
    if use_proj:
        alpha = torch.einsum("bn,bnd->bd", v.float(), x.float())
        phi_v = torch.bmm(Phi.float(), v.float().unsqueeze(-1)).squeeze(-1)
        c = v.float() - phi_v
        beta = torch.einsum("bnd,bn->bd", G, c)

    grad_x = torch.empty_like(x)
    stream_mix_bwd_dx_(G, Phi, v, beta, grad_x, N)
    grad_Phi = torch.empty(B, N, N, dtype=torch.float32, device=x.device)
    stream_mix_bwd_dphi_(G, x, v, alpha, grad_Phi, N)

    grad_v = None
    if use_proj:
        rho = (G * alpha.unsqueeze(1)).sum(dim=2)
        rho_part = rho - torch.bmm(Phi.float().mT, rho.unsqueeze(-1)).squeeze(-1)
        beta_part = torch.einsum("bd,bnd->bn", beta, x.float())
        grad_v = (rho_part + beta_part).to(v.dtype)

    return grad_Phi.to(Phi.dtype), grad_x, grad_out, grad_v


stream_mix_b.register_autograd(_stream_mix_backward, setup_context=_stream_mix_setup)
stream_mix_c.register_autograd(_stream_mix_backward, setup_context=_stream_mix_setup)


# --------------------------------------------------------------------------
# torch.library wrappers -- expm_t18
# --------------------------------------------------------------------------
@triton_op("hc_exp::expm_t18_b", mutates_args={})
def expm_t18_b(A: torch.Tensor, S: int = _MAX_S) -> torch.Tensor:
    B, N, _ = A.shape
    A_fp32 = A.to(torch.float32).contiguous()
    inv_scale = 2.0 ** (-S)
    out = torch.empty(B, N, N, dtype=A.dtype, device=A.device)
    NP = _next_pow2(N)
    wrap_triton(_expm_t18_fwd)[(B,)](
        A_fp32, out,
        inv_scale,
        *A_fp32.stride(), *out.stride(),
        N=N, NP=NP, MAX_S=S,
        OUT_DTYPE=_TORCH_TO_TL[A.dtype],
        LOW_PREC=A.dtype in (torch.float16, torch.bfloat16),
    )
    return out


@custom_op("hc_exp::expm_t18_c", mutates_args=())
def expm_t18_c(A: torch.Tensor, S: int = _MAX_S) -> torch.Tensor:
    return _expm_t18_no_grad(A, out_dtype=A.dtype, S=S)


@expm_t18_c.register_fake
def _(A, S=_MAX_S):
    return torch.empty_like(A)


@custom_op("hc_exp::expm_structured", mutates_args=())
def expm_structured(A_T: torch.Tensor, G: torch.Tensor, out_dtype: torch.dtype,
                    S: int) -> torch.Tensor:
    return _expm_t18_structure_no_grad(A_T, G, out_dtype=out_dtype, S=S)


@expm_structured.register_fake
def _(A_T, G, out_dtype, S):
    return torch.empty_like(G)  # matches the real impl's out = empty_like(G)


def _expm_setup(ctx, inputs, output):
    A, S = inputs
    ctx.save_for_backward(A)
    ctx.S = S


def _expm_backward(ctx, grad_out):
    # Same math as ops.expm_triton._ExpmT18TritonFn.backward.
    (A,) = ctx.saved_tensors
    A_T = A.float().transpose(-1, -2).contiguous()
    G = grad_out.float().contiguous()
    dA = expm_structured(A_T, G, A.dtype, ctx.S)
    return dA, None


expm_t18_b.register_autograd(_expm_backward, setup_context=_expm_setup)
expm_t18_c.register_autograd(_expm_backward, setup_context=_expm_setup)


# --------------------------------------------------------------------------
# torch.library wrappers -- expm_t18_block (forced/augmented variant)
# --------------------------------------------------------------------------
@triton_op("hc_exp::expm_t18_block_b", mutates_args={})
def expm_t18_block_b(A: torch.Tensor, S: int = _MAX_S) -> tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = A.shape
    A_fp32 = A.to(torch.float32).contiguous()
    inv_scale = 2.0 ** (-S)
    E = torch.empty(B, N, N, dtype=A.dtype, device=A.device)
    psi = torch.empty(B, N, N, dtype=A.dtype, device=A.device)
    NP = _next_pow2(N)
    wrap_triton(_expm_t18_augmented_fwd)[(B,)](
        A_fp32, E, psi,
        inv_scale,
        *A_fp32.stride(), *E.stride(), *psi.stride(),
        N=N, NP=NP, MAX_S=S,
        OUT_DTYPE=_TORCH_TO_TL[A.dtype],
        LOW_PREC=A.dtype in (torch.float16, torch.bfloat16),
    )
    return E, psi


@custom_op("hc_exp::expm_t18_block_c", mutates_args=())
def expm_t18_block_c(A: torch.Tensor, S: int = _MAX_S) -> tuple[torch.Tensor, torch.Tensor]:
    return _expm_t18_augmented_no_grad(A, out_dtype=A.dtype, S=S)


@expm_t18_block_c.register_fake
def _(A, S=_MAX_S):
    return torch.empty_like(A), torch.empty_like(A)


def _expm_block_setup(ctx, inputs, output):
    A, S = inputs
    ctx.save_for_backward(A.float().transpose(-1, -2).contiguous())
    ctx.input_dtype = A.dtype
    ctx.S = S


def _expm_block_backward(ctx, grad_E, grad_psi):
    # Same math as ops.expm_triton._ExpmT18AugmentedTritonFn.backward, with the
    # kernel launches routed through the ops above so AOTAutograd can trace it.
    (A_T,) = ctx.saved_tensors
    out_dtype, S = ctx.input_dtype, ctx.S
    B, N, _ = A_T.shape
    G_E = grad_E.float().contiguous()
    G_psi = grad_psi.float().contiguous()

    dA_E = expm_structured(A_T, G_E, out_dtype, S)

    M = torch.zeros(B, 3 * N, 3 * N, dtype=torch.float32, device=A_T.device)
    M[:, :N, :N] = A_T
    M[:, :N, N:2 * N] = G_psi
    M[:, N:2 * N, N:2 * N] = A_T
    M[:, N:2 * N, 2 * N:3 * N] = torch.eye(N, dtype=torch.float32, device=A_T.device)
    # expm_t18_c keeps M's fp32 dtype (the original rounds through out_dtype
    # first; identical for fp32 runs, negligible for bf16/fp16).
    expM = expm_t18_c(M, S)
    dA_psi = expM[:, :N, 2 * N:3 * N]

    return dA_E + dA_psi, None


expm_t18_block_b.register_autograd(_expm_block_backward, setup_context=_expm_block_setup)
expm_t18_block_c.register_autograd(_expm_block_backward, setup_context=_expm_block_setup)


# --------------------------------------------------------------------------
# pure-PyTorch references (differentiable, fully traceable)
# --------------------------------------------------------------------------
def stream_mix_ref(Phi, x, Y, v=None):
    out = torch.bmm(Phi, x) + Y
    if v is not None:
        proj = v.unsqueeze(-1) - torch.bmm(Phi, v.unsqueeze(-1))   # [B, N, 1]
        vTx = torch.einsum("bn,bnd->bd", v, x)                     # [B, D]
        out = out + proj * vTx.unsqueeze(1)
    return out


def expm_ref(A, S=_MAX_S):
    return torch.linalg.matrix_exp(A)


def expm_block_ref(A, S=_MAX_S):
    # exp([[A, I], [0, 0]]) -> top row blocks are (exp(A), phi_1(A)).
    B, N, _ = A.shape
    M = torch.zeros(B, 2 * N, 2 * N, dtype=A.dtype, device=A.device)
    M[:, :N, :N] = A
    M[:, :N, N:] = torch.eye(N, dtype=A.dtype, device=A.device)
    expM = torch.linalg.matrix_exp(M)
    return expM[:, :N, :N], expM[:, :N, N:]


# --------------------------------------------------------------------------
# harnesses: fusable elementwise ops on both sides of the op under test
# --------------------------------------------------------------------------
class StreamMixHarness(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.impl = impl

    def forward(self, Phi_raw, x_raw, Y_raw, v_raw):
        Phi = torch.tanh(Phi_raw) * 0.5
        x = F.silu(x_raw)
        Y = Y_raw * torch.sigmoid(Y_raw)
        v = F.normalize(v_raw, dim=-1) if v_raw is not None else None
        out = self.impl(Phi, x, Y, v)
        return F.gelu(out) * 1.5


class ExpmHarness(nn.Module):
    """Bounded generator (tanh * tau keeps ||A||_1 inside the fixed-S T18
    trust region) -> expm -> transition applied to the streams."""

    def __init__(self, impl, tau=0.5):
        super().__init__()
        self.impl = impl
        self.tau = tau

    def forward(self, A_raw, x):
        A = torch.tanh(A_raw) * self.tau
        E = self.impl(A)
        out = torch.bmm(E, x)
        return F.silu(out)


class ExpmBlockHarness(nn.Module):
    def __init__(self, impl, tau=0.5):
        super().__init__()
        self.impl = impl
        self.tau = tau

    def forward(self, A_raw, x, f):
        A = torch.tanh(A_raw) * self.tau
        E, psi = self.impl(A)
        out = torch.bmm(E, x) + torch.bmm(psi, f)
        return F.silu(out)


# --------------------------------------------------------------------------
# op suites
# --------------------------------------------------------------------------
def build_suites(args, device, dtype):
    B, N, D = args.batch, args.streams, args.dim
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)

    suites = {}

    sm_inputs = (randn(B, N, N), randn(B, N, D), randn(B, N, D),
                 None if args.no_proj else randn(B, N))
    suites["stream_mix"] = {
        "harness": StreamMixHarness,
        "inputs": sm_inputs,
        "variants": {
            "eager_ref": stream_mix_ref,
            "autograd_fn": stream_mix_add_small_nb,
            "triton_op": stream_mix_b,
            "custom_op": stream_mix_c,
        },
    }

    ex_inputs = (randn(B, N, N), randn(B, N, D))
    suites["expm"] = {
        "harness": ExpmHarness,
        "inputs": ex_inputs,
        "variants": {
            "eager_ref": expm_ref,
            "autograd_fn": expm_t18_triton,
            "triton_op": expm_t18_b,
            "custom_op": expm_t18_c,
        },
    }

    exb_inputs = (randn(B, N, N), randn(B, N, D), randn(B, N, D))
    suites["expm_block"] = {
        "harness": ExpmBlockHarness,
        "inputs": exb_inputs,
        "variants": {
            "eager_ref": expm_block_ref,
            "autograd_fn": expm_t18_block_triton,
            "triton_op": expm_t18_block_b,
            "custom_op": expm_t18_block_c,
        },
    }

    return suites


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------
def _clone_inputs(inputs, requires_grad):
    return tuple(t.detach().clone().requires_grad_(requires_grad) if t is not None else None
                 for t in inputs)


def explain_breaks(module, inputs):
    torch._dynamo.reset()
    ex = torch._dynamo.explain(module)(*inputs)
    reasons = sorted({str(r.reason_data if hasattr(r, "reason_data") else r) for r in ex.break_reasons})
    return {"graphs": ex.graph_count, "graph_breaks": ex.graph_break_count,
            "break_reasons": reasons[:8]}


def time_module(module, inputs, *, warmup, iters, backward):
    ins = _clone_inputs(inputs, requires_grad=backward)

    def run_once():
        if backward:
            out = module(*ins)
            out.float().square().mean().backward()
            for t in ins:
                if t is not None:
                    t.grad = None
        else:
            with torch.no_grad():
                module(*ins)

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


def check_correctness(module, ref_module, inputs):
    """Max abs error of outputs and of input grads vs the pure-torch reference."""
    ins = _clone_inputs(inputs, requires_grad=True)
    ref_ins = _clone_inputs(inputs, requires_grad=True)

    out = module(*ins)
    ref = ref_module(*ref_ins)
    out_err = (out.float() - ref.float()).abs().max().item()

    out.float().square().mean().backward()
    ref.float().square().mean().backward()
    grad_err = max(
        ((a.grad.float() - b.grad.float()).abs().max().item()
         for a, b in zip(ins, ref_ins) if a is not None and a.grad is not None and b.grad is not None),
        default=float("nan"),
    )
    return {"out_max_abs_err": out_err, "grad_max_abs_err": grad_err}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run_op(op_name, suite, args, results):
    print(f"\n==== op: {op_name} ====")
    inputs = suite["inputs"]
    ref_module = suite["harness"](suite["variants"]["eager_ref"])

    for vname, impl in suite["variants"].items():
        base = {"op": op_name, "variant": vname}
        try:
            module = suite["harness"](impl)

            # correctness (eager, vs pure-torch reference)
            if vname == "eager_ref":
                corr = {"out_max_abs_err": 0.0, "grad_max_abs_err": 0.0}
            else:
                corr = check_correctness(module, ref_module, inputs)
            base["correctness"] = corr
            tol = args.tolerance
            flag = "" if max(corr.values()) <= tol or corr["grad_max_abs_err"] != corr["grad_max_abs_err"] \
                else f"  <-- exceeds --tolerance {tol} (reported, not fatal)"
            print(f"[{vname}] out_err={corr['out_max_abs_err']:.3e} "
                  f"grad_err={corr['grad_max_abs_err']:.3e}{flag}")

            # graph breaks (once per variant; compile-mode independent)
            base["dynamo"] = explain_breaks(suite["harness"](impl), inputs)
            print(f"[{vname}] graphs={base['dynamo']['graphs']} "
                  f"breaks={base['dynamo']['graph_breaks']}")
            for reason in base["dynamo"]["break_reasons"]:
                print(f"          break: {reason[:140]}")

            # latency: eager + each compile mode
            base["timings"] = {}
            for backend in ["eager"] + [f"compile:{m}" for m in args.modes]:
                torch._dynamo.reset()
                module = suite["harness"](impl)
                if backend.startswith("compile:"):
                    module = torch.compile(module, mode=backend.split(":", 1)[1])
                cell = {}
                for backward in (False, True):
                    key = "fwd_bwd" if backward else "fwd"
                    try:
                        cell[key] = time_module(module, inputs, warmup=args.warmup,
                                                iters=args.iters, backward=backward)
                    except Exception as exc:  # report, never abort
                        cell[key] = {"error": f"{type(exc).__name__}: {exc}"}
                base["timings"][backend] = cell
                fwd = cell["fwd"].get("mean_ms")
                bwd = cell["fwd_bwd"].get("mean_ms")
                fwd_s = f"{fwd:8.3f}ms" if fwd is not None else f"FAILED: {cell['fwd']['error'][:60]}"
                bwd_s = f"{bwd:8.3f}ms" if bwd is not None else f"FAILED: {cell['fwd_bwd']['error'][:60]}"
                print(f"[{vname}] {backend:24s} fwd {fwd_s}   fwd+bwd {bwd_s}")
        except Exception as exc:  # report, never abort
            base["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{vname}] FAILED: {base['error']}")
            traceback.print_exc()
        results.append(base)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ops", nargs="+", default=["stream_mix", "expm"],
                   choices=["stream_mix", "expm", "expm_block"])
    p.add_argument("--modes", nargs="+", default=["default", "max-autotune"])
    p.add_argument("--batch", type=int, default=4096,
                   help="Effective batch = LM batch * seq_len (CGHC applies the ops per token).")
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--no-proj", action="store_true", help="Disable the projected stream_mix variant.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tolerance", type=float, default=5e-3,
                   help="Correctness flag threshold (reported only, never fatal).")
    p.add_argument("--output", default=None, help="Optional JSON report path.")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required (the Triton kernels are CUDA-only).")
    device = "cuda"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 64  # many (variant x mode) combos in one process

    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__} "
          f"triton={triton.__version__}")
    print(f"B={args.batch} N={args.streams} D={args.dim} dtype={args.dtype} "
          f"proj={not args.no_proj} warmup={args.warmup} iters={args.iters}")

    suites = build_suites(args, device, dtype)
    results = []
    for op_name in args.ops:
        run_op(op_name, suites[op_name], args, results)

    if args.output:
        report = {"experiment": "triton_op_compile", "torch": torch.__version__,
                  "triton": triton.__version__, "device": torch.cuda.get_device_name(),
                  "config": vars(args), "results": results}
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
