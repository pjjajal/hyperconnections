"""
S-ablation benchmark for expm_t18_block_triton.

The kernel now takes the number of scaling-and-squaring steps S as a constexpr:
it scales by 2^(-S) and squares exactly S times (no per-input norm reduction).
Two distinct notions of S appear here:

    s_req = ceil(log2(||A||_1 / theta_18))   (theta_18 = 3.01)
            = the S a given norm *needs* for accuracy
    S     = the constexpr you actually pass to the kernel

- CORRECTNESS sweeps ||A||_1 (hence s_req) at the default S, showing where the
  fixed-S scale-and-square is accurate (S >= s_req) vs under-resolved / over-scaled.
- PERF sweeps the constexpr S (--squarings) at a fixed representative norm. Runtime
  now scales with S — each S is one real block-squaring matmul — so this genuinely
  *controls* compute rather than simulating it.

Unlike expm_force_bench.py this focuses solely on *our* kernel: no head-to-head
timing against torch.linalg.matrix_exp or the pure-torch T18.

Usage
-----
# Run everything (default):
    python benchmarks/expm_norm_bench.py

# Only correctness / only perf:
    python benchmarks/expm_norm_bench.py --mode correctness
    python benchmarks/expm_norm_bench.py --mode perf

# Custom norm sweep / N / dtype:
    python benchmarks/expm_norm_bench.py --norms 0.5 5 50 500
    python benchmarks/expm_norm_bench.py --n 4 8 --dtype bf16
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date
from itertools import product
from typing import Sequence

import torch

from hyperconnections.ops import expm_t18_block_triton
from hyperconnections.ops.expm_triton import _MAX_S
from hyperconnections.ops.numbers import _THETA_18_F32 as _THETA_18

from bench_utils import (ok, fail, warn, cyan, bold, _dtype,
                         bench_stats, stat_fields, logger, setup_logging)
from expm_common import (_REPORT_DIR, _write_csv, _make_large_norm_A,
                         ref_torch_matrix_exp)
# Ground-truth references for psi and the augmented-matrix gradient live in the
# force benchmark; reuse them rather than duplicating ~40 lines.
from expm_force_bench import (ref_phi1_quadrature, ref_grad_matrix_exp_aug,
                              _build_aug_2N)


def _squarings_for_norm(norm: float) -> int:
    """Host-side S for target ||A||_1 = norm, matching _expm_t18_no_grad:
    A_norm is clamped up to theta_18, then S = ceil(log2(A_norm / theta_18))
    clamped to >= 0.  Returned uncapped so S > MAX_S (kernel saturation) is
    visible in the report."""
    a_norm = max(norm, _THETA_18)
    return max(0, math.ceil(math.log2(a_norm / _THETA_18)))


def _check_status(got: torch.Tensor, ref: torch.Tensor, atol: float):
    """Magnitude-aware correctness check, reporting status + abs/rel error.

    Returns (status, max_err, rel_err) where:
      - status == 'overflow' if got or ref is non-finite (exp(A) past the
        representable range) — not a meaningful correctness signal, so it is
        flagged distinctly rather than counted as a FAIL;
      - status == 'bigpass' if max_err == 0 (exact match);
      - else 'pass'/'fail' from the same threshold as expm_common._check:
        max_err <= atol * (1 + ||ref||_inf).
    rel_err = max_err / (1 + ||ref||_inf) is the quantity that threshold bounds,
    so it (not the raw absolute max_err) is what the atol should be read against.
    """
    g, r = got.float(), ref.float()
    if not (torch.isfinite(g).all() and torch.isfinite(r).all()):
        return "overflow", float("nan"), float("nan")
    max_err = (g - r).abs().max().item()
    ref_mag = r.abs().max().item()
    rel_err = max_err / (1.0 + ref_mag)
    if max_err == 0.0:
        return "bigpass", 0.0, 0.0
    status = "pass" if max_err <= atol * (1.0 + ref_mag) else "fail"
    return status, max_err, rel_err


_RESULT_LABEL = {
    "bigpass":  lambda: cyan("BIGPASS"),
    "pass":     lambda: ok("PASS"),
    "fail":     lambda: fail("FAIL"),
    "overflow": lambda: warn("OVERFLOW"),
}


def _corr_row(config, s_req, s_const, variant, check, max_err, rel_err, atol, status):
    return (
        f"{config:>30}  {s_req:>5}  {s_const:>3}  {variant:>8}  {check:>12}  "
        f"{max_err:>10.2e}  {rel_err:>9.2e}  {atol:>8.0e}  {_RESULT_LABEL[status]()}"
    )


###
### Correctness — Triton kernel vs ground truths only.
###
### Variant is always 'triton'; references are the independent ground truths:
###   exp  = torch.linalg.matrix_exp
###   quad = Gauss-Legendre integral ground truth for psi
###   aug  = autograd through matrix_exp of the explicit 2N augmented matrix
_CORR_HDR = (
    f"{'Config':>30}  {'s_req':>5}  {'S':>3}  {'Variant':>8}  {'Check':>12}  "
    f"{'MaxErr':>10}  {'RelErr':>9}  {'atol':>8}  Result"
)
_CORR_SEP = "-" * 109


def _corr_block(A: torch.Tensor, cfg_str: str, s_req: int, squarings: Sequence[int],
                atol_f: float, atol_b: float, all_passed: bool,
                csv_rows: list[dict]) -> bool:
    """Run kernel-vs-ground-truth correctness checks on one A, ablating the
    constexpr S over `squarings`.

    The ground truths (exp, phi_1 quadrature, augmented-matrix grad) depend only
    on A, not S, so they are computed once and reused across the S sweep.  Each S
    then shows accuracy vs s_req: S >= s_req is well-resolved, S < s_req
    under-squares (inaccurate), S > s_req over-scales (precision loss).
    """
    gt_E   = ref_torch_matrix_exp(A)
    gt_psi = ref_phi1_quadrature(A)
    g_aug  = ref_grad_matrix_exp_aug(A).to(A.dtype)

    def _emit(S, variant, check, got, ref, atol, acc):
        status, err, rel = _check_status(got, ref, atol)
        ### Overflow (non-finite) is not a correctness signal; only a genuine
        ### 'fail' clears all_passed.
        acc = acc and (status != "fail")
        logger.info(_corr_row(cfg_str, s_req, S, variant, check, err, rel, atol, status))
        csv_rows.append({"config": cfg_str, "s_req": s_req, "S": S, "variant": variant,
                         "check": check, "max_err": err, "rel_err": rel,
                         "atol": atol, "result": status})
        return acc

    for S in squarings:
        got_E, got_psi = expm_t18_block_triton(A, S=S)

        ### Backward: dL/dA for L = E.sum() + psi.sum() at this S.
        A_t = A.detach().clone().requires_grad_(True)
        E_t, psi_t = expm_t18_block_triton(A_t, S=S)
        (E_t.sum() + psi_t.sum()).backward()

        all_passed = _emit(S, "triton", "vs linalg.matrix_exp",    got_E,    gt_E,   atol_f, all_passed)
        all_passed = _emit(S, "triton", "vs Gauss-Legendre   ", got_psi,  gt_psi, atol_f, all_passed)
        all_passed = _emit(S, "triton", "vs autograd         ", A_t.grad, g_aug,  atol_b, all_passed)

    return all_passed


def run_correctness(
    ns: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
    norms: Sequence[float],
    squarings: Sequence[int],
) -> tuple[bool, list[dict]]:
    logger.info("\n" + bold("=" * 92) + "\n"
                + bold("  CORRECTNESS — norm x S sweep  "
                       "(s_req = S a norm needs; S = constexpr passed; S<s_req under-resolves)")
                + "\n" + bold("=" * 92) + "\n" + _CORR_HDR + "\n" + _CORR_SEP)

    all_passed = True
    csv_rows: list[dict] = []

    for dtype_name in dtypes:
        dtype = _dtype(dtype_name)
        ### Backward gradients scale with ||exp(A)|| (larger than the forward
        ### outputs), so atol_b is loosened vs atol_f, as in expm_force_bench.
        atol_f = 1e-3 if dtype == torch.float32 else 5e-2
        atol_b = 5e-3 if dtype == torch.float32 else 1e-1

        for norm, B, N in product(norms, bs, ns):
            A = _make_large_norm_A(B, N, dtype, norm)
            s_req = _squarings_for_norm(norm)
            cfg_str = f"|A|={norm:g} B={B} N={N} {dtype_name}"
            all_passed = _corr_block(A, cfg_str, s_req, squarings, atol_f, atol_b,
                                     all_passed, csv_rows)

        logger.info(_CORR_SEP)

    if all_passed:
        print(ok("All correctness checks passed."))
    else:
        print(fail("One or more correctness checks FAILED "
                   "(expected where S < s_req, or intrinsic T18/dtype precision loss at large norms)."))
    return all_passed, csv_rows


###
### Performance — Triton kernel as a function of the constexpr S.
###
### S is the number of scaling-and-squaring steps actually issued by the kernel,
### so runtime scales with it (each S = one real block-squaring). Held at a fixed
### representative norm since runtime is norm-independent; s_req is that norm's
### accuracy requirement, shown only for context.
_PERF_HDR = (
    f"{'Config':>22}  {'|A|':>8}  {'s_req':>5}  {'S':>3}  {'dtype':>6}  {'dir':>3}  "
    f"{'triton ms':>11}  {'Tri p99':>9}  {'Tri var':>9}  {'Tri mean':>9}"
)
_PERF_SEP = "-" * 104


def _perf_row(config, norm, s_req, s_const, dtype_name, direction, s):
    return (
        f"{config:>22}  {norm:>8g}  {s_req:>5}  {s_const:>3}  {dtype_name:>6}  {direction:>3}  "
        f"{s.median:>11.3f}  {s.p99:>9.3f}  {s.var:>9.2e}  {s.mean:>9.3f}"
    )


def run_perf(
    ns: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
    norms: Sequence[float],
    squarings: Sequence[int],
    warmup: int = 25,
    rep: int = 200,
    fwd: bool = True,
    bwd: bool = False,
) -> list[dict]:
    logger.info("\n" + bold("=" * 104) + "\n"
                + bold("  PERFORMANCE — Triton kernel vs constexpr S (fixed representative norm)")
                + "\n" + bold("=" * 104) + "\n" + _PERF_HDR + "\n" + _PERF_SEP)

    csv_rows: list[dict] = []
    ### Runtime is norm-independent (S alone drives the squarings), so fix one
    ### representative norm and sweep the constexpr S instead.
    rep_norm = max(norms)
    s_req    = _squarings_for_norm(rep_norm)

    for dtype_name in dtypes:
        dtype = _dtype(dtype_name)

        for N, B in product(ns, bs):
            cfg_str = f"B={B} N={N}"
            A = _make_large_norm_A(B, N, dtype, rep_norm)

            for S_const in squarings:
                if fwd:
                    s_tri = bench_stats(lambda: expm_t18_block_triton(A, S=S_const), warmup, rep)
                    logger.info(_perf_row(cfg_str, rep_norm, s_req, S_const, dtype_name, "fwd", s_tri))
                    csv_rows.append({
                        "config": cfg_str, "norm": rep_norm, "s_req": s_req, "S": S_const,
                        "dtype": dtype_name, "dir": "fwd",
                        **stat_fields("blk_triton", s_tri),
                    })

                if bwd:
                    A_g = A.detach().clone().requires_grad_(True)

                    def _b_tri(A_g=A_g, S_const=S_const):
                        A_g.grad = None
                        E, psi = expm_t18_block_triton(A_g, S=S_const)
                        (E.sum() + psi.sum()).backward()

                    s_b_tri = bench_stats(_b_tri, warmup, rep)
                    logger.info(_perf_row(cfg_str, rep_norm, s_req, S_const, dtype_name, "bwd", s_b_tri))
                    csv_rows.append({
                        "config": cfg_str, "norm": rep_norm, "s_req": s_req, "S": S_const,
                        "dtype": dtype_name, "dir": "bwd",
                        **stat_fields("blk_triton", s_b_tri),
                    })

            logger.info("")  # blank between (N, B) groups

        logger.info(_PERF_SEP)
    logger.info("")
    return csv_rows


###
### Entry point
###
def main():
    parser = argparse.ArgumentParser(
        description="expm_t18_block_triton S-ablation benchmark (norm sweep)")
    parser.add_argument(
        "--mode", choices=["correctness", "perf", "all"], default="all",
        help="Which sections to run (default: all)",
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[4, 8, 16],
        metavar="N", help="N values to benchmark (default: 4 8 16)",
    )
    parser.add_argument(
        "--b", type=int, nargs="+", default=[8192, 16384, 32768, 65536],
        metavar="B", help="batch sizes",
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], nargs="+",
        default=["fp32", "bf16"], metavar="DTYPE",
        help="dtypes to test (default: fp32 bf16)",
    )
    parser.add_argument(
        "--norms", type=float, nargs="+",
        default=[0.1, 1.0, 4.0, 10.0, 20.0], metavar="NORM",
        help="target ||A||_1 values for the correctness sweep; each has s_req = "
             "ceil(log2(||A||_1/theta_18)) (default: 0.1 1.0 10.0 100.0 1000.0)",
    )
    parser.add_argument(
        "--squarings", type=int, nargs="+", default=[0, 1, 2, 4], metavar="S",
        help="constexpr S values for the perf sweep — the kernel scales by 2^(-S) "
             "and squares exactly S times, so runtime scales with S. S < s_req of a "
             "norm under-resolves it (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--warmup", type=int, default=24,
        help="Triton do_bench warmup iterations (default: 24)",
    )
    parser.add_argument(
        "--rep", type=int, default=128,
        help="Triton do_bench repetitions (default: 128)",
    )
    parser.add_argument(
        "--fwd", action="store_true", default=False,
        help="Benchmark forward (default when neither --fwd nor --bwd is given)",
    )
    parser.add_argument(
        "--bwd", action="store_true", default=False,
        help="Benchmark backward",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Directory to write CSV reports (overrides default benchmark_reports/)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Print per-row correctness/perf tables (quiet by default)",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    run_fwd = args.fwd or not args.bwd
    run_bwd = args.bwd

    if not torch.cuda.is_available():
        print(fail("No CUDA device found. Exiting."))
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    norm_s = ", ".join(f"{nrm:g}(s_req={_squarings_for_norm(nrm)})" for nrm in args.norms)
    logger.info(f"\nDevice    : {dev}\nN vals    : {args.n}\nB vals    : {args.b}\n"
                f"dtypes    : {args.dtype}\nnorms     : {norm_s}\n"
                f"squarings : {args.squarings}  (constexpr S, perf sweep)\n"
                f"default S : {_MAX_S}\nbench fwd : {run_fwd}\nbench bwd : {run_bwd}")

    today = date.today().strftime("%Y_%m_%d")
    if run_fwd and run_bwd:
        dir_tag = "fwdbwd"
    elif run_bwd:
        dir_tag = "bwd"
    else:
        dir_tag = "fwd"
    report_dir = os.path.normpath(args.out_dir if args.out_dir else _REPORT_DIR)

    if args.mode in ("correctness", "all"):
        _, corr_rows = run_correctness(args.n, args.b, args.dtype, args.norms, args.squarings)
        corr_path = os.path.join(report_dir,
                                 f"benchmark_expm_norm_correctness_{today}.CSV")
        _write_csv(corr_rows, corr_path)

    if args.mode in ("perf", "all"):
        perf_rows = run_perf(args.n, args.b, args.dtype, norms=args.norms,
                             squarings=args.squarings,
                             warmup=args.warmup, rep=args.rep,
                             fwd=run_fwd, bwd=run_bwd)
        perf_path = os.path.join(report_dir,
                                 f"benchmark_expm_norm_perf_{dir_tag}_{today}.CSV")
        _write_csv(perf_rows, perf_path)


if __name__ == "__main__":
    main()
