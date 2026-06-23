"""
Numerical correctness and performance benchmark for expm_t18_triton.

Usage
-----
# Run everything (default):
    python benchmarks/expm_bench.py

# Only correctness:
    python benchmarks/expm_bench.py --mode correctness

# Only performance:
    python benchmarks/expm_bench.py --mode perf

# Forward only (explicit):
    python benchmarks/expm_bench.py --mode perf --fwd

# Backward only:
    python benchmarks/expm_bench.py --mode perf --bwd

# Both forward and backward:
    python benchmarks/expm_bench.py --mode perf --fwd --bwd

# Restrict to specific N values:
    python benchmarks/expm_bench.py --n 4 8

# Only bf16:
    python benchmarks/expm_bench.py --dtype bf16

Requirements: CUDA GPU, triton, torch.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from itertools import product
from typing import Sequence

import torch
import torch._logging

# torch._logging.set_logs(
#     dynamo=logging.ERROR,
#     aot=logging.ERROR,
#     inductor=logging.ERROR,
#     autotuning=False
# )

from hyperconnections.ops import expm_t18 as _expm_t18, expm_t18_triton

from bench_utils import ok, fail, warn, bold, _dtype, _corr_row, bench_stats, stat_fields, logger, setup_logging
from expm_common import (_REPORT_DIR, _write_csv, _A_FACTORIES, _A_LABEL,
                         _make_large_norm_A, ref_torch_matrix_exp, _check)


###
### Reference implementations
###
def ref_torch_matrix_exp_backward(A: torch.Tensor):
    """Return grad_A from .sum().backward() through torch.linalg.matrix_exp (fp32)."""
    A_r = A.detach().float().requires_grad_(True)
    torch.linalg.matrix_exp(A_r).sum().backward()
    return A_r.grad


### Compile eager torch expm_t18
torch._dynamo.config.cache_size_limit = 12
EXPMT18_COMPILE_MODE = os.environ.get("EXPMT18_COMPILE_MODE", "max-autotune")
expm_t18 = torch.compile(_expm_t18, mode=EXPMT18_COMPILE_MODE, fullgraph=False, dynamic=False)

###
### Correctness checks
###
_CORR_HDR = f"{'Config':>34}  {'Variant':>10}  {'Check':>10}  {'MaxErr':>10}  {'atol':>8}  Result"
_CORR_SEP = "-" * 92


def _corr_block(A: torch.Tensor, cfg_str: str, dtype: torch.dtype,
                atol_f: float, atol_b: float, all_passed: bool,
                csv_rows: list[dict]) -> bool:
    """Run fwd+bwd correctness on one A; return updated all_passed."""

    ### forward — Triton vs ground-truth (torch.linalg.matrix_exp)
    got = expm_t18_triton(A)
    ref = ref_torch_matrix_exp(A)
    passed, err = _check(got, ref, atol_f)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "triton", "fwd", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "fwd",
                     "max_err": err, "atol": atol_f, "passed": passed})

    ### forward — Triton vs compiled T18 (same algorithm, smaller tolerance)
    ref_t18 = expm_t18(A)
    passed, err = _check(got, ref_t18, atol_f)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "vs T18", "fwd", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "vs T18", "check": "fwd",
                     "max_err": err, "atol": atol_f, "passed": passed})

    ### backward — grad_A from Triton vs from torch.linalg.matrix_exp
    A_t = A.detach().clone().requires_grad_(True)
    expm_t18_triton(A_t).sum().backward()
    grad_ref = ref_torch_matrix_exp_backward(A)

    passed, err = _check(A_t.grad, grad_ref, atol_b)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "triton", "grad_A", err, atol_b, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "grad_A",
                     "max_err": err, "atol": atol_b, "passed": passed})

    return all_passed


def run_correctness(
    ns: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
) -> tuple[bool, list[dict]]:
    logger.info("\n" + bold("=" * 92) + "\n" + bold("  CORRECTNESS — random A (small norm, no/few squarings)") + "\n" + bold("=" * 92) + "\n" + _CORR_HDR + "\n" + _CORR_SEP)

    all_passed = True
    csv_rows: list[dict] = []

    for dtype_name in dtypes:
        dtype  = _dtype(dtype_name)
        ### bf16/fp16: round-off floor dominates; widen forward atol.  Backward
        ### uses an fp32 augmented kernel internally, then casts at the end —
        ### so its rounding profile matches input dtype.
        atol_f = 5e-4 if dtype == torch.float32 else 5e-2
        atol_b = 5e-3 if dtype == torch.float32 else 1e-1

        for B, N in product(bs, ns):
            A = _A_FACTORIES["random"](B, N, dtype)
            cfg_str = f"B={B} N={N} {dtype_name}"
            all_passed = _corr_block(A, cfg_str, dtype, atol_f, atol_b, all_passed, csv_rows)

        logger.info(_CORR_SEP)

    ### Structured-A correctness
    logger.info("\n" + bold("=" * 92) + "\n" + bold("  CORRECTNESS — structured A (skew / neg_psd / diagonal / large-norm)") + "\n" + bold("=" * 92) + "\n" + _CORR_HDR + "\n" + _CORR_SEP)

    for dtype_name in dtypes:
        dtype  = _dtype(dtype_name)
        atol_f = 1e-3 if dtype == torch.float32 else 5e-2
        atol_b = 5e-3 if dtype == torch.float32 else 1e-1

        for kind, B, N in product(["skew", "neg_psd", "diagonal", "large"], bs, ns):
            A = _A_FACTORIES[kind](B, N, dtype)
            cfg_str = f"[{_A_LABEL[kind]}] B={B} N={N} {dtype_name}"
            all_passed = _corr_block(A, cfg_str, dtype, atol_f, atol_b, all_passed, csv_rows)

        logger.info(_CORR_SEP)

    if all_passed:
        print(ok("All correctness checks passed."))
    else:
        print(fail("One or more correctness checks FAILED."))
    return all_passed, csv_rows


###
### Performance benchmark
###
_PERF_HDR = (
    f"{'Config':>26}  {'AType':>9}  {'dtype':>6}  "
    f"{'Triton ms':>10}  {'Tri p99':>9}  {'Tri var':>9}  "
    f"{'matrix_exp ms':>14}  {'T18 ms':>7}  "
    f"{'vs torch':>9}  {'vs T18':>7}"
)
_PERF_SEP = "-" * 130


def _perf_row(config, atype, dtype_name, s_tri, s_torch, s_t18):
    def _sp(t_ref):
        sp = t_ref / s_tri.median
        s = f"{sp:.2f}x"
        return ok(s) if sp >= 1.05 else (warn(s) if sp >= 0.95 else fail(s))
    return (
        f"{config:>26}  {atype:>9}  {dtype_name:>6}  "
        f"{s_tri.median:>10.3f}  {s_tri.p99:>9.3f}  {s_tri.var:>9.2e}  "
        f"{s_torch.median:>14.3f}  {s_t18.median:>7.3f}  "
        f"{_sp(s_torch.median):>9}  {_sp(s_t18.median):>7}"
    )


def run_perf(
    ns: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
    norms: Sequence[float] = (1.0, 3.0, 5.0),
    warmup: int = 25,
    rep: int = 200,
    fwd: bool = True,
    bwd: bool = False,
) -> list[dict]:
    logger.info("\n" + bold("=" * 130) + "\n" + bold("  PERFORMANCE — random / skew / neg_psd / diagonal / large-norm (per --norms)") + "\n" + bold("=" * 130) + "\n" + _PERF_HDR + "\n" + _PERF_SEP)

    std_kinds = ["random", "skew", "neg_psd", "diagonal"]
    csv_rows: list[dict] = []

    for dtype_name in dtypes:
        dtype = _dtype(dtype_name)

        for N, B in product(ns, bs):
            cfg_str = f"B={B} N={N}"
            ### (label, A, norm) jobs: standard kinds + one large-norm row per --norms value
            jobs = [(_A_LABEL[k], _A_FACTORIES[k](B, N, dtype), None) for k in std_kinds]
            jobs += [(f"lrg{nrm:g}", _make_large_norm_A(B, N, dtype, nrm), nrm) for nrm in norms]

            for label, A, norm in jobs:
                if fwd:
                    s_tri   = bench_stats(lambda: expm_t18_triton(A),      warmup, rep)
                    s_torch = bench_stats(lambda: ref_torch_matrix_exp(A), warmup, rep)
                    s_t18   = bench_stats(lambda: expm_t18(A),             warmup, rep)
                    logger.info(_perf_row(cfg_str, label + " fwd", dtype_name, s_tri, s_torch, s_t18))
                    csv_rows.append({
                        "config": cfg_str, "atype": label + " fwd", "dtype": dtype_name, "norm": norm,
                        **stat_fields("triton", s_tri), **stat_fields("matrix_exp", s_torch),
                        **stat_fields("t18", s_t18),
                        "speedup_vs_torch": s_torch.median / s_tri.median,
                        "speedup_vs_t18":   s_t18.median / s_tri.median,
                    })

                if bwd:
                    A_g = A.detach().clone().requires_grad_(True)

                    def _b_tri():
                        A_g.grad = None
                        expm_t18_triton(A_g).sum().backward()

                    def _b_torch():
                        A_g.grad = None
                        torch.linalg.matrix_exp(A_g).sum().backward()

                    def _b_t18():
                        A_g.grad = None
                        expm_t18(A_g).sum().backward()

                    s_b_tri   = bench_stats(_b_tri,   warmup, rep)
                    s_b_torch = bench_stats(_b_torch, warmup, rep)
                    s_b_t18   = bench_stats(_b_t18,   warmup, rep)
                    logger.info(_perf_row(cfg_str, label + " bwd", dtype_name, s_b_tri, s_b_torch, s_b_t18))
                    csv_rows.append({
                        "config": cfg_str, "atype": label + " bwd", "dtype": dtype_name, "norm": norm,
                        **stat_fields("triton", s_b_tri), **stat_fields("matrix_exp", s_b_torch),
                        **stat_fields("t18", s_b_t18),
                        "speedup_vs_torch": s_b_torch.median / s_b_tri.median,
                        "speedup_vs_t18":   s_b_t18.median / s_b_tri.median,
                    })

            logger.info("")  # blank between (N, B) groups

        logger.info(_PERF_SEP)
    logger.info("")
    return csv_rows


###
### Entry point
###
def main():
    parser = argparse.ArgumentParser(description="expm_t18_triton benchmark")
    parser.add_argument(
        "--mode", choices=["correctness", "perf", "all"], default="all",
        help="Which sections to run (default: all)",
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[4, 8, 16],
        metavar="N", help="N values to benchmark (default: 4 8 16)",
    )
    parser.add_argument(
        "--b", type=int, nargs="+", default=[256, 1024, 4096],
        metavar="B", help="batch sizes (default: 256 1024 4096)",
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], nargs="+",
        default=["fp32", "bf16"], metavar="DTYPE",
        help="dtypes to test (default: fp32 bf16)",
    )
    parser.add_argument(
        "--norms", type=float, nargs="+", default=[1.0, 3.0, 5.0], metavar="NORM",
        help="target ||A||_1 values for the large-norm perf case; one row each "
             "(default: 1.0 3.0 5.0)",
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
        help="Benchmark fwd+bwd",
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
    logger.info(f"\nDevice    : {dev}\nN vals    : {args.n}\nB vals    : {args.b}\ndtypes    : {args.dtype}\nbench fwd : {run_fwd}\nbench bwd : {run_bwd}")

    today = date.today().strftime("%Y_%m_%d")
    if run_fwd and run_bwd:
        dir_tag = "fwdbwd"
    elif run_bwd:
        dir_tag = "bwd"
    else:
        dir_tag = "fwd"
    if len(list(args.n)) == 1:
        dir_tag += f"_n{list(args.n)[0]}"

    report_dir = os.path.normpath(args.out_dir if args.out_dir else _REPORT_DIR)

    passed = True
    if args.mode in ("correctness", "all"):
        ### Correctness uses smaller B to keep the table compact
        passed, corr_rows = run_correctness(args.n, args.b, args.dtype)
        corr_path = os.path.join(report_dir,
                                 f"benchmark_expm_correctness_fwdbwd_{today}.CSV")
        _write_csv(corr_rows, corr_path)

    if args.mode in ("perf", "all"):
        perf_rows = run_perf(args.n, args.b, args.dtype, norms=args.norms,
                             warmup=args.warmup, rep=args.rep, fwd=run_fwd, bwd=run_bwd)
        perf_path = os.path.join(report_dir,
                                 f"benchmark_expm_perf_{dir_tag}_{today}.CSV")
        _write_csv(perf_rows, perf_path)


    # if args.mode in ("correctness", "all") and not passed:
    #     sys.exit(1)


if __name__ == "__main__":
    main()
