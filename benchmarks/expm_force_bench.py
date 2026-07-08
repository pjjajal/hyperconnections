"""
Numerical correctness and performance benchmark for expm_t18_block_triton.

Benchmarks the augmented T18 matrix exponential that computes (exp(A), phi_1(A))
together without materialising the 2N x 2N augmented matrix.

Usage
-----
# Run everything (default):
    python benchmarks/expm_force_bench.py

# Only correctness:
    python benchmarks/expm_force_bench.py --mode correctness

# Only performance:
    python benchmarks/expm_force_bench.py --mode perf

# Restrict to specific N values:
    python benchmarks/expm_force_bench.py --n 4 8

# Only bf16:
    python benchmarks/expm_force_bench.py --dtype bf16

Requirements: CUDA GPU, triton, torch.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from itertools import product
from typing import Sequence

import numpy as np

import torch
import torch._logging

from hyperconnections.ops import expm_t18_augmented_sparse as _expm_t18_augmented_sparse, expm_t18_block_triton

from bench_utils import (ok, fail, warn, bold, _dtype, _corr_row as _corr_row_base,
                         bench_stats, stat_fields, logger, setup_logging)
from expm_common import (_REPORT_DIR, _write_csv, _A_FACTORIES, _A_LABEL,
                         _make_large_norm_A, ref_torch_matrix_exp, _check)


def _corr_row(*args, **kwargs):
    """expm_force_bench uses a wider Check column ('quantity vs reference')."""
    return _corr_row_base(*args, check_width=12, **kwargs)


###
### Reference implementations (file-specific ground truths for psi / grad)
###
def ref_phi1_quadrature(A: torch.Tensor, n_nodes: int = 64) -> torch.Tensor:
    """Ground-truth phi_1(A) = integral_0^1 exp(theta A) d theta.

    Gauss-Legendre quadrature in float64. exp(theta A) is entire in theta,
    so GL converges super-algebraically; 64 nodes reach fp64 machine
    precision. Each node uses torch.linalg.matrix_exp (ground-truth exp).
    Independent of the T18 polynomial, so comparing both the PyTorch and
    Triton psi against it isolates algorithm error from kernel error.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    A64 = A.double()
    psi = torch.zeros_like(A64)
    for x, w in zip(nodes, weights):
        theta = 0.5 * (x + 1.0) # [-1,1] -> [0,1]
        psi += (0.5 * w) * torch.linalg.matrix_exp(theta * A64)
    return psi.to(A.dtype)


def _build_aug_2N(A: torch.Tensor) -> torch.Tensor:
    """[[A, I], [0, 0]] of shape [B, 2N, 2N], preserving autograd through A."""
    B_, N_, _ = A.shape
    eye  = torch.eye(N_, dtype=A.dtype, device=A.device).expand(B_, N_, N_).contiguous()
    zero = torch.zeros_like(eye)
    top = torch.cat([A,    eye ], dim=-1)        # [B, N, 2N]
    bot = torch.cat([zero, zero], dim=-1)        # [B, N, 2N]
    return torch.cat([top, bot], dim=-2)         # [B, 2N, 2N]


def ref_grad_matrix_exp_aug(A: torch.Tensor) -> torch.Tensor:
    """Ground-truth dL/dA for L = E.sum() + psi.sum() where (E, psi) =
    upper-row blocks of matrix_exp([[A, I], [0, 0]]).  fp32; cast at call site.

    Independent of the T18 polynomial — uses torch.linalg.matrix_exp as the
    reference and autograd's adjoint through the explicit 2N x 2N construction.
    Analogous to ref_phi1_quadrature for the forward.
    """
    A_r = A.detach().float().requires_grad_(True)
    M = _build_aug_2N(A_r)
    expM = torch.linalg.matrix_exp(M)
    expM[:, :A_r.shape[-1], :].sum().backward() # E.sum() + psi.sum()
    return A_r.grad


torch._dynamo.config.cache_size_limit = 12
EXPMT18_COMPILE_MODE = os.environ.get("EXPMT18_COMPILE_MODE", "max-autotune")
expm_t18_augmented_sparse = torch.compile(_expm_t18_augmented_sparse,fullgraph=False,mode=EXPMT18_COMPILE_MODE)


###
### Correctness checks
###
### Variant = implementation under test.  Check names the reference:
###   "vs linalg.matrix_exp" = torch.linalg.matrix_exp        (ground-truth E)
###   "vs Gauss-Legendre"    = GL integral ground truth for psi
###   "vs autograd"          = autograd through matrix_exp of the explicit 2N
###                            augmented matrix (ground-truth grad, T18-independent)
###   "<quantity> vs T18"    = pure-torch expm_t18_augmented_sparse (same algorithm);
###                            quantity kept since E/psi/grad share this one reference.
### The three ground-truth names match benchmarks/expm_norm_bench.py verbatim.
_CORR_HDR = f"{'Config':>34}  {'Variant':>10}  {'Check':>12}  {'MaxErr':>10}  {'atol':>8}  Result"
_CORR_SEP = "-" * 94


def _corr_block(A: torch.Tensor, cfg_str: str, dtype: torch.dtype,
                atol_f: float, atol_b: float, all_passed: bool,
                csv_rows: list[dict]) -> bool:
    """Run correctness checks on one A; return updated all_passed."""

    got_E, got_psi = expm_t18_block_triton(A)
    ref_E, ref_psi = expm_t18_augmented_sparse(A)
    gt_E   = ref_torch_matrix_exp(A)
    gt_psi = ref_phi1_quadrature(A)

    ### triton E vs exp ground truth (torch.linalg.matrix_exp)
    passed, err, rel = _check(got_E, gt_E, atol_f)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "triton", "vs linalg.matrix_exp", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "vs linalg.matrix_exp",
                     "max_err": err, "rel_err": rel, "atol": atol_f, "passed": passed})

    ### triton E vs same-algorithm pure-torch reference
    passed, err, rel = _check(got_E, ref_E, atol_f)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "triton", "E vs T18", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "E vs T18",
                     "max_err": err, "rel_err": rel, "atol": atol_f, "passed": passed})

    ### triton psi vs same-algorithm pure-torch reference
    passed, err, rel = _check(got_psi, ref_psi, atol_f)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "triton", "psi vs T18", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "psi vs T18",
                     "max_err": err, "rel_err": rel, "atol": atol_f, "passed": passed})

    ### triton psi vs quadrature ground truth — independent of T18
    passed, err, rel = _check(got_psi, gt_psi, atol_f)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "triton", "vs Gauss-Legendre", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "vs Gauss-Legendre",
                     "max_err": err, "rel_err": rel, "atol": atol_f, "passed": passed})

    ### pytorch-T18 psi vs quadrature ground truth — isolates algorithm error
    passed, err, rel = _check(ref_psi, gt_psi, atol_f)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "torch", "vs Gauss-Legendre", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "torch", "check": "vs Gauss-Legendre",
                     "max_err": err, "rel_err": rel, "atol": atol_f, "passed": passed})

    ### --- Backward: dL/dA for L = E.sum() + psi.sum() ---
    A_t = A.detach().clone().requires_grad_(True)
    E_t, psi_t = expm_t18_block_triton(A_t)
    (E_t.sum() + psi_t.sum()).backward()
    g_tri = A_t.grad

    A_r = A.detach().clone().requires_grad_(True)
    E_r, psi_r = expm_t18_augmented_sparse(A_r)
    (E_r.sum() + psi_r.sum()).backward()
    g_torch = A_r.grad

    g_aug = ref_grad_matrix_exp_aug(A).to(A.dtype)

    ### triton grad vs same-algorithm pure-torch reference
    passed, err, rel = _check(g_tri, g_torch, atol_b)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "triton", "grad vs T18", err, atol_b, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "grad vs T18",
                     "max_err": err, "rel_err": rel, "atol": atol_b, "passed": passed})

    ### triton grad vs augmented-matrix ground truth — independent of T18
    passed, err, rel = _check(g_tri, g_aug, atol_b)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "triton", "vs autograd", err, atol_b, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "vs autograd",
                     "max_err": err, "rel_err": rel, "atol": atol_b, "passed": passed})

    ### pytorch-T18 grad vs augmented-matrix ground truth — isolates algorithm error
    passed, err, rel = _check(g_torch, g_aug, atol_b)
    all_passed &= passed
    logger.info(_corr_row(cfg_str, "torch", "vs autograd", err, atol_b, passed))
    csv_rows.append({"config": cfg_str, "variant": "torch", "check": "vs autograd",
                     "max_err": err, "rel_err": rel, "atol": atol_b, "passed": passed})

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
        ### Backward gradients have larger magnitude than the outputs they came
        ### from (||L|| scales with ||exp(A)||), so atol_b is loosened vs atol_f.
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
### Speedup columns name the comparator they divide by, matching the
### timing columns: 'vs matrix_exp' = matrix_exp ms / blk_triton ms, etc.
_PERF_HDR = (
    f"{'Config':>26}  {'AType':>9}  {'dtype':>6}  "
    f"{'blk_triton ms':>14}  {'Tri p99':>9}  {'Tri var':>9}  "
    f"{'matrix_exp ms':>14}  {'blk_t18 ms':>11}  "
    f"{'vs matrix_exp':>13}  {'vs blk_t18':>10}"
)
_PERF_SEP = "-" * 135


def _perf_row(config, atype, dtype_name, s_tri, s_torch, s_t18):
    def _sp(t_ref):
        sp = t_ref / s_tri.median
        s = f"{sp:.2f}x"
        return ok(s) if sp >= 1.05 else (warn(s) if sp >= 0.95 else fail(s))
    return (
        f"{config:>26}  {atype:>9}  {dtype_name:>6}  "
        f"{s_tri.median:>14.3f}  {s_tri.p99:>9.3f}  {s_tri.var:>9.2e}  "
        f"{s_torch.median:>14.3f}  {s_t18.median:>11.3f}  "
        f"{_sp(s_torch.median):>13}  {_sp(s_t18.median):>10}"
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
    logger.info("\n" + bold("=" * 135) + "\n" + bold("  PERFORMANCE — random / skew / neg_psd / diagonal / large-norm (per --norms)") + "\n" + bold("=" * 135) + "\n" + _PERF_HDR + "\n" + _PERF_SEP)

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
                    s_tri   = bench_stats(lambda: expm_t18_block_triton(A),     warmup, rep)
                    s_torch = bench_stats(lambda: ref_torch_matrix_exp(A),      warmup, rep)
                    s_t18   = bench_stats(lambda: expm_t18_augmented_sparse(A), warmup, rep)
                    logger.info(_perf_row(cfg_str, label + " fwd", dtype_name, s_tri, s_torch, s_t18))
                    csv_rows.append({
                        "config": cfg_str, "atype": label + " fwd", "dtype": dtype_name, "norm": norm,
                        **stat_fields("blk_triton", s_tri), **stat_fields("matrix_exp", s_torch),
                        **stat_fields("blk_t18", s_t18),
                        "speedup_vs_torch": s_torch.median / s_tri.median,
                        "speedup_vs_t18":   s_t18.median / s_tri.median,
                    })

                if bwd:
                    A_g = A.detach().clone().requires_grad_(True)

                    def _b_tri():
                        A_g.grad = None
                        E, psi = expm_t18_block_triton(A_g)
                        (E.sum() + psi.sum()).backward()

                    def _b_aug():
                        ### Ground-truth: build [[A, I], [0, 0]] and backprop through
                        ### matrix_exp.  E.sum() + psi.sum() = expM[:, :N, :].sum().
                        A_g.grad = None
                        torch.linalg.matrix_exp(_build_aug_2N(A_g))[:, :N, :].sum().backward()

                    def _b_t18():
                        A_g.grad = None
                        E, psi = expm_t18_augmented_sparse(A_g)
                        (E.sum() + psi.sum()).backward()

                    s_b_tri = bench_stats(_b_tri, warmup, rep)
                    s_b_aug = bench_stats(_b_aug, warmup, rep)
                    s_b_t18 = bench_stats(_b_t18, warmup, rep)
                    logger.info(_perf_row(cfg_str, label + " bwd", dtype_name, s_b_tri, s_b_aug, s_b_t18))
                    csv_rows.append({
                        "config": cfg_str, "atype": label + " bwd", "dtype": dtype_name, "norm": norm,
                        **stat_fields("blk_triton", s_b_tri), **stat_fields("matrix_exp", s_b_aug),
                        **stat_fields("blk_t18", s_b_t18),
                        "speedup_vs_torch": s_b_aug.median / s_b_tri.median,
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
    parser = argparse.ArgumentParser(description="expm_t18_block_triton benchmark")
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
    logger.info(f"\nDevice    : {dev}\nN vals    : {args.n}\nB vals    : {args.b}\ndtypes    : {args.dtype}\nbench fwd : {run_fwd}\nbench bwd : {run_bwd}")

    today = date.today().strftime("%Y_%m_%d")
    if run_fwd and run_bwd:
        dir_tag = "fwdbwd"
    elif run_bwd:
        dir_tag = "bwd"
    else:
        dir_tag = "fwd"
    report_dir = os.path.normpath(args.out_dir if args.out_dir else _REPORT_DIR)

    passed = True
    if args.mode in ("correctness", "all"):
        passed, corr_rows = run_correctness(args.n, args.b, args.dtype)
        corr_path = os.path.join(report_dir,
                                 f"benchmark_expm_force_correctness_{today}.CSV")
        _write_csv(corr_rows, corr_path)

    if args.mode in ("perf", "all"):
        perf_rows = run_perf(args.n, args.b, args.dtype, norms=args.norms,
                             warmup=args.warmup, rep=args.rep,
                             fwd=run_fwd, bwd=run_bwd)
        perf_path = os.path.join(report_dir,
                                 f"benchmark_expm_force_perf_{dir_tag}_{today}.CSV")
        _write_csv(perf_rows, perf_path)

    # if args.mode in ("correctness", "all") and not passed:
    #     sys.exit(1)


if __name__ == "__main__":
    main()
