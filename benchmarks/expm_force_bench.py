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
import csv
import os
import sys
from datetime import date
from itertools import product
from typing import Sequence

import numpy as np
import torch
import triton
import triton.testing

from hyperconnections.ops import expm_t18_augmented_sparse, expm_t18_block_triton as _expm_t18_block_triton

from bench_utils import DEVICE, ok, fail, warn, bold, _dtype, _corr_row as _corr_row_base


def _corr_row(*args, **kwargs):
    """expm_force_bench uses a wider Check column ('quantity vs reference')."""
    return _corr_row_base(*args, check_width=12, **kwargs)


###
### CSV export
###
_REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_reports")

def _write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> saved {path}")


###
### Input factories
###
def _make_random_A(B: int, N: int, dtype: torch.dtype, scale: float = 0.3, seed: int = 0):
    """Random matrix scaled to keep ||A||_1 <= theta_18 (no scaling+squaring)."""
    torch.manual_seed(seed)
    return torch.randn(B, N, N, device=DEVICE, dtype=dtype) * scale


def _make_skew_A(B: int, N: int, dtype: torch.dtype, scale: float = 0.5, seed: int = 1):
    """Skew-symmetric: M = (G - G^T)/2.  exp(M) is orthogonal."""
    torch.manual_seed(seed)
    G = torch.randn(B, N, N, device=DEVICE, dtype=dtype) * scale
    return 0.5 * (G - G.transpose(-1, -2))


def _make_neg_psd_A(B: int, N: int, dtype: torch.dtype, scale: float = 0.4, seed: int = 2):
    """Negative semi-definite: -R R^T (relevant to dissipative generators)."""
    torch.manual_seed(seed)
    R = torch.randn(B, N, N, device=DEVICE, dtype=dtype) * scale
    return -(R @ R.transpose(-1, -2))


def _make_diag_A(B: int, N: int, dtype: torch.dtype, scale: float = 0.5, seed: int = 3):
    """Diagonal A: exp(diag(d)) = diag(exp(d))."""
    torch.manual_seed(seed)
    d = torch.randn(B, N, device=DEVICE, dtype=dtype) * scale
    return torch.diag_embed(d)


def _make_large_norm_A(B: int, N: int, dtype: torch.dtype, scale: float = 10.0, seed: int = 4):
    """Norm-stress: ||A||_1 lands around 3-5, triggering s=1-2 squarings.

    Larger scales (||A||_1 -> 10+) drive ||exp(A)|| to thousands and cause
    intrinsic fp32/bf16 precision loss in *any* T18 implementation; that
    isn't a kernel bug, so we keep the stress test inside a regime where
    the comparison stays meaningful.
    """
    torch.manual_seed(seed)
    return torch.randn(B, N, N, device=DEVICE, dtype=dtype) * (scale / N ** 0.5)


_A_FACTORIES = {
    "random":   _make_random_A,
    "skew":     _make_skew_A,
    "neg_psd":  _make_neg_psd_A,
    "diagonal": _make_diag_A,
    "large":    _make_large_norm_A,
}

_A_LABEL = {
    "random":   "rand",
    "skew":     "skew",
    "neg_psd":  "npsd",
    "diagonal": "diag",
    "large":    "lrg ",
}


###
### Reference implementations
###
def ref_torch_matrix_exp(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_exp(A.float()).to(A.dtype)


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
        theta = 0.5 * (x + 1.0)          # [-1,1] -> [0,1]
        psi += (0.5 * w) * torch.linalg.matrix_exp(theta * A64)
    return psi.to(A.dtype)

expm_t18_block_triton = torch.compile(_expm_t18_block_triton,fullgraph=False,mode="max-autotune")
torch._dynamo.config.cache_size_limit = 12
EXPMT18_COMPILE_MODE = os.environ.get("EXPMT18_COMPILE_MODE", "max-autotune")

###
### Correctness checks
###
def _check(got: torch.Tensor, ref: torch.Tensor, atol: float) -> tuple[bool, float]:
    """Magnitude-aware check: passes if max_err <= atol * (1 + ||ref||_inf).

    A pure absolute tolerance is misleading for matrix-exp outputs, where
    ||exp(A)|| scales with A's norm.  Folding ref's magnitude in keeps
    the threshold meaningful across different input norms and dtypes.
    """
    diff    = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    ref_mag = ref.float().abs().max().item()
    return max_err <= atol * (1.0 + ref_mag), max_err


### Variant = implementation under test; Check = "<quantity> vs <reference>".
### References: exp = torch.linalg.matrix_exp, T18 = pure-torch
### expm_t18_augmented_sparse, quad = Gauss-Legendre integral ground truth.
_CORR_HDR = f"{'Config':>34}  {'Variant':>10}  {'Check':>12}  {'MaxErr':>10}  {'atol':>8}  Result"
_CORR_SEP = "-" * 94


def _corr_block(A: torch.Tensor, cfg_str: str, dtype: torch.dtype,
                atol_f: float, all_passed: bool,
                csv_rows: list[dict]) -> bool:
    """Run correctness checks on one A; return updated all_passed."""

    got_E, got_psi = expm_t18_block_triton(A)
    ref_E, ref_psi = expm_t18_augmented_sparse(A)
    gt_E   = ref_torch_matrix_exp(A)
    gt_psi = ref_phi1_quadrature(A)

    ### triton E vs exp ground truth (torch.linalg.matrix_exp)
    passed, err = _check(got_E, gt_E, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "triton", "E vs exp", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "E vs exp",
                     "max_err": err, "atol": atol_f, "passed": passed})

    ### triton E vs same-algorithm pure-torch reference
    passed, err = _check(got_E, ref_E, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "triton", "E vs T18", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "E vs T18",
                     "max_err": err, "atol": atol_f, "passed": passed})

    ### triton psi vs same-algorithm pure-torch reference
    passed, err = _check(got_psi, ref_psi, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "triton", "psi vs T18", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "psi vs T18",
                     "max_err": err, "atol": atol_f, "passed": passed})

    ### triton psi vs quadrature ground truth — independent of T18
    passed, err = _check(got_psi, gt_psi, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "triton", "psi vs quad", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "triton", "check": "psi vs quad",
                     "max_err": err, "atol": atol_f, "passed": passed})

    ### pytorch-T18 psi vs quadrature ground truth — isolates algorithm error
    passed, err = _check(ref_psi, gt_psi, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "torch", "psi vs quad", err, atol_f, passed))
    csv_rows.append({"config": cfg_str, "variant": "torch", "check": "psi vs quad",
                     "max_err": err, "atol": atol_f, "passed": passed})

    return all_passed


def run_correctness(
    ns: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
) -> tuple[bool, list[dict]]:
    print()
    print(bold("=" * 92))
    print(bold("  CORRECTNESS — random A (small norm, no/few squarings)"))
    print(bold("=" * 92))
    print(_CORR_HDR)
    print(_CORR_SEP)

    all_passed = True
    csv_rows: list[dict] = []

    for dtype_name in dtypes:
        dtype  = _dtype(dtype_name)
        atol_f = 5e-4 if dtype == torch.float32 else 5e-2

        for B, N in product(bs, ns):
            A = _A_FACTORIES["random"](B, N, dtype)
            cfg_str = f"B={B} N={N} {dtype_name}"
            all_passed = _corr_block(A, cfg_str, dtype, atol_f, all_passed, csv_rows)

        print(_CORR_SEP)

    ### Structured-A correctness
    print()
    print(bold("=" * 92))
    print(bold("  CORRECTNESS — structured A (skew / neg_psd / diagonal / large-norm)"))
    print(bold("=" * 92))
    print(_CORR_HDR)
    print(_CORR_SEP)

    for dtype_name in dtypes:
        dtype  = _dtype(dtype_name)
        atol_f = 1e-3 if dtype == torch.float32 else 5e-2

        for kind, B, N in product(["skew", "neg_psd", "diagonal", "large"], bs, ns):
            A = _A_FACTORIES[kind](B, N, dtype)
            cfg_str = f"[{_A_LABEL[kind]}] B={B} N={N} {dtype_name}"
            all_passed = _corr_block(A, cfg_str, dtype, atol_f, all_passed, csv_rows)

        print(_CORR_SEP)

    print()
    if all_passed:
        print(ok("All correctness checks passed."))
    else:
        print(fail("One or more correctness checks FAILED."))
    print()
    return all_passed, csv_rows


###
### Performance benchmark
###
### Speedup columns name the comparator they divide by, matching the
### timing columns: 'vs matrix_exp' = matrix_exp ms / blk_triton ms, etc.
_PERF_HDR = (
    f"{'Config':>26}  {'AType':>5}  {'dtype':>6}  "
    f"{'blk_triton ms':>14}  {'matrix_exp ms':>14}  {'blk_t18 ms':>11}  "
    f"{'vs matrix_exp':>13}  {'vs blk_t18':>10}"
)
_PERF_SEP = "-" * 115


def _perf_row(config, atype, dtype_name, t_tri, t_torch, t_t18):
    def _sp(t_ref):
        sp = t_ref / t_tri
        s = f"{sp:.2f}x"
        return ok(s) if sp >= 1.05 else (warn(s) if sp >= 0.95 else fail(s))
    return (
        f"{config:>26}  {atype:>5}  {dtype_name:>6}  "
        f"{t_tri:>14.3f}  {t_torch:>14.3f}  {t_t18:>11.3f}  "
        f"{_sp(t_torch):>13}  {_sp(t_t18):>10}"
    )


def run_perf(
    ns: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
    warmup: int = 25,
    rep: int = 200,
) -> list[dict]:
    print()
    print(bold("=" * 115))
    print(bold("  PERFORMANCE — random / skew / neg_psd / diagonal / large-norm"))
    print(bold("=" * 115))
    print(_PERF_HDR)
    print(_PERF_SEP)

    kinds = ["random", "skew", "neg_psd", "diagonal", "large"]
    csv_rows: list[dict] = []

    for dtype_name in dtypes:
        dtype = _dtype(dtype_name)

        for N, B in product(ns, bs):
            cfg_str = f"B={B} N={N}"
            for kind in kinds:
                A = _A_FACTORIES[kind](B, N, dtype)
                label = _A_LABEL[kind]

                t_tri   = triton.testing.do_bench(lambda: expm_t18_block_triton(A),    warmup=warmup, rep=rep)
                t_torch = triton.testing.do_bench(lambda: ref_torch_matrix_exp(A),     warmup=warmup, rep=rep)
                t_t18   = triton.testing.do_bench(lambda: expm_t18_augmented_sparse(A), warmup=warmup, rep=rep)
                print(_perf_row(cfg_str, label + " fwd", dtype_name, t_tri, t_torch, t_t18))
                csv_rows.append({
                    "config": cfg_str, "atype": label + " fwd", "dtype": dtype_name,
                    "blk_triton_ms": t_tri, "matrix_exp_ms": t_torch, "blk_t18_ms": t_t18,
                    "speedup_vs_torch": t_torch / t_tri, "speedup_vs_t18": t_t18 / t_tri,
                })

            print()  # blank between (N, B) groups

        print(_PERF_SEP)
    print()
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
        "--warmup", type=int, default=24,
        help="Triton do_bench warmup iterations (default: 24)",
    )
    parser.add_argument(
        "--rep", type=int, default=128,
        help="Triton do_bench repetitions (default: 128)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(fail("No CUDA device found. Exiting."))
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    print(f"\nDevice    : {dev}")
    print(f"N vals    : {args.n}")
    print(f"B vals    : {args.b}")
    print(f"dtypes    : {args.dtype}")

    today = date.today().strftime("%Y_%m_%d")
    report_dir = os.path.normpath(_REPORT_DIR)

    passed = True
    if args.mode in ("correctness", "all"):
        passed, corr_rows = run_correctness(args.n, args.b, args.dtype)
        corr_path = os.path.join(report_dir,
                                 f"benchmark_expm_force_correctness_{today}.CSV")
        _write_csv(corr_rows, corr_path)

    if args.mode in ("perf", "all"):
        perf_rows = run_perf(args.n, args.b, args.dtype,
                             warmup=args.warmup, rep=args.rep)
        perf_path = os.path.join(report_dir,
                                 f"benchmark_expm_force_perf_fwd_{today}.CSV")
        _write_csv(perf_rows, perf_path)

    if args.mode in ("correctness", "all") and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
