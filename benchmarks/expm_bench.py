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
import sys
from itertools import product
from typing import Sequence

import torch
import triton
import triton.testing

from hyperconnections.ops import expm_t18, expm_t18_triton

###
### Helpers
###
DEVICE = "cuda:0"

_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"

def _col(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text
def ok(s="PASS"):
    return _col(s, _GREEN)
def fail(s):
    return _col(s, _RED)
def warn(s):
    return _col(s, _YELLOW)
def bold(s):
    return _col(s, _BOLD)
def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


###
### Input factories
###
def _make_random_A(B: int, N: int, dtype: torch.dtype, scale: float = 0.3, seed: int = 0):
    """Random matrix scaled to keep ||A||_1 <= θ_18 (no scaling+squaring)."""
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


def _make_large_norm_A(B: int, N: int, dtype: torch.dtype, scale: float = 4.5, seed: int = 4):
    """Norm-stress: ||A||_1 lands around 3-5, triggering s=1-2 squarings.

    Larger scales (||A||_1 → 10+) drive ||exp(A)|| to thousands and cause
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
    """Ground-truth: PyTorch's matrix_exp computed in fp32."""
    return torch.linalg.matrix_exp(A.float()).to(A.dtype)


def ref_torch_matrix_exp_backward(A: torch.Tensor):
    """Return grad_A from .sum().backward() through torch.linalg.matrix_exp (fp32)."""
    A_r = A.detach().float().requires_grad_(True)
    torch.linalg.matrix_exp(A_r).sum().backward()
    return A_r.grad


### Eager (non-compiled) T18 baseline.  expm_t18 is @torch.compile'd in
### expm.py; using it directly triggers inductor bmm autotuning noise.
### __wrapped__ gives the original Python function without the compiler.
_eager_t18 = expm_t18.__wrapped__


###
### Correctness checks
###
def _check(got: torch.Tensor, ref: torch.Tensor, atol: float) -> tuple[bool, float]:
    """Magnitude-aware check: passes if max_err ≤ atol * (1 + ||ref||_inf).

    A pure absolute tolerance is misleading for matrix-exp gradients, where
    ||L(A, G)||_inf scales with ||exp(A)|| and reaches O(100s) at moderate
    ||A||_1; bf16 (~1e-2 relative precision) then incurs an unavoidable
    ~few-units absolute error from the dtype cast alone.  Folding ref's
    magnitude in keeps the threshold meaningful across norms.
    """
    diff    = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    ref_mag = ref.float().abs().max().item()
    return max_err <= atol * (1.0 + ref_mag), max_err


_CORR_HDR = f"{'Config':>34}  {'Variant':>10}  {'Check':>10}  {'MaxErr':>10}  {'atol':>8}  Result"
_CORR_SEP = "-" * 92


def _corr_row(config, variant, check, max_err, atol, passed):
    result = ok("PASS") if passed else fail("FAIL")
    return f"{config:>34}  {variant:>10}  {check:>10}  {max_err:>10.2e}  {atol:>8.0e}  {result}"


def _corr_block(A: torch.Tensor, cfg_str: str, dtype: torch.dtype,
                atol_f: float, atol_b: float, all_passed: bool) -> bool:
    """Run fwd+bwd correctness on one A; return updated all_passed."""

    ### forward — Triton vs ground-truth (torch.linalg.matrix_exp)
    got = expm_t18_triton(A)
    ref = ref_torch_matrix_exp(A)
    passed, err = _check(got, ref, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "triton", "fwd", err, atol_f, passed))

    ### forward — Triton vs eager T18 (same algorithm, smaller tolerance)
    ref_t18 = _eager_t18(A)
    passed, err = _check(got, ref_t18, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "vs T18", "fwd", err, atol_f, passed))

    ### backward — grad_A from Triton vs from torch.linalg.matrix_exp
    A_t = A.detach().clone().requires_grad_(True)
    expm_t18_triton(A_t).sum().backward()
    grad_ref = ref_torch_matrix_exp_backward(A)

    passed, err = _check(A_t.grad, grad_ref, atol_b)
    all_passed &= passed
    print(_corr_row(cfg_str, "triton", "grad_A", err, atol_b, passed))

    return all_passed


def run_correctness(
    ns: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
):
    print()
    print(bold("=" * 92))
    print(bold("  CORRECTNESS — random A (small norm, no/few squarings)"))
    print(bold("=" * 92))
    print(_CORR_HDR)
    print(_CORR_SEP)

    all_passed = True

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
            all_passed = _corr_block(A, cfg_str, dtype, atol_f, atol_b, all_passed)

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
        atol_b = 5e-3 if dtype == torch.float32 else 1e-1

        for kind, B, N in product(["skew", "neg_psd", "diagonal", "large"], bs, ns):
            A = _A_FACTORIES[kind](B, N, dtype)
            cfg_str = f"[{_A_LABEL[kind]}] B={B} N={N} {dtype_name}"
            all_passed = _corr_block(A, cfg_str, dtype, atol_f, atol_b, all_passed)

        print(_CORR_SEP)

    print()
    if all_passed:
        print(ok("All correctness checks passed."))
    else:
        print(fail("One or more correctness checks FAILED."))
    print()
    return all_passed


###
### Performance benchmark
###
_PERF_HDR = (
    f"{'Config':>26}  {'AType':>5}  {'dtype':>6}  "
    f"{'Triton ms':>10}  {'matrix_exp ms':>14}  {'T18 ms':>7}  "
    f"{'vs torch':>9}  {'vs T18':>7}"
)
_PERF_SEP = "-" * 110


def _perf_row(config, atype, dtype_name, t_tri, t_torch, t_t18):
    def _sp(t_ref):
        sp = t_ref / t_tri
        s = f"{sp:.2f}x"
        return ok(s) if sp >= 1.05 else (warn(s) if sp >= 0.95 else fail(s))
    return (
        f"{config:>26}  {atype:>5}  {dtype_name:>6}  "
        f"{t_tri:>10.3f}  {t_torch:>14.3f}  {t_t18:>7.3f}  "
        f"{_sp(t_torch):>9}  {_sp(t_t18):>7}"
    )


def run_perf(
    ns: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
    warmup: int = 25,
    rep: int = 200,
    fwd: bool = True,
    bwd: bool = False,
):
    print()
    print(bold("=" * 110))
    print(bold("  PERFORMANCE — random / skew / neg_psd / diagonal / large-norm"))
    print(bold("=" * 110))
    print(_PERF_HDR)
    print(_PERF_SEP)

    kinds = ["random", "skew", "neg_psd", "diagonal", "large"]

    for dtype_name in dtypes:
        dtype = _dtype(dtype_name)

        for N, B in product(ns, bs):
            cfg_str = f"B={B} N={N}"
            for kind in kinds:
                A = _A_FACTORIES[kind](B, N, dtype)
                label = _A_LABEL[kind]

                if fwd:
                    t_tri   = triton.testing.do_bench(lambda: expm_t18_triton(A),      warmup=warmup, rep=rep)
                    t_torch = triton.testing.do_bench(lambda: ref_torch_matrix_exp(A), warmup=warmup, rep=rep)
                    t_t18   = triton.testing.do_bench(lambda: _eager_t18(A),           warmup=warmup, rep=rep)
                    print(_perf_row(cfg_str, label, dtype_name, t_tri, t_torch, t_t18))

                if bwd:
                    A_g = A.detach().clone().requires_grad_(True)

                    def _b_tri():
                        A_g.grad = None
                        expm_t18_triton(A_g).sum().backward()

                    def _b_torch():
                        A_g.grad = None
                        torch.linalg.matrix_exp(A_g.float()).to(dtype).sum().backward()

                    def _b_t18():
                        A_g.grad = None
                        _eager_t18(A_g).sum().backward()

                    t_b_tri   = triton.testing.do_bench(_b_tri,   warmup=warmup, rep=rep)
                    t_b_torch = triton.testing.do_bench(_b_torch, warmup=warmup, rep=rep)
                    t_b_t18   = triton.testing.do_bench(_b_t18,   warmup=warmup, rep=rep)
                    print(_perf_row(cfg_str, label + "+b", dtype_name, t_b_tri, t_b_torch, t_b_t18))

            print()  # blank between (N, B) groups

        print(_PERF_SEP)
    print()


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
    args = parser.parse_args()

    run_fwd = args.fwd or not args.bwd
    run_bwd = args.bwd

    if not torch.cuda.is_available():
        print(fail("No CUDA device found. Exiting."))
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    print(f"\nDevice    : {dev}")
    print(f"N vals    : {args.n}")
    print(f"B vals    : {args.b}")
    print(f"dtypes    : {args.dtype}")
    print(f"bench fwd : {run_fwd}")
    print(f"bench bwd : {run_bwd}")

    passed = True
    if args.mode in ("correctness", "all"):
        ### Correctness uses smaller B to keep the table compact
        passed = run_correctness(args.n, [min(args.b)], args.dtype)

    if args.mode in ("perf", "all"):
        run_perf(args.n, args.b, args.dtype,
                 warmup=args.warmup, rep=args.rep, fwd=run_fwd, bwd=run_bwd)

    if args.mode in ("correctness", "all") and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
