"""Shared helpers for the matrix-exponential benchmarks.

Coalesces the code that expm_bench.py and expm_force_bench.py would otherwise
duplicate verbatim: input factories, the matrix_exp reference, the magnitude-aware
correctness check, and CSV/report-path plumbing.
"""
from __future__ import annotations

import csv
import os

import torch

from bench_utils import DEVICE

_REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_reports")


###
### CSV export
###
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


def _make_large_norm_A(B: int, N: int, dtype: torch.dtype, target_norm: float = 5.0, seed: int = 4):
    """Norm-stress: random A rescaled so ||A||_1 == target_norm per batch matrix.

    ||A||_1 directly controls the number of scaling-and-squaring steps s in T18,
    so sweeping target_norm walks across the s=0,1,2,... thresholds.  Very large
    norms drive ||exp(A)|| to thousands and cause intrinsic fp32/bf16 precision
    loss in *any* T18 implementation (not a kernel bug), so keep the sweep within
    a regime where the comparison stays meaningful.
    """
    torch.manual_seed(seed)
    A = torch.randn(B, N, N, device=DEVICE, dtype=dtype)
    cur = torch.linalg.matrix_norm(A.float(), ord=1).clamp_min(1e-12)   # [B] max abs col-sum
    return (A.float() * (target_norm / cur).view(B, 1, 1)).to(dtype)


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
### Reference implementation
###
def ref_torch_matrix_exp(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_exp(A.float()).to(A.dtype)


###
### Correctness check
###
def _check(got: torch.Tensor, ref: torch.Tensor, atol: float) -> tuple[bool, float]:
    """Magnitude-aware check: passes if max_err <= atol * (1 + ||ref||_inf).

    A pure absolute tolerance is misleading for matrix-exp outputs/gradients,
    whose magnitude scales with ||exp(A)|| and reaches O(100s) at moderate
    ||A||_1; bf16 (~1e-2 relative precision) then incurs an unavoidable few-units
    absolute error from the dtype cast alone.  Folding ref's magnitude in keeps
    the threshold meaningful across norms and dtypes.
    """
    diff    = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    ref_mag = ref.float().abs().max().item()
    return max_err <= atol * (1.0 + ref_mag), max_err
