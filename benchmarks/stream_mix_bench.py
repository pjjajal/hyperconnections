"""
Numerical correctness and performance benchmark for stream_mix_add.

Usage
-----
# Run everything (default):
    python benchmarks/stream_mix_bench.py

# Only correctness:
    python benchmarks/stream_mix_bench.py --mode correctness

# Only performance (random + structured):
    python benchmarks/stream_mix_bench.py --mode perf

# Forward only (explicit):
    python benchmarks/stream_mix_bench.py --mode perf --fwd

# Backward only:
    python benchmarks/stream_mix_bench.py --mode perf --bwd

# Both forward and backward:
    python benchmarks/stream_mix_bench.py --mode perf --fwd --bwd

# Restrict to specific N values:
    python benchmarks/stream_mix_bench.py --mode perf --n 4 8

# Only float16:
    python benchmarks/stream_mix_bench.py --mode perf --dtype fp16

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
from einops import einsum

from hyperconnections.ops import stream_mix_add

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


def _make(B, N, D, dtype, seed=0):
    torch.manual_seed(seed)
    Phi = torch.randn(B, N, N, device=DEVICE, dtype=dtype)
    x   = torch.randn(B, N, D, device=DEVICE, dtype=dtype)
    Y   = torch.randn(B, N, D, device=DEVICE, dtype=dtype)
    return Phi, x, Y


def _make_v(B, N, dtype, seed=7):
    torch.manual_seed(seed)
    v = torch.randn(B, N, device=DEVICE, dtype=dtype)
    return torch.nn.functional.normalize(v, dim=-1)


###
### Structured Phi factories
###

def _make_skew_phi(B: int, N: int, dtype: torch.dtype, seed: int = 1) -> torch.Tensor:
    """Skew-symmetric: Phi = M - M^T  (Phi^T = -Phi)."""
    torch.manual_seed(seed)
    M = torch.randn(B, N, N, device=DEVICE, dtype=dtype) / N ** 0.5
    return M - M.mT


def _make_psd_phi(B: int, N: int, dtype: torch.dtype, seed: int = 2) -> torch.Tensor:
    """Symmetric positive semi-definite: Phi = R @ R^T."""
    torch.manual_seed(seed)
    R = torch.randn(B, N, N, device=DEVICE, dtype=dtype) / N ** 0.5
    return R @ R.mT


def _make_diag_phi(B: int, N: int, dtype: torch.dtype, seed: int = 3) -> torch.Tensor:
    """Diagonal: Phi = diag(d), d ~ N(0,1)."""
    torch.manual_seed(seed)
    d = torch.randn(B, N, device=DEVICE, dtype=dtype)
    return torch.diag_embed(d)

###
### Maps a short name to a factory (B, N, dtype) -> Phi
###
_PHI_FACTORIES = {
    "random":   lambda B, N, dt: _make(B, N, 1, dt, seed=0)[0],  # D unused
    "skew_sym": _make_skew_phi,
    "psd":      _make_psd_phi,
    "diagonal": _make_diag_phi,
}

_PHI_LABEL = {
    "random":   "rand",
    "skew_sym": "skew",
    "psd":      "psd ",
    "diagonal": "diag",
}


###
### Reference implementations
###
def ref_no_proj(Phi, x, Y):
    x_mixed = einsum(Phi, x, "b n1 n2, b n2 d -> b n1 d") + Y
    return x_mixed


def ref_diagonal_add(Phi, x, Y):
    """Structure-aware O(ND) baseline for diagonal Phi (vs O(N²D) bmm)."""
    d = torch.diagonal(Phi, dim1=-2, dim2=-1)   # [B, N]
    return d.unsqueeze(-1) * x + Y


def ref_proj(Phi, x, Y, v):
    b,n,d = x.shape 
    proj_matrix = einsum(
        v, v, "b n1, b n2 -> b n1 n2"
    )  # [b, n, n]
    orthogonal_proj = (
        torch.eye(n, device=x.device, dtype=x.dtype) - proj_matrix
    )  # [b, n, n]
    x_proj = einsum(
        proj_matrix, x, "b n1 n2, b n2 d -> b n1 d"
    )  # [b, n, block_size]
    x_orth = einsum(
        orthogonal_proj, x, "b n1 n2, b n2 d -> b n1 d"
    )  # [b, n, block_size]
    x_mixed = x_proj + einsum(
        Phi, x_orth, "b n1 n2, b n2 d -> b n1 d"
    )  # [B*, n, block_size]
    return x_mixed + Y


### torch.compile baselines — compiled once at import time.
### mode="reduce-overhead" enables CUDA graph capture and op fusion.
_ref_no_proj_compiled = torch.compile(ref_no_proj)
_ref_proj_compiled = torch.compile(ref_proj)


def ref_proj_backward(Phi, x, Y, v):
    """Return (grad_Phi, grad_x, grad_Y) using PyTorch autograd on ref_proj."""
    Phi_r = Phi.detach().float().requires_grad_(True)
    x_r   = x.detach().float().requires_grad_(True)
    Y_r   = Y.detach().float().requires_grad_(True)
    v_r   = v.detach().float()
    ref_proj(Phi_r, x_r, Y_r, v_r).sum().backward()
    return Phi_r.grad, x_r.grad, Y_r.grad


def ref_no_proj_backward(Phi, x, Y):
    Phi_r = Phi.detach().float().requires_grad_(True)
    x_r   = x.detach().float().requires_grad_(True)
    Y_r   = Y.detach().float().requires_grad_(True)
    ref_no_proj(Phi_r, x_r, Y_r).sum().backward()
    return Phi_r.grad, x_r.grad, Y_r.grad


###
### Correctness checks
###

def _check(label: str, got: torch.Tensor, ref: torch.Tensor, atol: float) -> tuple[bool, float]:
    diff = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    passed = max_err <= atol
    return passed, max_err


# Column widths for the correctness table
_CORR_HDR = f"{'Config':>30}  {'Variant':>12}  {'Check':>10}  {'MaxErr':>10}  {'atol':>8}  Result"
_CORR_SEP = "-" * 90


def _corr_row(config, variant, check, max_err, atol, passed):
    result = ok("PASS") if passed else fail("FAIL")
    return f"{config:>30}  {variant:>12}  {check:>10}  {max_err:>10.2e}  {atol:>8.0e}  {result}"


def _corr_block(Phi, x, Y, v, cfg_str, dtype, atol_f, atol_b, all_passed):
    """Run fwd+bwd correctness for one (Phi, x, Y, v) combo; return updated all_passed."""

    ### forward no-proj
    got = stream_mix_add(Phi, x, Y)
    ref = ref_no_proj(Phi.float(), x.float(), Y.float()).to(dtype)
    passed, err = _check("", got, ref, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "no-proj", "fwd", err, atol_f, passed))

    ### forward proj
    got = stream_mix_add(Phi, x, Y, v=v)
    ref = ref_proj(Phi.float(), x.float(), Y.float(), v.float()).to(dtype)
    passed, err = _check("", got, ref, atol_f)
    all_passed &= passed
    print(_corr_row(cfg_str, "proj", "fwd", err, atol_f, passed))

    ### backward no-proj
    Phi_t = Phi.detach().clone().requires_grad_(True)
    x_t   = x.detach().clone().requires_grad_(True)
    Y_t   = Y.detach().clone().requires_grad_(True)
    stream_mix_add(Phi_t, x_t, Y_t).sum().backward()
    gP_r, gx_r, gY_r = ref_no_proj_backward(Phi, x, Y)

    for name, got_g, ref_g in [
        ("grad_Phi", Phi_t.grad, gP_r),
        ("grad_x",   x_t.grad,   gx_r),
        ("grad_Y",   Y_t.grad,   gY_r),
    ]:
        passed, err = _check("", got_g, ref_g, atol_b)
        all_passed &= passed
        print(_corr_row(cfg_str, "no-proj", name, err, atol_b, passed))

    ### backward proj
    Phi_t = Phi.detach().clone().requires_grad_(True)
    x_t   = x.detach().clone().requires_grad_(True)
    Y_t   = Y.detach().clone().requires_grad_(True)
    stream_mix_add(Phi_t, x_t, Y_t, v=v).sum().backward()
    gP_r, gx_r, gY_r = ref_proj_backward(Phi, x, Y, v)

    for name, got_g, ref_g in [
        ("grad_Phi", Phi_t.grad, gP_r),
        ("grad_x",   x_t.grad,   gx_r),
        ("grad_Y",   Y_t.grad,   gY_r),
    ]:
        passed, err = _check("", got_g, ref_g, atol_b)
        all_passed &= passed
        print(_corr_row(cfg_str, "proj", name, err, atol_b, passed))

    return all_passed


def _derive_D_vals(ms: Sequence[int], embed_dims: Sequence[int]) -> list[int]:
    """Return sorted unique D = embed_dim // m values for valid (m, embed_dim) pairs."""
    return sorted({e // m for m, e in product(ms, embed_dims) if e % m == 0})


def run_correctness(
    ns: Sequence[int],
    ms: Sequence[int],
    embed_dims: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
):
    """Run forward + backward correctness checks and print a summary table."""
    print()
    print(bold("=" * 90))
    print(bold(" CORRECTNESS — random Phi"))
    print(bold("=" * 90))
    print(_CORR_HDR)
    print(_CORR_SEP)

    # D = embed_dim // m; deduplicate so the table stays concise
    # (correctness only depends on D, not on how it was derived from m)
    D_vals = _derive_D_vals(ms, embed_dims)
    configs = list(product(bs, ns, D_vals))
    configs += [(4, ns[0], 100)]   # non-power-of-2 D to test masking

    all_passed = True

    for dtype_name in dtypes:
        dtype  = _dtype(dtype_name)
        atol_f = 1e-3 if dtype == torch.float32 else 2e-2
        atol_b = 2e-3 if dtype == torch.float32 else 4e-2

        for B, N, D in configs:
            Phi, x, Y = _make(B, N, D, dtype)
            v = _make_v(B, N, dtype)
            cfg_str = f"B={B} N={N} D={D} {dtype_name}"
            all_passed = _corr_block(Phi, x, Y, v, cfg_str, dtype, atol_f, atol_b, all_passed)

        print(_CORR_SEP)

    ### Structured Phi correctness
    ### Focused config set to keep the table concise.
    print()
    print(bold("=" * 90))
    print(bold("  CORRECTNESS — Structured Phi"))
    print(bold("=" * 90))
    print(_CORR_HDR)
    print(_CORR_SEP)

    for dtype_name in dtypes:
        struct_D_vals = _derive_D_vals(ms, embed_dims)
        struct_configs = list(product(bs, ns, struct_D_vals))
        
        dtype  = _dtype(dtype_name)
        atol_f = 1e-3 if dtype == torch.float32 else 2e-2
        atol_b = 2e-3 if dtype == torch.float32 else 4e-2

        for phi_type, (B, N, D) in product(
            ["skew_sym", "psd", "diagonal"], struct_configs
        ):
            factory = _PHI_FACTORIES[phi_type]
            # Generate Phi with appropriate structure; x, Y are always random
            _, x, Y = _make(B, N, D, dtype, seed=42)
            Phi = factory(B, N, dtype).contiguous()
            v   = _make_v(B, N, dtype)
            cfg_str = f"[{_PHI_LABEL[phi_type]}] B={B} N={N} D={D} {dtype_name}"
            all_passed = _corr_block(Phi, x, Y, v, cfg_str, dtype, atol_f, atol_b, all_passed)

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

def _bytes_no_proj(B, N, D, elem_bytes):
    """Bytes touched in the ideal (no data reuse) model: read Phi, x, Y; write out."""
    return elem_bytes * (B * N * N + 3 * B * N * D)


def _bytes_proj(B, N, D, elem_bytes):
    """Same as no-proj plus reading v once per row (N reads × N rows per batch)."""
    return elem_bytes * (B * N * N + 3 * B * N * D + B * N)


def _bytes_no_proj_bwd(B, N, D, elem_bytes):
    """Ideal bytes for fwd+bwd (no-proj): Phi read×2, x read×2, G read×2, Y/out/grad_x/grad_Phi once each."""
    return elem_bytes * (3 * B * N * N + 7 * B * N * D)


def _bytes_proj_bwd(B, N, D, elem_bytes):
    """Ideal bytes for fwd+bwd (proj): no-proj total plus v read×3."""
    return _bytes_no_proj_bwd(B, N, D, elem_bytes) + elem_bytes * 3 * B * N


_PERF_HDR = (
    f"{'Config':>30}  {'Variant':>12}  {'dtype':>6}  "
    f"{'Triton ms':>10}  {'Eager ms':>9}  {'Compiled ms':>12}  "
    f"{'vs Eager':>9}  {'vs Cmp':>7}  {'BW GB/s':>9}"
)
_PERF_SEP = "-" * 120


def _perf_row(config, variant, dtype_name, t_tri, t_eager, t_compiled, bw_gbs):
    def _sp(t_ref):
        sp = t_ref / t_tri
        s = f"{sp:.2f}x"
        return ok(s) if sp >= 1.05 else (warn(s) if sp >= 0.95 else fail(s))
    return (
        f"{config:>30}  {variant:>12}  {dtype_name:>6}  "
        f"{t_tri:>10.3f}  {t_eager:>9.3f}  {t_compiled:>12.3f}  "
        f"{_sp(t_eager):>9}  {_sp(t_compiled):>7}  {bw_gbs:>9.1f}"
    )


def run_perf(
    ns: Sequence[int],
    ms: Sequence[int],
    embed_dims: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
    warmup: int = 25,
    rep: int = 200,
    fwd: bool = True,
    bwd: bool = False,
):
    """Benchmark Triton kernel vs PyTorch bmm+add reference."""
    print()
    print(bold("=" * 120))
    print(bold("  PERFORMANCE"))
    print(bold("=" * 120))
    print(_PERF_HDR)
    print(_PERF_SEP)

    for dtype_name in dtypes:
        dtype = _dtype(dtype_name)
        elem  = torch.finfo(dtype).bits // 8

        for N, B, m, embed_dim in product(ns, bs, ms, embed_dims):
            if embed_dim % m != 0:
                continue
            D = embed_dim // m
            Phi, x, Y = _make(B, N, D, dtype)
            v = _make_v(B, N, dtype)
            cfg_str = f"B={B} N={N} m={m} D={D}"

            if fwd:
                ### no-proj
                t_tri = triton.testing.do_bench(
                    lambda: stream_mix_add(Phi, x, Y),
                    warmup=warmup, rep=rep,
                )
                t_eager = triton.testing.do_bench(
                    lambda: ref_no_proj(Phi, x, Y),
                    warmup=warmup, rep=rep,
                )
                t_compiled = triton.testing.do_bench(
                    lambda: _ref_no_proj_compiled(Phi, x, Y),
                    warmup=warmup, rep=rep,
                )
                bw = _bytes_no_proj(B, N, D, elem) / (t_tri * 1e-3) / 1e9
                print(_perf_row(cfg_str, "no-proj", dtype_name, t_tri, t_eager, t_compiled, bw))

                ### proj
                t_tri_p = triton.testing.do_bench(
                    lambda: stream_mix_add(Phi, x, Y, v=v),
                    warmup=warmup, rep=rep,
                )
                t_eager_p = triton.testing.do_bench(
                    lambda: ref_proj(Phi, x, Y, v),
                    warmup=warmup, rep=rep,
                )
                t_compiled_p = triton.testing.do_bench(
                    lambda: _ref_proj_compiled(Phi, x, Y, v),
                    warmup=warmup, rep=rep,
                )
                bw_p = _bytes_proj(B, N, D, elem) / (t_tri_p * 1e-3) / 1e9
                print(_perf_row(cfg_str, "proj", dtype_name, t_tri_p, t_eager_p, t_compiled_p, bw_p))

            if bwd:
                ### no-proj fwd+bwd
                Phi_g = Phi.clone().requires_grad_(True)
                x_g   = x.clone().requires_grad_(True)
                Y_g   = Y.clone().requires_grad_(True)

                def _b_tri():
                    Phi_g.grad = x_g.grad = Y_g.grad = None
                    stream_mix_add(Phi_g, x_g, Y_g).sum().backward()
                def _b_ea():
                    Phi_g.grad = x_g.grad = Y_g.grad = None
                    ref_no_proj(Phi_g, x_g, Y_g).sum().backward()
                def _b_cmp():
                    Phi_g.grad = x_g.grad = Y_g.grad = None
                    _ref_no_proj_compiled(Phi_g, x_g, Y_g).sum().backward()

                t_b   = triton.testing.do_bench(_b_tri, warmup=warmup, rep=rep)
                t_b_e = triton.testing.do_bench(_b_ea,  warmup=warmup, rep=rep)
                t_b_c = triton.testing.do_bench(_b_cmp, warmup=warmup, rep=rep)
                bw_b  = _bytes_no_proj_bwd(B, N, D, elem) / (t_b * 1e-3) / 1e9
                print(_perf_row(cfg_str, "no-proj+bwd", dtype_name, t_b, t_b_e, t_b_c, bw_b))

                ### proj fwd+bwd
                Phi_g2 = Phi.clone().requires_grad_(True)
                x_g2   = x.clone().requires_grad_(True)
                Y_g2   = Y.clone().requires_grad_(True)

                def _bp_tri():
                    Phi_g2.grad = x_g2.grad = Y_g2.grad = None
                    stream_mix_add(Phi_g2, x_g2, Y_g2, v=v).sum().backward()
                def _bp_ea():
                    Phi_g2.grad = x_g2.grad = Y_g2.grad = None
                    ref_proj(Phi_g2, x_g2, Y_g2, v).sum().backward()
                def _bp_cmp():
                    Phi_g2.grad = x_g2.grad = Y_g2.grad = None
                    _ref_proj_compiled(Phi_g2, x_g2, Y_g2, v).sum().backward()

                t_bp   = triton.testing.do_bench(_bp_tri, warmup=warmup, rep=rep)
                t_bp_e = triton.testing.do_bench(_bp_ea,  warmup=warmup, rep=rep)
                t_bp_c = triton.testing.do_bench(_bp_cmp, warmup=warmup, rep=rep)
                bw_bp  = _bytes_proj_bwd(B, N, D, elem) / (t_bp * 1e-3) / 1e9
                print(_perf_row(cfg_str, "proj+bwd", dtype_name, t_bp, t_bp_e, t_bp_c, bw_bp))

        print(_PERF_SEP)
    print()


###
### Structured-matrix performance benchmark
###

_SPHDR = (
    f"{'Config':>24}  {'PhiType':>8}  {'dtype':>6}  "
    f"{'Triton ms':>10}  {'Eager ms':>9}  {'Compiled ms':>12}  {'Speedup':>8}  "
    f"{'Diag-fast ms':>13}  {'vs Diag':>8}"
)
_SPSEP = "-" * 125


def run_structured_perf(
    ns: Sequence[int],
    ms: Sequence[int],
    embed_dims: Sequence[int],
    bs: Sequence[int],
    dtypes: Sequence[str],
    warmup: int = 25,
    rep: int = 200,
    fwd: bool = True,
    bwd: bool = False,
):
    """Benchmark all four Phi structures.

    For diagonal Phi an additional structure-aware PyTorch baseline is shown:
      ref_diagonal_add  uses  d * x + Y  (O(ND) vs O(N²D)), revealing the
      overhead of loading the full Phi matrix in the Triton kernel.
    """
    print()
    print(bold("=" * 125))
    print(bold("  STRUCTURED-MATRIX PERFORMANCE"))
    print(bold("=" * 125))
    print(_SPHDR)
    print(_SPSEP)

    phi_types = list(_PHI_FACTORIES.keys())   # random, skew_sym, psd, diagonal

    for dtype_name in dtypes:
        dtype = _dtype(dtype_name)
        elem  = torch.finfo(dtype).bits // 8

        for N, B, m, embed_dim in product(ns, bs, ms, embed_dims):
            if embed_dim % m != 0:
                continue
            D = embed_dim // m
            _, x, Y = _make(B, N, D, dtype, seed=0)
            v = _make_v(B, N, dtype)
            cfg_str = f"B={B} N={N} m={m} D={D}"

            for phi_type in phi_types:
                Phi  = _PHI_FACTORIES[phi_type](B, N, dtype).contiguous()
                label = _PHI_LABEL[phi_type]

                if fwd:
                    t_tri = triton.testing.do_bench(
                        lambda: stream_mix_add(Phi, x, Y),
                        warmup=warmup, rep=rep,
                    )
                    t_eager = triton.testing.do_bench(
                        lambda: ref_no_proj(Phi, x, Y),
                        warmup=warmup, rep=rep,
                    )
                    t_comp = triton.testing.do_bench(
                        lambda: _ref_no_proj_compiled(Phi, x, Y),
                        warmup=warmup, rep=rep,
                    )

                    def _sp(t_ref):
                        sp = t_ref / t_tri
                        s = f"{sp:.2f}x"
                        return ok(s) if sp >= 1.05 else (warn(s) if sp >= 0.95 else fail(s))

                    if phi_type == "diagonal":
                        t_diag = triton.testing.do_bench(
                            lambda: ref_diagonal_add(Phi, x, Y),
                            warmup=warmup, rep=rep,
                        )
                        ratio = t_diag / t_tri
                        diag_str = f"{t_diag:>13.3f}"
                        vs_diag = f"{ratio:.2f}x"
                        vd_col = ok(vs_diag) if ratio >= 1.05 else (
                            warn(vs_diag) if ratio >= 0.95 else fail(vs_diag)
                        )
                    else:
                        diag_str = f"{'N/A':>13}"
                        vd_col   = f"{'N/A':>8}"

                    bw = _bytes_no_proj(B, N, D, elem) / (t_tri * 1e-3) / 1e9
                    print(
                        f"{cfg_str:>24}  {label:>8}  {dtype_name:>6}  "
                        f"{t_tri:>10.3f}  {t_eager:>9.3f}  {t_comp:>12.3f}  {_sp(t_eager):>8}  "
                        f"{diag_str}  {vd_col:>8}"
                    )

                if bwd:
                    Phi_g = Phi.clone().requires_grad_(True)
                    x_g   = x.clone().requires_grad_(True)
                    Y_g   = Y.clone().requires_grad_(True)

                    def _sb_tri():
                        Phi_g.grad = x_g.grad = Y_g.grad = None
                        stream_mix_add(Phi_g, x_g, Y_g).sum().backward()
                    def _sb_ea():
                        Phi_g.grad = x_g.grad = Y_g.grad = None
                        ref_no_proj(Phi_g, x_g, Y_g).sum().backward()
                    def _sb_cmp():
                        Phi_g.grad = x_g.grad = Y_g.grad = None
                        _ref_no_proj_compiled(Phi_g, x_g, Y_g).sum().backward()

                    t_tri_b  = triton.testing.do_bench(_sb_tri, warmup=warmup, rep=rep)
                    t_ea_b   = triton.testing.do_bench(_sb_ea,  warmup=warmup, rep=rep)
                    t_cmp_b  = triton.testing.do_bench(_sb_cmp, warmup=warmup, rep=rep)

                    def _spb(t_ref):
                        sp = t_ref / t_tri_b
                        s = f"{sp:.2f}x"
                        return ok(s) if sp >= 1.05 else (warn(s) if sp >= 0.95 else fail(s))

                    bw_b    = _bytes_no_proj_bwd(B, N, D, elem) / (t_tri_b * 1e-3) / 1e9
                    label_b = label.rstrip() + "+bwd"
                    print(
                        f"{cfg_str:>24}  {label_b:>8}  {dtype_name:>6}  "
                        f"{t_tri_b:>10.3f}  {t_ea_b:>9.3f}  {t_cmp_b:>12.3f}  {_spb(t_ea_b):>8}  "
                        f"{'N/A':>13}  {'N/A':>8}"
                    )

            print()   # blank line between (N,B,D) groups

        print(_SPSEP)

    print()


###
### Entry point
###
def main():
    parser = argparse.ArgumentParser(description="stream_mix_add benchmark")
    parser.add_argument(
        "--mode", choices=["correctness", "perf", "all"], default="all",
        help="Which sections to run (default: all)",
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[4],
        metavar="N", help="N_STREAMS values to benchmark (default: 4 8 16)",
    )
    parser.add_argument(
        "--m", type=int, nargs="+", default=[1],
        metavar="M", help="m values (modules per HC layer, default: 1)",
    )
    parser.add_argument(
        "--b", type=int, nargs="+", default=[1024],
        metavar="B", help="batch size values, shared across correctness and perf (default: 64 512 2048)",
    )
    parser.add_argument(
        "--embed-dim", type=int, nargs="+", default=[1024],
        metavar="E",
        help="embed_dim values; D = embed_dim // m (default: 128 256 512)",
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], nargs="+",
        default=["bf16"], metavar="DTYPE",
        help="dtypes to test (default: bf16)",
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
        help="Benchmark forward pass (default when neither --fwd nor --bwd is given)",
    )
    parser.add_argument(
        "--bwd", action="store_true", default=False,
        help="Benchmark fwd+bwd pass (Triton vs eager vs compiled) in performance tables",
    )
    args = parser.parse_args()

    # --fwd and --bwd are independent; default (neither given) → fwd only
    run_fwd = args.fwd or not args.bwd
    run_bwd = args.bwd

    if not torch.cuda.is_available():
        print(fail("No CUDA device found. Exiting."))
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    print(f"\nDevice     : {dev}")
    print(f"N vals     : {args.n}")
    print(f"m vals     : {args.m}")
    print(f"B vals     : {args.b}")
    print(f"embed_dims : {args.embed_dim}")
    print(f"dtypes     : {args.dtype}")
    print(f"bench fwd  : {run_fwd}")
    print(f"bench bwd  : {run_bwd}")

    passed = True
    if args.mode in ("correctness", "all"):
        passed = run_correctness(args.n, args.m, args.embed_dim, args.b, args.dtype)

    if args.mode in ("perf", "all"):
        run_perf(args.n, args.m, args.embed_dim, args.b, args.dtype,
                 warmup=args.warmup, rep=args.rep, fwd=run_fwd, bwd=run_bwd)
        run_structured_perf(args.n, args.m, args.embed_dim, args.b, args.dtype,
                            warmup=args.warmup, rep=args.rep, fwd=run_fwd, bwd=run_bwd)

    if args.mode in ("correctness", "all") and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
