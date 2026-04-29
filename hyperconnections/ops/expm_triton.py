"""
Triton implementation of the optimised T18 matrix-exponential approximation.

Algorithm: Bader, Blanes & Casas, "Computing the Matrix Exponential with an
Optimized Taylor Polynomial Approximation", Mathematics 2019, 7, 1174.

Forward
-------
Single fused kernel.  Grid = (B,).  One CTA per batch element.

  - Loads A[b] and a global scaling factor inv_scale = 2^(-s), where
    s = ceil(log2(max_b ||A_b||_1 / θ₁₈)) is precomputed *on-device* by the
    Python wrapper and passed as a 0-d tensor (no host sync).
  - Computes A^2, A^3, A^6 and the polynomial blocks B1..B5 in registers
    (N is a compile-time constant).  Each matmul is an unrolled
    outer-product accumulation  C = Σ_k col_k(A) · row_k(B)  via a
    static_range over k.  The naïve `tl.sum(A[:,:,None]*B[None,:,:],axis=1)`
    contraction looks identical mathematically but at NP≥16 the Triton
    backend silently lowers it to a TF32 MMA — which drops the fp32
    mantissa from 23 to 10 bits and drives ~1e-3 errors on the BBC
    polynomial.  Going through static_range over k keeps the IR as
    scalar FFMAs and preserves full fp32 precision.  tl.dot is
    intentionally avoided for the same reason and to support N<16.
  - Applies up to MAX_S=8 squarings, masked on s via tl.where.  Wasted matmul
    work for the masked-off branch is negligible relative to polynomial cost,
    and s is uniform across the batch (global scaling).
  - Accumulation is fp32 regardless of input dtype; output is cast to
    A.dtype on store.

Backward (autograd)
-------------------
Najfeld–Havel / Higham §10.6 augmented-matrix identity:

    exp([[X, E]; [0, X]])  =  [[exp(X), L(X,E)]; [0, exp(X)]]

where L is the Frechet derivative of exp at X in direction E.  The Frobenius
adjoint of L(A, ⋅) is L(Aᵀ, ⋅), so given upstream gradient G = ∂L/∂Y with
Y = exp(A):

    ∂L/∂A  =  L(Aᵀ, G)  =  upper-right block of  exp([[Aᵀ, G]; [0, Aᵀ]]).

Backward therefore assembles a [B, 2N, 2N] fp32 augmented matrix and recurses
into the same Triton kernel with N → 2N, slicing out the top-right block.
For the common case N=4 the backward call runs at N_aug=8 — still well inside
the unrolled regime.

Padding
-------
Triton requires power-of-2 tile sizes for tl.arange, so we launch with
NP = next_pow2(N) and mask both loads and stores.  The block-zero structure
of A in the padded rows/columns (from masked loads with other=0) means the
padded tile entries never propagate into the [:N,:N] result, even though the
identity matrix used in the polynomial is full-NP×NP.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


###
### T18 polynomial coefficients (Bader, Blanes & Casas 2019, eq. 15, k=5)
### a01 = b01 = b04 = b14 = 0 (dropped from the polynomial below)
###
_THETA_18 = 3.01      # scaling threshold for u <= 2^-24 (fp32)
_MAX_S    = 8         # max squarings ⇒ ||A||_1 up to θ₁₈ · 2^8 ≈ 770

_TORCH_TO_TL = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}

###
### Autotune configs — keyed on NP (next_pow2(N)) since the whole tile lives in
### registers and the only tunable axes are warp count and software-pipeline
### depth.  No BLOCK_D: the kernel is fully unrolled at compile time.
### Backward reuses the same kernel at N→2N, so its NP key lands in a separate
### autotune bucket automatically.
###
_FWD_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=1),
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
]

###
### Triton kernel
###
@triton.jit
def _matmul_nn(A, B, NP: tl.constexpr, n_idx):
    """A @ B for square [NP, NP] register tiles, fp32, scalar-FFMA only.

    Implemented as Σ_k outer(col_k(A), row_k(B)) via a static_range loop.
    The naïve broadcast contraction tl.sum(A[:,:,None] * B[None,:,:], axis=1)
    is mathematically identical but the Triton backend lowers it to a TF32
    MMA at NP≥16 (verified empirically on sm_80, Triton 3.6) — which loses
    13 mantissa bits and produces ~1e-3 errors in the BBC polynomial.

    n_idx is passed in from the caller (it's the same tl.arange(0, NP) the
    caller already built for load/store offsets) so we don't rebuild it
    NP times across the per-k loop.
    """
    R = tl.zeros([NP, NP], dtype=tl.float32)
    for k in tl.static_range(NP):
        e_k   = (n_idx == k).to(tl.float32)            # [NP]
        col_k = tl.sum(A * e_k[None, :], axis=1)       # [NP]  = A[:, k]
        row_k = tl.sum(B * e_k[:, None], axis=0)       # [NP]  = B[k, :]
        R = R + col_k[:, None] * row_k[None, :]
    return R


@triton.autotune(configs=_FWD_CONFIGS, key=["NP"])
@triton.jit
def _expm_t18_fwd(
    A_ptr, out_ptr,
    s_ptr, inv_scale_ptr,                # 0-d device scalars (no host sync)
    stride_a_b, stride_a_n1, stride_a_n2,
    stride_o_b, stride_o_n1, stride_o_n2,
    N:         tl.constexpr,             # logical matrix size
    NP:        tl.constexpr,             # next_pow2(N) — Triton tile size
    MAX_S:     tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_b  = tl.program_id(0)
    n_idx  = tl.arange(0, NP)
    n_mask = n_idx < N
    mask2d = n_mask[:, None] & n_mask[None, :]

    ### Load A[b], upcast, and apply the scaling 1/2^s
    a_off = (
        pid_b * stride_a_b
        + n_idx[:, None] * stride_a_n1
        + n_idx[None, :] * stride_a_n2
    )
    A = tl.load(A_ptr + a_off, mask=mask2d, other=0.0).to(tl.float32)
    inv_scale = tl.load(inv_scale_ptr).to(tl.float32)
    A = A * inv_scale

    ### A^2, A^3, A^6 — see _matmul_nn for why we don't use the naïve
    ### broadcast+sum contraction.
    A2 = _matmul_nn(A,  A,  NP, n_idx)
    A3 = _matmul_nn(A2, A,  NP, n_idx)
    A6 = _matmul_nn(A3, A3, NP, n_idx)

    ### Identity (NP×NP).  Padded diag ones live only in the [N:,N:] block.
    eye = tl.where(n_idx[:, None] == n_idx[None, :], 1.0, 0.0)

    ### Polynomial blocks (a01 = b01 = b04 = b14 = 0 dropped)
    a11 = -0.10036558103014462
    a21 = -0.00802924648241157
    a31 = -0.00089213849804573

    b11 =  0.39784974949645076
    b21 =  1.36783778460411719
    b31 =  0.49828962252538268
    b61 = -0.00063789819459247233

    b02 = -10.96763960529620626
    b12 =  1.68015813878906197
    b22 =  0.05717798464788655
    b32 = -0.00698210122488052
    b62 =  3.349750170860705e-05

    b03 = -0.09043168323908106
    b13 = -0.06764045190713819
    b23 =  0.06759613017740597
    b33 =  0.02955525704293155
    b63 = -1.391802575160607e-05

    b24 = -0.09233646193671186
    b34 = -0.01693649390020817
    b64 = -1.400867981820361e-05

    B1 = a11 * A           + a21 * A2 + a31 * A3
    B2 = b11 * A           + b21 * A2 + b31 * A3 + b61 * A6
    B3 = b02 * eye + b12 * A + b22 * A2 + b32 * A3 + b62 * A6
    B4 = b03 * eye + b13 * A + b23 * A2 + b33 * A3 + b63 * A6
    B5 = b24 * A2 + b34 * A3 + b64 * A6

    ### A9 = B1 @ B5 + B4
    A9 = _matmul_nn(B1, B5, NP, n_idx) + B4

    ### T18 = B2 + (B3 + A9) @ A9
    T18 = B2 + _matmul_nn(B3 + A9, A9, NP, n_idx)

    ### Repeated squaring, gated by global s. Always evaluates the matmul;
    ### The wasted work is uniform across the grid and small vs polynomial.
    s_val = tl.load(s_ptr).to(tl.int32)
    for i in tl.static_range(MAX_S):
        T18_sq = _matmul_nn(T18, T18, NP, n_idx)
        T18    = tl.where(s_val > i, T18_sq, T18)

    ### Store output
    o_off = (
        pid_b * stride_o_b
        + n_idx[:, None] * stride_o_n1
        + n_idx[None, :] * stride_o_n2
    )
    tl.store(out_ptr + o_off, T18.to(OUT_DTYPE), mask=mask2d)


###
### Python launcher (no autograd)
###
def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _expm_t18_no_grad(A: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Triton T18 forward.  No autograd wrapping; used by the autograd
    Function for both forward and the augmented-matrix backward."""
    if not A.is_cuda:
        raise RuntimeError("expm_t18_triton requires CUDA tensors")
    if A.dim() != 3 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"expected [B, N, N], got {tuple(A.shape)}")
    if out_dtype not in _TORCH_TO_TL:
        raise ValueError(f"unsupported out_dtype {out_dtype}")

    B, N, _ = A.shape

    A_fp32 = A.to(torch.float32).contiguous()

    ### Global scaling factor — all on-device, no host sync.
    A_norm    = torch.linalg.matrix_norm(A_fp32, ord=1).max().clamp_min(_THETA_18)
    s         = torch.ceil(torch.log2(A_norm / _THETA_18)).clamp(min=0).to(torch.int32)
    inv_scale = torch.exp2(-s.float())

    out = torch.empty(B, N, N, dtype=out_dtype, device=A.device)
    NP  = _next_pow2(N)

    _expm_t18_fwd[(B,)](
        A_fp32, out,
        s, inv_scale,
        *A_fp32.stride(), *out.stride(),
        N=N, NP=NP, MAX_S=_MAX_S,
        OUT_DTYPE=_TORCH_TO_TL[out_dtype],
    )
    return out


###
### Autograd Function (Higham §10.6 augmented-matrix backward)
###
class _ExpmT18TritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor) -> torch.Tensor:
        out = _expm_t18_no_grad(A, out_dtype=A.dtype)
        ctx.save_for_backward(A)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None

        (A,) = ctx.saved_tensors
        B, N, _ = A.shape

        ### Build M = [[Aᵀ, G]; [0, Aᵀ]]  shape [B, 2N, 2N], fp32.
        ### Top-right block of exp(M) equals L(Aᵀ, G) = ∂L/∂A.
        A_T = A.float().transpose(-1, -2).contiguous()
        G   = grad_out.float().contiguous()

        M = torch.zeros(B, 2 * N, 2 * N, dtype=torch.float32, device=A.device)
        M[:, :N, :N] = A_T
        M[:, N:, N:] = A_T
        M[:, :N, N:] = G

        expM = _expm_t18_no_grad(M, out_dtype=torch.float32)
        return expM[:, :N, N:].to(A.dtype)


###
### Public API
###
def expm_t18_triton(A: torch.Tensor) -> torch.Tensor:
    """Triton T18 matrix exponential.

    Args:
        A: [B, N, N] fp32 / bf16 / fp16 tensor on CUDA.

    Returns:
        exp(A) with the same shape and dtype as A.
    """
    return _ExpmT18TritonFn.apply(A)
