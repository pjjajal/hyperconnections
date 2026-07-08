"""
Triton implementation of the optimised T18 matrix-exponential approximation.

Algorithm: Bader, Blanes & Casas, "Computing the Matrix Exponential with an
Optimized Taylor Polynomial Approximation", Mathematics 2019, 7, 1174.

Forward
-------
Single fused kernel.  Grid = (B,).  One CTA per batch element.

  - Loads A[b] and a fixed scaling factor inv_scale = 2^(-S), where S is the
    (constexpr) number of scaling-and-squaring steps passed by the caller. No
    per-input norm is computed — S alone sets both the scaling and the squaring
    count (accurate for ||A||_1 <= θ₁₈·2^S; smaller norms are over-scaled).
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
  - Applies exactly MAX_S=S squarings, unconditionally (no masking) — the
    squaring count is the constexpr S, matching inv_scale = 2^(-S).
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

from .numbers import (
    _a11, _a21, _a31,
    _b11, _b21, _b31, _b61,
    _b02, _b12, _b22, _b32, _b62,
    _b03, _b13, _b23, _b33, _b63,
    _b24, _b34, _b64,
)

### Convert to tl.constexpr(...)
_a11 = tl.constexpr(_a11); _a21 = tl.constexpr(_a21); _a31 = tl.constexpr(_a31)
_b11 = tl.constexpr(_b11); _b21 = tl.constexpr(_b21); _b31 = tl.constexpr(_b31); _b61 = tl.constexpr(_b61)
_b02 = tl.constexpr(_b02); _b12 = tl.constexpr(_b12); _b22 = tl.constexpr(_b22); _b32 = tl.constexpr(_b32); _b62 = tl.constexpr(_b62)
_b03 = tl.constexpr(_b03); _b13 = tl.constexpr(_b13); _b23 = tl.constexpr(_b23); _b33 = tl.constexpr(_b33); _b63 = tl.constexpr(_b63)
_b24 = tl.constexpr(_b24); _b34 = tl.constexpr(_b34); _b64 = tl.constexpr(_b64)

### Default number of scaling-and-squaring steps S (not a cap: the public fns
### accept any S >= 0, driving both inv_scale = 2^(-S) and the squaring count).
_MAX_S = 2

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


@triton.jit
def _matmul_nn(A, B, NP: tl.constexpr, n_idx, LOW_PREC: tl.constexpr = False):
    ### Use tl.dot (downcompiled to proper GEMM by Triton) for NP >= 16
    ### Otherwise, use FFMA on register tiles.
    if NP >= 16:
        if LOW_PREC:
            ### bf16/fp16 callers have ~1e-2 relative tolerance headroom, so feed
            ### the tensor cores bf16 operands (2x the TF32 rate on A100, and 4x
            ### IEEE-fp32 emulation) while KEEPING the fp32 accumulator via
            ### out_dtype.  fp32 callers (out_dtype=fp32, bench atol 5e-4/5e-3)
            ### fall through to the IEEE path below — bf16 inputs would blow that.
            return tl.dot(A.to(tl.bfloat16), B.to(tl.bfloat16), out_dtype=tl.float32)
        return tl.dot(A, B, input_precision="ieee", out_dtype=tl.float32)
    R = tl.zeros([NP, NP], dtype=tl.float32)
    for k in tl.static_range(NP):
        e_k   = (n_idx == k).to(tl.float32)            # [NP]
        col_k = tl.sum(A * e_k[None, :], axis=1)       # [NP]  = A[:, k]
        row_k = tl.sum(B * e_k[:, None], axis=0)       # [NP]  = B[k, :]
        R = R + col_k[:, None] * row_k[None, :]
    return R


@triton.jit
def _matmul_nn_alt(A, B, NP: tl.constexpr, n_idx):
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


###
### Helper for block-structured backward (or forward)
###
@triton.jit
def _pair_mul(D1, U1, D2, U2, NP: tl.constexpr, n_idx, LOW_PREC: tl.constexpr = False):
    ### Structured block product:
    ### [[D1, U1], [0, D1]] @ [[D2, U2], [0, D2]] = [[D1@D2, D1@U2 + U1@D2], [0, D1@D2]]
    D = _matmul_nn(D1, D2, NP, n_idx, LOW_PREC)
    U = _matmul_nn(D1, U2, NP, n_idx, LOW_PREC) + _matmul_nn(U1, D2, NP, n_idx, LOW_PREC)
    return D, U


@triton.jit
def _pair_mul_alt(D1, U1, D2, U2, R, NP: tl.constexpr, n_idx, LOW_PREC: tl.constexpr = False):
    ### Structured block product:
    ### [[D1, U1], [0, D1]] @ [[D2, U2], [0, D2]] = [[D1@D2, D1@U2 + U1@D2], [0, D1@D2]]
    D = _matmul_nn(D1, D2, NP, n_idx, LOW_PREC)
    U = _matmul_nn(D1, U2, NP, n_idx, LOW_PREC) + _matmul_nn(U1, D2, NP, n_idx, LOW_PREC)
    return D, U


###
### Alternate forward function for block matrix of form:
### M = [A G; 0 A];
###
@triton.autotune(configs=_FWD_CONFIGS, key=["NP"])
@triton.jit
def _expm_t18_structured_fwd(
    X_ptr, G_ptr, out_ptr,
    inv_scale,                           # scalar 2^(-MAX_S); no data-dependent s
    stride_x_b, stride_x_n1, stride_x_n2,
    stride_g_b, stride_g_n1, stride_g_n2,
    stride_o_b, stride_o_n1, stride_o_n2,
    N:         tl.constexpr,
    NP:        tl.constexpr,
    MAX_S:     tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    LOW_PREC:  tl.constexpr = False,
):
    pid_b  = tl.program_id(0)
    n_idx  = tl.arange(0, NP)
    n_mask = n_idx < N
    mask2d = n_mask[:, None] & n_mask[None, :]

    ### Load diagonal block X and upper block G.
    ### For backward, X should be A^T and G should be grad_out.
    x_off = (
        pid_b * stride_x_b
        + n_idx[:, None] * stride_x_n1
        + n_idx[None, :] * stride_x_n2
    )
    g_off = (
        pid_b * stride_g_b
        + n_idx[:, None] * stride_g_n1
        + n_idx[None, :] * stride_g_n2
    )

    X = tl.load(X_ptr + x_off, mask=mask2d, other=0.0).to(tl.float32)
    G = tl.load(G_ptr + g_off, mask=mask2d, other=0.0).to(tl.float32)

    ### Fixed scaling 2^(-MAX_S) (passed in), applied to the full block matrix
    ### M = [[X, G], [0, X]], not only X.
    D1 = X * inv_scale
    U1 = G * inv_scale

    ### Structured powers:
    ### M^1 = (D1, U1)
    ### M^2 = (D2, U2)
    ### M^3 = (D3, U3)
    ### M^6 = (D6, U6)
    D2, U2 = _pair_mul(D1, U1, D1, U1, NP, n_idx, LOW_PREC)
    D3, U3 = _pair_mul(D2, U2, D1, U1, NP, n_idx, LOW_PREC)
    D6, U6 = _pair_mul(D3, U3, D3, U3, NP, n_idx, LOW_PREC)

    ### Identity contributes only to diagonal blocks.
    eye = tl.where(n_idx[:, None] == n_idx[None, :], 1.0, 0.0)

    ### B1 = a11 M + a21 M^2 + a31 M^3
    DB1 = _a11 * D1 + _a21 * D2 + _a31 * D3
    UB1 = _a11 * U1 + _a21 * U2 + _a31 * U3

    ### B2 = b11 M + b21 M^2 + b31 M^3 + b61 M^6
    DB2 = _b11 * D1 + _b21 * D2 + _b31 * D3 + _b61 * D6
    UB2 = _b11 * U1 + _b21 * U2 + _b31 * U3 + _b61 * U6

    ### B3 = b02 I + b12 M + b22 M^2 + b32 M^3 + b62 M^6
    DB3 = _b02 * eye + _b12 * D1 + _b22 * D2 + _b32 * D3 + _b62 * D6
    UB3 =              _b12 * U1 + _b22 * U2 + _b32 * U3 + _b62 * U6

    ### B4 = b03 I + b13 M + b23 M^2 + b33 M^3 + b63 M^6
    DB4 = _b03 * eye + _b13 * D1 + _b23 * D2 + _b33 * D3 + _b63 * D6
    UB4 =              _b13 * U1 + _b23 * U2 + _b33 * U3 + _b63 * U6

    ### B5 = b24 M^2 + b34 M^3 + b64 M^6
    DB5 = _b24 * D2 + _b34 * D3 + _b64 * D6
    UB5 = _b24 * U2 + _b34 * U3 + _b64 * U6

    ### A9 = B1 @ B5 + B4
    DA9_tmp, UA9_tmp = _pair_mul(DB1, UB1, DB5, UB5, NP, n_idx, LOW_PREC)

    DA9 = DA9_tmp + DB4
    UA9 = UA9_tmp + UB4

    ### T18 = B2 + (B3 + A9) @ A9
    DC = DB3 + DA9
    UC = UB3 + UA9

    DCA9, UCA9 = _pair_mul(DC, UC, DA9, UA9, NP, n_idx, LOW_PREC)

    DT18 = DB2 + DCA9
    UT18 = UB2 + UCA9

    ### Structured repeated squaring: (D, U)^2 = (D @ D, D @ U + U @ D).
    ### Exactly MAX_S squarings (unconditional), matching inv_scale = 2^(-MAX_S).
    for i in tl.static_range(MAX_S):
        DT18, UT18 = _pair_mul(DT18, UT18, DT18, UT18, NP, n_idx, LOW_PREC)

    ### Store only the upper-right block, i.e. the Frechet derivative block.
    o_off = (
        pid_b * stride_o_b
        + n_idx[:, None] * stride_o_n1
        + n_idx[None, :] * stride_o_n2
    )

    tl.store(out_ptr + o_off, UT18.to(OUT_DTYPE), mask=mask2d)


###
### Inner Triton Forward
###
@triton.autotune(configs=_FWD_CONFIGS, key=["NP"])
@triton.jit
def _expm_t18_fwd(
    A_ptr, out_ptr,
    inv_scale,                           # scalar 2^(-MAX_S); no data-dependent s
    stride_a_b, stride_a_n1, stride_a_n2,
    stride_o_b, stride_o_n1, stride_o_n2,
    N:         tl.constexpr,             # logical matrix size
    NP:        tl.constexpr,             # next_pow2(N) — Triton tile size
    MAX_S:     tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    LOW_PREC:  tl.constexpr = False,
):
    pid_b  = tl.program_id(0)
    n_idx  = tl.arange(0, NP)
    n_mask = n_idx < N
    mask2d = n_mask[:, None] & n_mask[None, :]

    ### Load A[b], upcast, and apply the fixed scaling 1/2^MAX_S (passed in).
    a_off = (
        pid_b * stride_a_b
        + n_idx[:, None] * stride_a_n1
        + n_idx[None, :] * stride_a_n2
    )
    A = tl.load(A_ptr + a_off, mask=mask2d, other=0.0).to(tl.float32)
    A = A * inv_scale

    ### A^2, A^3, A^6 — see _matmul_nn for why we don't use the naïve
    ### broadcast+sum contraction.
    A2 = _matmul_nn(A,  A,  NP, n_idx, LOW_PREC)
    A3 = _matmul_nn(A2, A,  NP, n_idx, LOW_PREC)
    A6 = _matmul_nn(A3, A3, NP, n_idx, LOW_PREC)

    ### Identity (NP×NP).  Padded diag ones live only in the [N:,N:] block.
    eye = tl.where(n_idx[:, None] == n_idx[None, :], 1.0, 0.0)

    B1 = _a11 * A            + _a21 * A2 + _a31 * A3
    B2 = _b11 * A            + _b21 * A2 + _b31 * A3 + _b61 * A6
    B3 = _b02 * eye + _b12 * A + _b22 * A2 + _b32 * A3 + _b62 * A6
    B4 = _b03 * eye + _b13 * A + _b23 * A2 + _b33 * A3 + _b63 * A6
    B5 = _b24 * A2 + _b34 * A3 + _b64 * A6

    ### A9 = B1 @ B5 + B4
    A9 = _matmul_nn(B1, B5, NP, n_idx, LOW_PREC) + B4

    ### T18 = B2 + (B3 + A9) @ A9
    T18 = B2 + _matmul_nn(B3 + A9, A9, NP, n_idx, LOW_PREC)

    ### Exactly MAX_S squarings (unconditional), matching inv_scale = 2^(-MAX_S).
    for i in tl.static_range(MAX_S):
        T18 = _matmul_nn(T18, T18, NP, n_idx, LOW_PREC)

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


###
### Forward
###
def _expm_t18_no_grad(A: torch.Tensor, out_dtype: torch.dtype, S: int = _MAX_S) -> torch.Tensor:
    """Triton T18 forward.  No autograd wrapping; used by the autograd
    Function for both forward and the augmented-matrix backward.

    Fixed scaling-and-squaring: scale by 2^(-S) and square exactly S times.
    Correct for ||A||_1 <= theta_18 * 2^S; smaller norms are over-scaled.
    """
    if not A.is_cuda:
        raise RuntimeError("expm_t18_triton requires CUDA tensors")
    if A.dim() != 3 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"expected [B, N, N], got {tuple(A.shape)}")
    if out_dtype not in _TORCH_TO_TL:
        raise ValueError(f"unsupported out_dtype {out_dtype}")

    B, N, _ = A.shape

    A_fp32 = A.to(torch.float32).contiguous()

    ### Fixed scaling factor 2^(-S) — no norm reduction, no host sync.
    inv_scale = 2.0 ** (-S)

    out = torch.empty(B, N, N, dtype=out_dtype, device=A.device)
    NP  = _next_pow2(N)

    _expm_t18_fwd[(B,)](
        A_fp32, out,
        inv_scale,
        *A_fp32.stride(), *out.stride(),
        N=N, NP=NP, MAX_S=S,
        OUT_DTYPE=_TORCH_TO_TL[out_dtype],
        LOW_PREC=out_dtype in (torch.float16, torch.bfloat16),
    )
    return out


###
### Forward
###
def _expm_t18_structure_no_grad(A_T: torch.Tensor, G: torch.Tensor, out_dtype: torch.dtype, S: int = _MAX_S) -> torch.Tensor:
    """Triton T18 forward.  No autograd wrapping; used by the autograd
    Function for both forward and the augmented-matrix backward.

    Fixed scaling-and-squaring: scale by 2^(-S) and square exactly S times.
    Correct for ||M||_1 <= theta_18 * 2^S (M = [[X, G], [0, X]]).
    """
    if not A_T.is_cuda:
        raise RuntimeError("expm_t18_structure_no_grad requires CUDA tensors")
    if A_T.dim() != 3 or A_T.shape[-1] != A_T.shape[-2]:
        raise ValueError(f"expected [B, N, N], got {tuple(A_T.shape)}")
    if out_dtype not in _TORCH_TO_TL:
        raise ValueError(f"unsupported out_dtype {out_dtype}")

    B, N, _ = A_T.shape
    # X = A_T
    X = A_T.to(torch.float32).contiguous()

    ### Fixed scaling factor 2^(-S) — no norm reduction, no host sync.
    inv_scale = 2.0 ** (-S)

    out = torch.empty_like(G)
    NP  = _next_pow2(N)

    _expm_t18_structured_fwd[(B,)](
        X, G, out,
        inv_scale,
        *X.stride(), *G.stride(), *out.stride(),
        N=N, NP=NP, MAX_S=S,
        OUT_DTYPE=_TORCH_TO_TL[out_dtype],
        LOW_PREC=out_dtype in (torch.float16, torch.bfloat16),
    )
    return out


###
### Autograd Function (Higham §10.6 augmented-matrix backward)
###
class _ExpmT18TritonFn(torch.autograd.Function):
    """
    dL/dA = L_exp(A^T, grad_E) + L_phi1(A^T, grad_psi)

    Part 1 -- L_exp(A^T, grad_E):
    The standard block formula for the Fréchet derivative of the
    matrix exponential gives

        exp([[A, G],
            [0, A]])
        =
        [[exp(A), L_exp(A,G)],
        [0,      exp(A)]]

    See Najfeld and Havel (1995), Mathias (1992), and Higham,
    Functions of Matrices, Sec. 10.6.
    """
    @staticmethod
    def forward(ctx, A: torch.Tensor, S: int) -> torch.Tensor:
        out = _expm_t18_no_grad(A, out_dtype=A.dtype, S=S)
        ctx.save_for_backward(A)
        ctx.S = S
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None

        (A,) = ctx.saved_tensors
        B, N, _ = A.shape

        ### Build M = [[Aᵀ, G]; [0, Aᵀ]]  shape [B, 2N, 2N], fp32.
        ### Top-right block of exp(M) equals L(Aᵀ, G) = ∂L/∂A.
        A_T = A.float().transpose(-1, -2).contiguous()
        G   = grad_out.float().contiguous()

        ### STANDARD BWD
        # M = torch.zeros(B, 2 * N, 2 * N, dtype=torch.float32, device=A.device)
        # M[:, :N, :N] = A_T
        # M[:, N:, N:] = A_T
        # M[:, :N, N:] = G

        # dExpM = _expm_t18_no_grad(M, out_dtype=A.dtype)
        # return dExpM[:, :N, N:]

        ### BLOCK-STRUTURE BWD
        dExpM = _expm_t18_structure_no_grad(A_T, G, out_dtype=A.dtype, S=ctx.S)
        return dExpM, None


###
### Public API
###
def expm_t18_triton(A: torch.Tensor, S: int = _MAX_S) -> torch.Tensor:
    """Triton T18 matrix exponential.

    Args:
        A: [B, N, N] fp32 / bf16 / fp16 tensor on CUDA.
        S: number of scaling-and-squaring steps (constexpr in-kernel). A is
           scaled by 2^(-S) and squared exactly S times, so the result is
           accurate for ||A||_1 <= theta_18 * 2^S (~3.01*2^S); smaller norms are
           over-scaled. Default _MAX_S=2.

    Returns:
        exp(A) with the same shape and dtype as A.
    """
    if not A.is_cuda:
        raise RuntimeError("expm_t18_triton requires CUDA tensors")
    if S < 0:
        raise ValueError(f"S must be >= 0, got {S}")

    return _ExpmT18TritonFn.apply(A, S)


###
### Helper for the augmented forcing matrix exp([[A, I]; [0, 0]])
###
@triton.jit
def _blk_mul_c(D1, U1, c1, D2, U2, c2, NP: tl.constexpr, n_idx, LOW_PREC: tl.constexpr = False):
    """Structured block product for the augmented forcing matrix.

    A block triple (D, U, c) represents [[D, U], [0, c*I]].  The product

      [[D1,U1],[0,c1 I]] @ [[D2,U2],[0,c2 I]]
        = [[D1@D2,  D1@U2 + c2*U1],  [0,  c1*c2 I]]

    The cross-term is the scalar scaling c2*U1 — NOT the matrix product
    U1@D2 used by _pair_mul.  _pair_mul is correct only for the Frechet
    structure [[D,U],[0,D]] (lower-right block = D); here the lower-right
    block is c*I, so the scalar c must be tracked and propagated.
    """
    D = _matmul_nn(D1, D2, NP, n_idx, LOW_PREC)
    U = _matmul_nn(D1, U2, NP, n_idx, LOW_PREC) + c2 * U1
    c = c1 * c2
    return D, U, c


###
### Triton kernel for exp([[A, I]; [0, 0]])
###
### Tracks each block as a triple (D, U, c) representing [[D, U], [0, c*I]]:
###   D -> exp(A),  U -> phi_1(A).
###
### Z = [[A, I], [0, 0]] has c = 0, but the identity injected by the
### polynomial has c = 1, so the intermediate blocks B3, B4, A9, T18 carry
### a nonzero scalar.  The block product therefore goes through _blk_mul_c
### (cross-term c2*U1), not _pair_mul (cross-term U1@D2).
###
@triton.autotune(configs=_FWD_CONFIGS, key=["NP"])
@triton.jit
def _expm_t18_augmented_fwd(
    A_ptr, E_ptr, psi_ptr,
    inv_scale,                           # scalar 2^(-MAX_S); no data-dependent s
    stride_a_b, stride_a_n1, stride_a_n2,
    stride_e_b, stride_e_n1, stride_e_n2,
    stride_p_b, stride_p_n1, stride_p_n2,
    N:         tl.constexpr,
    NP:        tl.constexpr,
    MAX_S:     tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    LOW_PREC:  tl.constexpr = False,
):
    """Structured T18 forward for Z = [[A, I], [0, 0]].

    Block triples (D, U, c) represent [[D, U], [0, c*I]].  Z itself is
    (A*inv_scale, I*inv_scale, 0); the identity is (I, 0, 1).  Products go
    through _blk_mul_c so the scalar c is propagated correctly.
    Stores E = DT18 (approx exp(A)) and psi = UT18 (approx phi_1(A)).
    """
    pid_b  = tl.program_id(0)
    n_idx  = tl.arange(0, NP)
    n_mask = n_idx < N
    mask2d = n_mask[:, None] & n_mask[None, :]

    a_off = (
        pid_b * stride_a_b
        + n_idx[:, None] * stride_a_n1
        + n_idx[None, :] * stride_a_n2
    )
    A = tl.load(A_ptr + a_off, mask=mask2d, other=0.0).to(tl.float32)

    eye = tl.where(n_idx[:, None] == n_idx[None, :], 1.0, 0.0)

    ### Z = (D1, U1, 0):  scaled augmented matrix [[A, I], [0, 0]] / scale.
    D1 = A * inv_scale
    U1 = eye * inv_scale
    c1 = 0.0

    ### Structured powers of Z.  Each has c = 0 (0^k = 0 for k >= 1), so the
    ### c2*U1 cross-term vanishes for every Z-power product.
    D2, U2, c2 = _blk_mul_c(D1, U1, c1, D1, U1, c1, NP, n_idx, LOW_PREC)
    D3, U3, c3 = _blk_mul_c(D2, U2, c2, D1, U1, c1, NP, n_idx, LOW_PREC)
    D6, U6, c6 = _blk_mul_c(D3, U3, c3, D3, U3, c3, NP, n_idx, LOW_PREC)

    ### Polynomial blocks.  The identity carries c = 1, so each block's
    ### scalar is its identity coefficient (a01=b01=b04=b14=0, hence
    ### cB1 = cB2 = cB5 = 0; cB3 = b02; cB4 = b03).
    ### B1 = a11 Z + a21 Z^2 + a31 Z^3
    DB1 = _a11 * D1 + _a21 * D2 + _a31 * D3
    UB1 = _a11 * U1 + _a21 * U2 + _a31 * U3
    cB1 = 0.0

    ### B2 = b11 Z + b21 Z^2 + b31 Z^3 + b61 Z^6
    DB2 = _b11 * D1 + _b21 * D2 + _b31 * D3 + _b61 * D6
    UB2 = _b11 * U1 + _b21 * U2 + _b31 * U3 + _b61 * U6
    cB2 = 0.0

    ### B3 = b02 I + b12 Z + b22 Z^2 + b32 Z^3 + b62 Z^6
    DB3 = _b02 * eye + _b12 * D1 + _b22 * D2 + _b32 * D3 + _b62 * D6
    UB3 =              _b12 * U1 + _b22 * U2 + _b32 * U3 + _b62 * U6
    cB3 = _b02

    ### B4 = b03 I + b13 Z + b23 Z^2 + b33 Z^3 + b63 Z^6
    DB4 = _b03 * eye + _b13 * D1 + _b23 * D2 + _b33 * D3 + _b63 * D6
    UB4 =              _b13 * U1 + _b23 * U2 + _b33 * U3 + _b63 * U6
    cB4 = _b03

    ### B5 = b24 Z^2 + b34 Z^3 + b64 Z^6
    DB5 = _b24 * D2 + _b34 * D3 + _b64 * D6
    UB5 = _b24 * U2 + _b34 * U3 + _b64 * U6
    cB5 = 0.0

    ### A9 = B1 @ B5 + B4
    DA9_tmp, UA9_tmp, cA9_tmp = _blk_mul_c(DB1, UB1, cB1, DB5, UB5, cB5, NP, n_idx, LOW_PREC)
    DA9 = DA9_tmp + DB4
    UA9 = UA9_tmp + UB4
    cA9 = cA9_tmp + cB4

    ### T18 = B2 + (B3 + A9) @ A9
    DC = DB3 + DA9
    UC = UB3 + UA9
    cC = cB3 + cA9
    DCA9, UCA9, cCA9 = _blk_mul_c(DC, UC, cC, DA9, UA9, cA9, NP, n_idx, LOW_PREC)
    DT18 = DB2 + DCA9
    UT18 = UB2 + UCA9
    cT18 = cB2 + cCA9

    ### Repeated squaring: (D, U, c)^2 = (D@D, D@U + c*U, c^2).
    ### Exactly MAX_S squarings (unconditional), matching inv_scale = 2^(-MAX_S).
    for i in tl.static_range(MAX_S):
        DT18, UT18, cT18 = _blk_mul_c(DT18, UT18, cT18, DT18, UT18, cT18, NP, n_idx, LOW_PREC)

    ### Store exp(A) and phi_1(A)
    e_off = (
        pid_b * stride_e_b
        + n_idx[:, None] * stride_e_n1
        + n_idx[None, :] * stride_e_n2
    )
    p_off = (
        pid_b * stride_p_b
        + n_idx[:, None] * stride_p_n1
        + n_idx[None, :] * stride_p_n2
    )
    tl.store(E_ptr   + e_off, DT18.to(OUT_DTYPE), mask=mask2d)
    tl.store(psi_ptr + p_off, UT18.to(OUT_DTYPE), mask=mask2d)


###
### Python launcher (no autograd)
###
def _expm_t18_augmented_no_grad(
    A: torch.Tensor, out_dtype: torch.dtype, S: int = _MAX_S
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton T18 augmented forward. Returns (exp(A), phi_1(A)). No autograd.

    Fixed scaling-and-squaring: scale by 2^(-S) and square exactly S times.
    Correct for ||A||_1 <= theta_18 * 2^S; smaller norms are over-scaled.
    """
    if not A.is_cuda:
        raise RuntimeError("_expm_t18_augmented_no_grad requires CUDA tensors")
    if A.dim() != 3 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"expected [B, N, N], got {tuple(A.shape)}")
    if out_dtype not in _TORCH_TO_TL:
        raise ValueError(f"unsupported out_dtype {out_dtype}")

    B, N, _ = A.shape
    A_fp32 = A.to(torch.float32).contiguous()

    ### Fixed scaling factor 2^(-S) — no norm reduction, no host sync.
    inv_scale = 2.0 ** (-S)

    E   = torch.empty(B, N, N, dtype=out_dtype, device=A.device)
    psi = torch.empty(B, N, N, dtype=out_dtype, device=A.device)
    NP  = _next_pow2(N)

    _expm_t18_augmented_fwd[(B,)](
        A_fp32, E, psi,
        inv_scale,
        *A_fp32.stride(), *E.stride(), *psi.stride(),
        N=N, NP=NP, MAX_S=S,
        OUT_DTYPE=_TORCH_TO_TL[out_dtype],
        LOW_PREC=out_dtype in (torch.float16, torch.bfloat16),
    )
    return E, psi


###
### Autograd Function
###
class _ExpmT18AugmentedTritonFn(torch.autograd.Function):
    """
    Backward decomposes into two adjoint Fréchet derivatives:

    dL/dA = L_exp(A^T, grad_E) + L_phi1(A^T, grad_psi)

    Part 1 -- L_exp(A^T, grad_E):
    The standard block formula for the Fréchet derivative of the
    matrix exponential gives

        exp([[A, G],
            [0, A]])
        =
        [[exp(A), L_exp(A,G)],
        [0,      exp(A)]]

    See Najfeld and Havel (1995), Mathias (1992), and Higham,
    Functions of Matrices, Sec. 10.6.

    Part 2 -- L_phi1(A^T, grad_psi):
    Since phi_1(A) = integral_0^1 exp(theta A) d theta,
    its Fréchet derivative is

        L_phi1(A,G)
            = integral_{s,t >= 0, s+t <= 1}
                exp(sA) G exp(tA) ds dt.

    This follows from the same Van Loan / block-upper-triangular
    matrix exponential identity for integrals involving matrix
    exponentials. In particular, for

        M = [[A^T,  G,   0],
            [0,    A^T, I],
            [0,    0,   0]],

    the (1,3) block of exp(M) is

        integral_{s,t >= 0, s+t <= 1}
            exp(sA^T) G exp(tA^T) ds dt
        = L_phi1(A^T, G).
    """
    @staticmethod
    def forward(ctx, A: torch.Tensor, S: int):
        E, psi = _expm_t18_augmented_no_grad(A, out_dtype=A.dtype, S=S)
        ctx.save_for_backward(A.float().transpose(-1, -2).contiguous())
        ctx.input_dtype = A.dtype
        ctx.S = S
        return E, psi

    @staticmethod
    def backward(ctx, grad_E: torch.Tensor, grad_psi: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None

        (A_T,) = ctx.saved_tensors
        out_dtype = ctx.input_dtype
        S = ctx.S
        B, N, _ = A_T.shape
        G_E   = grad_E.float().contiguous()
        G_psi = grad_psi.float().contiguous()

        ### Part 1: L_exp(A^T, grad_E) via block-structured backward at N.
        dA_E = _expm_t18_structure_no_grad(A_T, G_E, out_dtype=out_dtype, S=S)

        ### Part 2: L_phi_1(A^T, grad_psi) via 3N x 3N augmented exp.
        ### M = [[A^T, G_psi, 0], [0, A^T, I], [0, 0, 0]] — (1,3) block of
        ### exp(M) is integral_{s+t<=1} exp(sA^T) G_psi exp(tA^T) ds dt
        ### = L_phi_1(A^T, G_psi).
        M = torch.zeros(B, 3 * N, 3 * N, dtype=torch.float32, device=A_T.device)
        M[:,  :N,      :N   ] = A_T
        M[:,  :N,     N:2*N ] = G_psi
        M[:, N:2*N,   N:2*N ] = A_T
        eye = torch.eye(N, dtype=torch.float32, device=A_T.device)
        M[:, N:2*N, 2*N:3*N] = eye

        ### NOTE: Revisit this at some point... Debug it!
        # expM = torch.linalg.matrix_exp(M).to(out_dtype)
        expM = _expm_t18_no_grad(M, out_dtype=out_dtype, S=S)
        dA_psi = expM[:, :N, 2*N:3*N]

        return dA_E + dA_psi, None


###
### Public API
###
def expm_t18_block_triton(A: torch.Tensor, S: int = _MAX_S) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton T18 augmented matrix exponential.

    Computes exp([[A, I]; [0, 0]]) via the block-structured T18 polynomial,
    returning (exp(A), phi_1(A)) without materialising the 2N x 2N matrix.

    phi_1(A) = integral_0^1 exp(theta A) d theta is the matrix phi function
    needed for exact integration of dx/dt = Ax + f over one step:
        x(1) = exp(A) x(0) + phi_1(A) f.

    Args:
        A: [B, N, N] fp32 / bf16 / fp16 tensor on CUDA.
        S: number of scaling-and-squaring steps (constexpr in-kernel). A is
           scaled by 2^(-S) and squared exactly S times, so the result is
           accurate for ||A||_1 <= theta_18 * 2^S (~3.01*2^S); smaller norms are
           over-scaled. S also drives the backward (its augmented matrices must
           likewise satisfy the bound). Default _MAX_S=2.

    Returns:
        E:   [B, N, N], approximation to exp(A).
        psi: [B, N, N], approximation to phi_1(A).
    """
    if not A.is_cuda:
        raise RuntimeError("expm_t18_block_triton requires CUDA tensors")
    if S < 0:
        raise ValueError(f"S must be >= 0, got {S}")
    return _ExpmT18AugmentedTritonFn.apply(A, S)