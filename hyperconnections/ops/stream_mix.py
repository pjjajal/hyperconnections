"""
Fused stream-mixing Triton kernel.

Computes  out = Phi @ x + Y                               [no-proj]
or        out = Phi @ x + (v - Phi@v)(v^T x) + Y         [proj]

where:
  Phi  [B, N, N]   transition matrix
  x    [B, N, D]   stream state
  Y    [B, N, D]   source term
  v    [B, N]      unit-norm projection direction (None → no-proj)
  out  [B, N, D]

Accumulation is always in float32 regardless of input dtype.

Kernel layout (forward and bwd_dx):
  Grid: (B * N, cdiv(D, BLOCK_D))
  One program per output row (b, n_out), vectorised over a D-tile.
  N_STREAMS is tl.constexpr so the inner loop is fully unrolled.

Kernel layout (bwd_dPhi):
  Grid: (B * N * N,)
  One program per scalar grad_Phi[b, n1, n2], loops over D.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

_FWD_CONFIGS = [
    triton.Config({"BLOCK_D": 32},  num_warps=2),
    triton.Config({"BLOCK_D": 64},  num_warps=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 256}, num_warps=4),
]


@triton.autotune(configs=_FWD_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_fwd(
    Phi_ptr, x_ptr, Y_ptr, out_ptr, v_ptr,
    B, D,
    stride_phi_b, stride_phi_n1, stride_phi_n2,
    stride_x_b,   stride_x_n,   stride_x_d,
    stride_y_b,   stride_y_n,   stride_y_d,
    stride_o_b,   stride_o_n,   stride_o_d,
    stride_v_b,   stride_v_n,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_bn = tl.program_id(0)
    pid_d  = tl.program_id(1)
    b  = pid_bn // N_STREAMS
    n1 = pid_bn %  N_STREAMS

    d_off  = pid_d * BLOCK_D
    d_idx  = d_off + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    if USE_PROJ:
        alpha = tl.zeros([BLOCK_D], dtype=tl.float32)  # v^T x  [BLOCK_D]
        phi_v = 0.0   # Phi[n1,:] @ v  (scalar; promoted to tl.float32 on first iter)
        v_n1  = tl.load(v_ptr + b * stride_v_b + n1 * stride_v_n).to(tl.float32)

    # Inner loop — fully unrolled at compile time (N_STREAMS is constexpr)
    for n2 in tl.static_range(N_STREAMS):
        phi_val = tl.load(
            Phi_ptr + b * stride_phi_b + n1 * stride_phi_n1 + n2 * stride_phi_n2
        ).to(tl.float32)

        x_vec = tl.load(
            x_ptr + b * stride_x_b + n2 * stride_x_n + d_idx * stride_x_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        acc = acc + phi_val * x_vec

        if USE_PROJ:
            v_n2  = tl.load(v_ptr + b * stride_v_b + n2 * stride_v_n).to(tl.float32)
            alpha = alpha + v_n2 * x_vec
            phi_v = phi_v + phi_val * v_n2

    if USE_PROJ:
        acc = acc + (v_n1 - phi_v) * alpha

    y_vec = tl.load(
        Y_ptr + b * stride_y_b + n1 * stride_y_n + d_idx * stride_y_d,
        mask=d_mask, other=0.0,
    ).to(tl.float32)
    acc = acc + y_vec

    tl.store(
        out_ptr + b * stride_o_b + n1 * stride_o_n + d_idx * stride_o_d,
        acc,          # float32 → pointer dtype cast handled by Triton
        mask=d_mask,
    )


# ---------------------------------------------------------------------------
# Backward kernel: grad_x
#
# grad_x[b, n2, d] = (Phi^T @ G)[b, n2, d]                                [no-proj]
#                  = (Phi^T @ G)[b, n2, d] + v[b,n2] * beta[b,d]          [proj]
#
# where beta[b,d] = Σ_n1( G[b,n1,d] * (v[b,n1] - phi_v[b,n1]) )
#       phi_v[b,n1] = Σ_n2'( Phi[b,n1,n2'] * v[b,n2'] )
# ---------------------------------------------------------------------------

@triton.autotune(configs=_FWD_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_bwd_dx(
    G_ptr, Phi_ptr, v_ptr, grad_x_ptr,
    B, D,
    stride_g_b,   stride_g_n,   stride_g_d,
    stride_phi_b, stride_phi_n1, stride_phi_n2,
    stride_gx_b,  stride_gx_n,  stride_gx_d,
    stride_v_b,   stride_v_n,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_bn = tl.program_id(0)
    pid_d  = tl.program_id(1)
    b  = pid_bn // N_STREAMS
    n2 = pid_bn %  N_STREAMS   # this program produces grad_x[b, n2, :]

    d_off  = pid_d * BLOCK_D
    d_idx  = d_off + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    acc  = tl.zeros([BLOCK_D], dtype=tl.float32)
    if USE_PROJ:
        beta = tl.zeros([BLOCK_D], dtype=tl.float32)
        v_n2 = tl.load(v_ptr + b * stride_v_b + n2 * stride_v_n).to(tl.float32)

    for n1 in tl.static_range(N_STREAMS):
        # Phi^T term: Phi[n1, n2] (note: row=n1, col=n2)
        phi_n1_n2 = tl.load(
            Phi_ptr + b * stride_phi_b + n1 * stride_phi_n1 + n2 * stride_phi_n2
        ).to(tl.float32)

        g_vec = tl.load(
            G_ptr + b * stride_g_b + n1 * stride_g_n + d_idx * stride_g_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        acc = acc + phi_n1_n2 * g_vec

        if USE_PROJ:
            # phi_v[n1] = Σ_n2p( Phi[b,n1,n2p] * v[b,n2p] )
            phi_v_n1 = 0.0
            for n2p in tl.static_range(N_STREAMS):
                p = tl.load(
                    Phi_ptr + b * stride_phi_b + n1 * stride_phi_n1 + n2p * stride_phi_n2
                ).to(tl.float32)
                vp = tl.load(v_ptr + b * stride_v_b + n2p * stride_v_n).to(tl.float32)
                phi_v_n1 = phi_v_n1 + p * vp

            v_n1 = tl.load(v_ptr + b * stride_v_b + n1 * stride_v_n).to(tl.float32)
            beta = beta + (v_n1 - phi_v_n1) * g_vec

    if USE_PROJ:
        acc = acc + v_n2 * beta

    tl.store(
        grad_x_ptr + b * stride_gx_b + n2 * stride_gx_n + d_idx * stride_gx_d,
        acc,
        mask=d_mask,
    )


# ---------------------------------------------------------------------------
# Backward kernel: grad_Phi
#
# grad_Phi[b, n1, n2] = Σ_d( G[b,n1,d] * x_eff[b,n2,d] )
#
# where x_eff = x                              [no-proj]
#       x_eff = x - v[n2] * alpha             [proj]
#       alpha[b,d] = Σ_n( v[b,n] * x[b,n,d] )
# ---------------------------------------------------------------------------

_DPHI_CONFIGS = [
    triton.Config({"BLOCK_D": 64},  num_warps=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4),
    triton.Config({"BLOCK_D": 256}, num_warps=4),
]


@triton.autotune(configs=_DPHI_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_bwd_dPhi(
    G_ptr, x_ptr, v_ptr, grad_Phi_ptr,
    B, D,
    stride_g_b,   stride_g_n,   stride_g_d,
    stride_x_b,   stride_x_n,   stride_x_d,
    stride_v_b,   stride_v_n,
    stride_gP_b,  stride_gP_n1, stride_gP_n2,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid = tl.program_id(0)
    b  = pid // (N_STREAMS * N_STREAMS)
    n1 = (pid // N_STREAMS) % N_STREAMS
    n2 = pid % N_STREAMS

    if USE_PROJ:
        v_n2 = tl.load(v_ptr + b * stride_v_b + n2 * stride_v_n).to(tl.float32)

    dp_acc = 0.0  # scalar accumulator (promoted to tl.float32 on first iter)

    n_blocks = tl.cdiv(D, BLOCK_D)
    for i in range(n_blocks):
        d_off  = i * BLOCK_D
        d_idx  = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D

        g_vec = tl.load(
            G_ptr + b * stride_g_b + n1 * stride_g_n + d_idx * stride_g_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        x_vec = tl.load(
            x_ptr + b * stride_x_b + n2 * stride_x_n + d_idx * stride_x_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        if USE_PROJ:
            # alpha[d] = Σ_n( v[n] * x[b,n,d] )
            alpha = tl.zeros([BLOCK_D], dtype=tl.float32)
            for n in tl.static_range(N_STREAMS):
                v_n = tl.load(v_ptr + b * stride_v_b + n * stride_v_n).to(tl.float32)
                x_n = tl.load(
                    x_ptr + b * stride_x_b + n * stride_x_n + d_idx * stride_x_d,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)
                alpha = alpha + v_n * x_n
            x_vec = x_vec - v_n2 * alpha   # x_eff

        dp_acc = dp_acc + tl.sum(g_vec * x_vec, axis=0)

    tl.store(
        grad_Phi_ptr + b * stride_gP_b + n1 * stride_gP_n1 + n2 * stride_gP_n2,
        dp_acc,
    )


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------

def _make_v_arg(v: torch.Tensor | None, B: int, N: int, device, dtype):
    """Return a safe dummy tensor when v is None so we can always pass strides."""
    if v is not None:
        return v.to(dtype=torch.float32).contiguous()
    return torch.zeros(B, N, dtype=torch.float32, device=device)


def _launch_fwd(Phi, x, Y, v, out):
    B, N, D = x.shape
    use_proj = v is not None
    v_arg = _make_v_arg(v, B, N, x.device, x.dtype)

    grid = lambda meta: (B * N, triton.cdiv(D, meta["BLOCK_D"]))
    _stream_mix_fwd[grid](
        Phi, x, Y, out, v_arg,
        B, D,
        *Phi.stride(), *x.stride(), *Y.stride(), *out.stride(), *v_arg.stride(),
        N_STREAMS=N,
        USE_PROJ=use_proj,
    )


def _launch_bwd_dx(G, Phi, v, grad_x, N):
    B, _, D = G.shape
    use_proj = v is not None
    v_arg = _make_v_arg(v, B, N, G.device, G.dtype)

    grid = lambda meta: (B * N, triton.cdiv(D, meta["BLOCK_D"]))
    _stream_mix_bwd_dx[grid](
        G, Phi, v_arg, grad_x,
        B, D,
        *G.stride(), *Phi.stride(), *grad_x.stride(), *v_arg.stride(),
        N_STREAMS=N,
        USE_PROJ=use_proj,
    )


def _launch_bwd_dPhi(G, x, v, grad_Phi, N):
    B, _, D = G.shape
    use_proj = v is not None
    v_arg = _make_v_arg(v, B, N, G.device, G.dtype)

    grid = lambda meta: (B * N * N,)
    _stream_mix_bwd_dPhi[grid](
        G, x, v_arg, grad_Phi,
        B, D,
        *G.stride(), *x.stride(), *v_arg.stride(), *grad_Phi.stride(),
        N_STREAMS=N,
        USE_PROJ=use_proj,
    )


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _StreamMixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Phi, x, Y, v):
        # Ensure contiguous float32 inputs for the kernels
        Phi_c = Phi.float().contiguous()
        x_c   = x.float().contiguous()
        Y_c   = Y.float().contiguous()
        v_c   = v.float().contiguous() if v is not None else None

        out = torch.empty_like(x_c)
        _launch_fwd(Phi_c, x_c, Y_c, v_c, out)

        ctx.save_for_backward(Phi_c, x_c, v_c)
        ctx.orig_dtype = x.dtype
        # Return in the original dtype
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        Phi, x, v = ctx.saved_tensors
        B, N, D   = x.shape
        use_proj  = v is not None

        G = grad_out.float().contiguous()

        # --- grad_x ---
        grad_x = torch.empty_like(x)
        _launch_bwd_dx(G, Phi, v, grad_x, N)

        # --- grad_Phi ---
        grad_Phi = torch.empty(B, N, N, dtype=torch.float32, device=x.device)
        _launch_bwd_dPhi(G, x, v, grad_Phi, N)

        # grad_Y = grad_out (identity)
        grad_Y = grad_out

        # grad_v: not yet implemented
        # TODO: grad_v[b,k] = dot(G[b,k,:], alpha) - (Phi^T @ rho)[b,k] + dot(x[b,k,:], beta)
        grad_v = None

        return (
            grad_Phi.to(Phi.dtype) if ctx.needs_input_grad[0] else None,
            grad_x.to(ctx.orig_dtype) if ctx.needs_input_grad[1] else None,
            grad_Y if ctx.needs_input_grad[2] else None,
            grad_v,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream_mix_add(
    Phi: torch.Tensor,
    x: torch.Tensor,
    Y: torch.Tensor,
    v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused  out = Phi @ x + Y  (or projected variant).

    Args:
        Phi: [B, N, N] transition matrix.
        x:   [B, N, D] stream state.
        Y:   [B, N, D] source term.
        v:   [B, N]    unit-norm projection direction, or None.

    Returns:
        out: [B, N, D]

    Note:
        Gradient w.r.t. v is not yet implemented; ensure v does not
        require grad when calling through autograd.
    """
    if not x.is_cuda:
        raise RuntimeError("stream_mix_add requires CUDA tensors")
    B, N, D = x.shape
    assert Phi.shape == (B, N, N)
    assert Y.shape   == (B, N, D)
    if v is not None:
        assert v.shape == (B, N)
    return _StreamMixFn.apply(Phi, x, Y, v)
