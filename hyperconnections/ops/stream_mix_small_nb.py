"""
Fused stream-mixing Triton kernel — small-N×B variant.

Computes  out = Phi @ x + Y                               [no-proj]
or        out = Phi @ x + (v - Phi@v)(v^T x) + Y         [proj]

where:
  Phi  [B, N, N]   transition matrix
  x    [B, N, D]   stream state
  Y    [B, N, D]   source term
  v    [B, N]      unit-norm projection direction (None → no-proj)
  out  [B, N, D]

Accumulation is always in float32 regardless of input dtype.

This variant should be dispatched when either N < 16 or the x footprint
(B*N*D*elem_bytes) fits within ~75% of L2 cache, so cross-CTA reuse of
x[b, n2, d_tile] across the N programs per batch element is served from
L2 without explicit shared memory staging.

Kernel layout (forward and bwd_dx):
  Grid: (B * N, cdiv(D, BLOCK_D))
  One program per output row (b, n_out), vectorised over a D-tile.
  N_STREAMS is tl.constexpr so the inner loop is fully unrolled.

Kernel layout (bwd_dPhi):
  Grid: (B * N * N,)
  One program per scalar grad_Phi[b, n1, n2], loops over D.

Backward shared intermediates (proj case only, computed once in Python):
  alpha  [B, D]  = v^T x
  phi_v  [B, N]  = Phi @ v
  c      [B, N]  = v - phi_v
  beta   [B, D]  = einsum("bnd,bn->bd", G, c)

Precomputing alpha and beta in Python eliminates O(N²) redundant Phi
loads from bwd_dx and O(N³) redundant x loads from bwd_dPhi.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

###
### Autotune configs
### num_stages enables software pipelining; narrower BLOCK_D options give the
### autotuner room to avoid register spill at large N.
###
###
### Autotune configs
### BLOCK_D must be a power-of-2 ≥ 16 for tl.dot on Ampere (sm_80).
###
_FWD_CONFIGS = [
    triton.Config({"BLOCK_D": 64},  num_warps=1, num_stages=2),
    triton.Config({"BLOCK_D": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_D": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 64},  num_warps=8, num_stages=4),
    ###
    triton.Config({"BLOCK_D": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=4),
    ###
    triton.Config({"BLOCK_D": 256}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=4),
]

_DPHI_CONFIGS = [
    triton.Config({"BLOCK_D": 64},  num_warps=1, num_stages=2),
    triton.Config({"BLOCK_D": 64},  num_warps=2, num_stages=4),
    triton.Config({"BLOCK_D": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 64},  num_warps=4, num_stages=4),
    ###
    triton.Config({"BLOCK_D": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=4),
    ###
    triton.Config({"BLOCK_D": 256}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=4),
]


###
### Forward kernel
###
@triton.autotune(configs=_FWD_CONFIGS, key=["D", "N_STREAMS"], cache_results=True)
@triton.jit
def _stream_mix_fwd(
    Phi_ptr, x_ptr, Y_ptr, out_ptr, v_ptr,
    D,
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
        alpha = tl.zeros([BLOCK_D], dtype=tl.float32)  # v^T x
        phi_v = 0.0   # Phi[n1,:] @ v  (scalar)
        v_n1  = tl.load(v_ptr + b * stride_v_b + n1 * stride_v_n).to(tl.float32)

    ### Inner loop — fully unrolled at compile time (N_STREAMS is constexpr)
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

    tl.store(
        out_ptr + b * stride_o_b + n1 * stride_o_n + d_idx * stride_o_d,
        acc + y_vec,
        mask=d_mask,
    )


###
### Backward kernel: grad_x
###
### no-proj:  grad_x[b, n2, d] = (Phi^T @ G)[b, n2, d]
### proj:     grad_x[b, n2, d] = (Phi^T @ G)[b, n2, d] + v[b, n2] * beta[b, d]
###
### beta[b, d] = Σ_n1( G[b,n1,d] * c[b,n1] )  where c = v - Phi@v
### beta is precomputed in Python and passed as beta_ptr.
### This removes the O(N²) nested loop from the original implementation.
###
@triton.autotune(configs=_FWD_CONFIGS, key=["D", "N_STREAMS"], cache_results=True)
@triton.jit
def _stream_mix_bwd_dx(
    G_ptr, Phi_ptr, v_ptr, beta_ptr, grad_x_ptr,
    D,
    stride_g_b,    stride_g_n,    stride_g_d,
    stride_phi_b,  stride_phi_n1, stride_phi_n2,
    stride_v_b,    stride_v_n,
    stride_beta_b, stride_beta_d,
    stride_gx_b,   stride_gx_n,   stride_gx_d,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_bn = tl.program_id(0)
    pid_d  = tl.program_id(1)
    b  = pid_bn // N_STREAMS
    n2 = pid_bn %  N_STREAMS

    d_off  = pid_d * BLOCK_D
    d_idx  = d_off + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    if USE_PROJ:
        v_n2 = tl.load(v_ptr + b * stride_v_b + n2 * stride_v_n).to(tl.float32)

    # Phi^T @ G: load column n2 of Phi (= row n2 of Phi^T)
    for n1 in tl.static_range(N_STREAMS):
        phi_n1_n2 = tl.load(
            Phi_ptr + b * stride_phi_b + n1 * stride_phi_n1 + n2 * stride_phi_n2
        ).to(tl.float32)
        g_vec = tl.load(
            G_ptr + b * stride_g_b + n1 * stride_g_n + d_idx * stride_g_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)
        acc = acc + phi_n1_n2 * g_vec

    if USE_PROJ:
        # beta precomputed in Python — single load, no nested N loop
        beta_vec = tl.load(
            beta_ptr + b * stride_beta_b + d_idx * stride_beta_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)
        acc = acc + v_n2 * beta_vec

    tl.store(
        grad_x_ptr + b * stride_gx_b + n2 * stride_gx_n + d_idx * stride_gx_d,
        acc,
        mask=d_mask,
    )


###
### Fused reduce-over-N kernel: alpha and beta in one pass
###
### alpha[b, d] = Σ_n v[b,n] * x[b,n,d]        (drives grad_Phi's x_eff)
### beta[b, d]  = Σ_n c[b,n] * G[b,n,d]         (drives grad_x's proj term)
###
### Both contract the tiny N dim, which torch.einsum lowers to a skinny
### [B,1,N]@[B,N,D] bmm with poor SM utilisation. Fusing them into one
### bandwidth-bound kernel reads x and G exactly once each. v and c are loaded
### in fp32 (accumulation is fp32), so — unlike the einsum path, which had to
### cast c down to G's dtype — beta keeps full fp32 precision in c.
###
@triton.autotune(configs=_FWD_CONFIGS, key=["D", "N_STREAMS"], cache_results=True)
@triton.jit
def _alpha_beta_fused(
    x_ptr, G_ptr, v_ptr, c_ptr, alpha_ptr, beta_ptr,
    D,
    stride_x_b,  stride_x_n,  stride_x_d,
    stride_g_b,  stride_g_n,  stride_g_d,
    stride_v_b,  stride_v_n,
    stride_c_b,  stride_c_n,
    stride_a_b,  stride_a_d,
    stride_be_b, stride_be_d,
    N_STREAMS: tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_idx  = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    a_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    b_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for n in tl.static_range(N_STREAMS):
        v_n = tl.load(v_ptr + pid_b * stride_v_b + n * stride_v_n).to(tl.float32)
        c_n = tl.load(c_ptr + pid_b * stride_c_b + n * stride_c_n).to(tl.float32)
        x_vec = tl.load(
            x_ptr + pid_b * stride_x_b + n * stride_x_n + d_idx * stride_x_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)
        g_vec = tl.load(
            G_ptr + pid_b * stride_g_b + n * stride_g_n + d_idx * stride_g_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)
        a_acc = a_acc + v_n * x_vec
        b_acc = b_acc + c_n * g_vec

    tl.store(alpha_ptr + pid_b * stride_a_b + d_idx * stride_a_d, a_acc, mask=d_mask)
    tl.store(beta_ptr  + pid_b * stride_be_b + d_idx * stride_be_d, b_acc, mask=d_mask)


###
### Fused reduce-over-D kernel: grad_Phi, rho and beta_part in one pass
###
### grad_Phi[b,n1,n2] = Σ_d G[b,n1,d]·x_eff[b,n2,d],  x_eff = x - v·alpha  (proj)
### rho[b,n1]         = Σ_d G[b,n1,d]·alpha[b,d]        (grad_v, alpha-term)
### beta_part[b,n2]   = Σ_d beta[b,d]·x[b,n2,d]         (grad_v, c-term)
###
### Grid = (B,). One program per batch element loads G[b] and x[b] exactly once
### (the old per-(b,n1,n2) kernel reloaded each N times) and accumulates all
### N·N + 2N outputs in registers over a running D loop. This folds the standalone
### grad_Phi kernel and the rho / beta_part einsums into a single streaming pass.
###
@triton.autotune(configs=_DPHI_CONFIGS, key=["D", "N_STREAMS"], cache_results=True)
@triton.jit
def _dphi_rho_bp_fused(
    G_ptr, x_ptr, v_ptr, alpha_ptr, beta_ptr,
    grad_Phi_ptr, rho_ptr, bp_ptr,
    D,
    stride_g_b,   stride_g_n,   stride_g_d,
    stride_x_b,   stride_x_n,   stride_x_d,
    stride_v_b,   stride_v_n,
    stride_a_b,   stride_a_d,
    stride_be_b,  stride_be_d,
    stride_gP_b,  stride_gP_n1, stride_gP_n2,
    stride_rho_b, stride_rho_n,
    stride_bp_b,  stride_bp_n,
    N_STREAMS: tl.constexpr,
    N_POW2:    tl.constexpr,
    USE_PROJ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    b = tl.program_id(0)
    # N padded to a power of 2 for tl.arange; padded lanes are masked to 0 on
    # load so they never enter the [:N, :N] result.
    n_idx  = tl.arange(0, N_POW2)
    n_mask = n_idx < N_STREAMS

    gP_acc  = tl.zeros([N_POW2, N_POW2], dtype=tl.float32)
    rho_acc = tl.zeros([N_POW2], dtype=tl.float32)
    bp_acc  = tl.zeros([N_POW2], dtype=tl.float32)

    v = tl.zeros([N_POW2], dtype=tl.float32)
    if USE_PROJ:
        v = tl.load(v_ptr + b * stride_v_b + n_idx * stride_v_n,
                    mask=n_mask, other=0.0).to(tl.float32)

    n_blocks = tl.cdiv(D, BLOCK_D)
    for i in range(n_blocks):
        d_idx  = i * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D
        nd_mask = n_mask[:, None] & d_mask[None, :]

        g = tl.load(
            G_ptr + b * stride_g_b + n_idx[:, None] * stride_g_n + d_idx[None, :] * stride_g_d,
            mask=nd_mask, other=0.0,
        ).to(tl.float32)   # [N_POW2, BLOCK_D]
        xt = tl.load(
            x_ptr + b * stride_x_b + n_idx[:, None] * stride_x_n + d_idx[None, :] * stride_x_d,
            mask=nd_mask, other=0.0,
        ).to(tl.float32)   # [N_POW2, BLOCK_D]

        if USE_PROJ:
            alpha = tl.load(alpha_ptr + b * stride_a_b + d_idx * stride_a_d,
                            mask=d_mask, other=0.0).to(tl.float32)   # [BLOCK_D]
            beta  = tl.load(beta_ptr + b * stride_be_b + d_idx * stride_be_d,
                            mask=d_mask, other=0.0).to(tl.float32)   # [BLOCK_D]
            xeff = xt - v[:, None] * alpha[None, :]                  # [N_POW2, BLOCK_D]
            rho_acc = rho_acc + tl.sum(g * alpha[None, :], axis=1)   # [N_POW2]
            bp_acc  = bp_acc + tl.sum(beta[None, :] * xt, axis=1)    # [N_POW2]
        else:
            xeff = xt

        # grad_Phi[n1, n2] = Σ_d g[n1, d] · xeff[n2, d]
        gP_acc = gP_acc + tl.sum(g[:, None, :] * xeff[None, :, :], axis=2)  # [N_POW2, N_POW2]

    nn_mask = n_mask[:, None] & n_mask[None, :]
    tl.store(
        grad_Phi_ptr + b * stride_gP_b + n_idx[:, None] * stride_gP_n1 + n_idx[None, :] * stride_gP_n2,
        gP_acc, mask=nn_mask,
    )
    if USE_PROJ:
        tl.store(rho_ptr + b * stride_rho_b + n_idx * stride_rho_n, rho_acc, mask=n_mask)
        tl.store(bp_ptr  + b * stride_bp_b  + n_idx * stride_bp_n,  bp_acc,  mask=n_mask)


###
### Python helpers
###
def _make_v_arg(v: torch.Tensor | None, B: int, N: int, device, dtype):
    if v is not None:
        return v.contiguous()
    return torch.zeros(B, N, dtype=dtype, device=device)


def _make_bd_arg(t: torch.Tensor | None, B: int, D: int, device):
    """[B, D] dummy when t is None; strides are never dereferenced."""
    if t is not None:
        return t.contiguous()
    return torch.zeros(B, D, dtype=torch.float32, device=device)


def _launch_fwd(Phi, x, Y, v, out):
    B, N, D = x.shape
    use_proj = v is not None
    v_arg = _make_v_arg(v, B, N, x.device, x.dtype)
    grid = lambda meta: (B * N, triton.cdiv(D, meta["BLOCK_D"]))
    _stream_mix_fwd[grid](
        Phi, x, Y, out, v_arg,
        D,
        *Phi.stride(), *x.stride(), *Y.stride(), *out.stride(), *v_arg.stride(),
        N_STREAMS=N, USE_PROJ=use_proj,
    )


def _launch_bwd_dx(G, Phi, v, beta, grad_x, N):
    B, _, D = G.shape
    use_proj = v is not None
    v_arg    = _make_v_arg(v, B, N, G.device, G.dtype)
    beta_arg = _make_bd_arg(beta, B, D, G.device)
    grid = lambda meta: (B * N, triton.cdiv(D, meta["BLOCK_D"]))
    _stream_mix_bwd_dx[grid](
        G, Phi, v_arg, beta_arg, grad_x,
        D,
        *G.stride(), *Phi.stride(), *v_arg.stride(), *beta_arg.stride(), *grad_x.stride(),
        N_STREAMS=N, USE_PROJ=use_proj,
    )


def _launch_alpha_beta(x, G, v, c):
    """Fused reduce-over-N: returns (alpha, beta), each [B, D] fp32."""
    B, N, D = x.shape
    alpha = torch.empty(B, D, dtype=torch.float32, device=x.device)
    beta  = torch.empty(B, D, dtype=torch.float32, device=x.device)
    grid  = lambda meta: (B, triton.cdiv(D, meta["BLOCK_D"]))
    _alpha_beta_fused[grid](
        x, G, v, c, alpha, beta,
        D,
        *x.stride(), *G.stride(), *v.stride(), *c.stride(), *alpha.stride(), *beta.stride(),
        N_STREAMS=N,
    )
    return alpha, beta


def _launch_dphi_rho_bp(G, x, v, alpha, beta, N):
    """Fused reduce-over-D. Returns (grad_Phi [B,N,N], rho [B,N], beta_part [B,N]).

    rho and beta_part are None in the no-proj case.
    """
    B, _, D = G.shape
    use_proj = v is not None
    grad_Phi = torch.empty(B, N, N, dtype=torch.float32, device=x.device)
    rho = torch.empty(B, N, dtype=torch.float32, device=x.device) if use_proj else None
    bp  = torch.empty(B, N, dtype=torch.float32, device=x.device) if use_proj else None

    v_arg   = _make_v_arg(v, B, N, G.device, G.dtype)
    a_arg   = _make_bd_arg(alpha, B, D, G.device)
    be_arg  = _make_bd_arg(beta, B, D, G.device)
    rho_arg = rho if use_proj else torch.empty(B, N, dtype=torch.float32, device=x.device)
    bp_arg  = bp  if use_proj else torch.empty(B, N, dtype=torch.float32, device=x.device)

    grid = (B,)
    _dphi_rho_bp_fused[grid](
        G, x, v_arg, a_arg, be_arg, grad_Phi, rho_arg, bp_arg,
        D,
        *G.stride(), *x.stride(), *v_arg.stride(), *a_arg.stride(), *be_arg.stride(),
        *grad_Phi.stride(), *rho_arg.stride(), *bp_arg.stride(),
        N_STREAMS=N, N_POW2=triton.next_power_of_2(N), USE_PROJ=use_proj,
    )
    return grad_Phi, rho, bp


###
### Autograd Function
###
class _StreamMixFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Phi, x, Y, v):
        Phi_c = Phi.contiguous()
        x_c   = x.contiguous()
        Y_c   = Y.contiguous()
        v_c   = v.contiguous() if v is not None else None

        out = torch.empty_like(x_c)
        _launch_fwd(Phi_c, x_c, Y_c, v_c, out)

        ctx.save_for_backward(Phi_c, x_c, v_c)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Phi, x, v = ctx.saved_tensors
        B, N, D   = x.shape
        use_proj  = v is not None

        # G keeps grad_out's native dtype: the Triton kernels upcast on load
        # (.to(tl.float32)) and grad_out already arrives in the input dtype, so an
        # explicit .float() here recovers no precision — it only doubles the size
        # of the [B, N, D] tensor that every backward kernel/einsum then streams.
        G = grad_out.contiguous()

        ### Shared intermediates (proj only)
        # alpha[b,d] = Σ_n v[b,n]·x[b,n,d]  and  beta[b,d] = Σ_n c[b,n]·G[b,n,d]
        # both contract the tiny N dim. A single fused kernel reads x and G once
        # each (vs two skinny einsums), keeps v/c in fp32 (full-precision beta),
        # and eliminates the O(N²)/O(N³) per-program loops the kernels once used.
        alpha = beta = phi_v = c = None
        if use_proj:
            phi_v = torch.bmm(Phi.float(), v.float().unsqueeze(-1)).squeeze(-1)  # [B, N]
            c     = v.float() - phi_v                                            # [B, N]
            alpha, beta = _launch_alpha_beta(x, G, v, c)                         # [B, D], [B, D]

        ### grad_x (Triton)
        grad_x = torch.empty_like(x)
        _launch_bwd_dx(G, Phi, v, beta, grad_x, N)

        ### grad_Phi + grad_v reductions (fused Triton, one pass over G and x)
        # A single reduce-over-D kernel produces grad_Phi[B,N,N] plus the two
        # grad_v reductions:
        #   rho[b,n]       = Σ_d G[b,n,d] · alpha[b,d]     [d/dv of the c-term]
        #   beta_part[b,n] = Σ_d beta[b,d] · x[b,n,d]      [d/dv of the alpha-term]
        # replacing the standalone grad_Phi kernel (which reloaded G/x N× per b)
        # and the two rho/beta_part einsums.
        grad_Phi, rho, beta_part = _launch_dphi_rho_bp(G, x, v, alpha, beta, N)

        ### grad_Y = grad_out (identity)
        grad_Y = grad_out

        ### grad_v (PyTorch): assemble from the fused rho / beta_part
        #   grad_v = (I - Phi^T) @ rho  +  beta_part
        grad_v = None
        if use_proj and ctx.needs_input_grad[3]:
            rho_part = rho - torch.bmm(Phi.float().mT, rho.unsqueeze(-1)).squeeze(-1)  # [B, N]
            grad_v   = (rho_part + beta_part).to(v.dtype)

        return (
            grad_Phi.to(Phi.dtype) if ctx.needs_input_grad[0] else None,
            grad_x if ctx.needs_input_grad[1] else None,
            grad_Y if ctx.needs_input_grad[2] else None,
            grad_v,
        )


###
### Public API
###
def stream_mix_add_small_nb(
    Phi: torch.Tensor,
    x: torch.Tensor,
    Y: torch.Tensor,
    v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused  out = Phi @ x + Y  (or projected variant) — small-NB kernel.

    Args:
        Phi: [B, N, N] transition matrix.
        x:   [B, N, D] stream state.
        Y:   [B, N, D] source term.
        v:   [B, N]    unit-norm projection direction, or None.

    Returns:
        out: [B, N, D]
    """
    return _StreamMixFn.apply(Phi, x, Y, v)
