import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from timm.layers import trunc_normal_

from hyperconnections.temporal_writeback import TemporalWriteback


# Input Dimension: d_in = (n / m) * embed_dim
# n / m is the expansion ratio of the embedding dimension.
# when m = 1, we get regular hyperconnections and input_dim = n * embed_dim.
# when m = n, we get the regular connections, where the input dimension is embed_dim.
# when n > m > 1, we get generalized hyperconnections.

# Sinkhorn-Knopp bias init: exp(1 · I_n) is strongly diagonally dominant so
# Sinkhorn(exp(1 · I_n)) ≈ I_n, matching the identity-mapping starting point.
_SINKHORN_BIAS_INIT = 5.0


class ManifoldHyperConnections(nn.Module):
    ### Static anchor matrices to exclude from weight decay (saturated-logit /
    ### identity inits — see the weight-decay note in cghc.py). Collected by
    ### ContinuousGenHyperConnections.split_decay_param_groups via hasattr.
    NO_DECAY_PARAM_NAMES = frozenset({"read_in", "write_out", "stream_mixing"})

    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        bias: bool = False,
        elementwise_affine: bool = False,
        sinkhorn_iters: int = 20,
        sinkhorn_bias_init: float = _SINKHORN_BIAS_INIT,
        writeback_kernel_sizes: tuple[int, ...] = (),
        writeback_orthogonalize: bool = True,
    ) -> None:
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_bias_init = sinkhorn_bias_init
        self.writeback_components = len(writeback_kernel_sizes) + 1

        assert embed_dim % m == 0, (
            f"embed_dim ({embed_dim}) must be divisible by m ({m})"
        )
        assert input_dim == int(
            (n / m) * embed_dim
        ), (
            "Input dimension must be (n / m) * embed_dim, "
            f"but got {input_dim} and {(n / m) * embed_dim}"
        )

        self.block_size = embed_dim // m  # block size = embed_dim / m = input_dim / n

        # read_in (H^pre): [n, m]
        self.read_in = nn.Parameter(torch.empty(n, m))
        self.alpha_read_in = nn.Parameter(torch.empty(1))

        # write_out (H^post): [n, m] or [n, writeback_components, m]
        write_out_shape = (
            (n, m)
            if self.writeback_components == 1
            else (n, self.writeback_components, m)
        )
        self.write_out = nn.Parameter(torch.empty(write_out_shape))
        self.alpha_write_out = nn.Parameter(torch.empty(1))

        # stream_mixing (H^res): [n, n]
        self.stream_mixing = nn.Parameter(torch.empty(n, n))
        self.alpha_stream_mixing = nn.Parameter(torch.empty(1))

        self.proj_read_in = nn.Linear(input_dim, n * m, bias=bias)
        self.proj_write_out = nn.Linear(
            input_dim,
            n * m * self.writeback_components,
            bias=bias,
        )
        self.proj_stream_mixing = nn.Linear(input_dim, n * n, bias=bias)

        self.norm = nn.RMSNorm(input_dim, elementwise_affine=elementwise_affine)

        self.module = module
        self.writeback = (
            TemporalWriteback(
                embed_dim,
                writeback_kernel_sizes,
                orthogonalize=writeback_orthogonalize,
            )
            if writeback_kernel_sizes
            else None
        )
        self.init_weights()

    def init_weights(self):
        # read_in: initialise so σ(read_in) = 1/n (uniform read across all streams).
        # σ(b) = 1/n  →  b = logit(1/n) = log(1 / (n-1))  for n > 1.
        logit_1_over_n = math.log(1.0 / (self.n - 1)) if self.n > 1 else 10.0
        nn.init.constant_(self.read_in, logit_1_over_n)

        # write_out: initialise so 2·σ(write_out) = 1.
        # 2·σ(0) = 1  →  write_out = 0.
        nn.init.zeros_(self.write_out)

        # stream_mixing: initialise so Sinkhorn(exp(stream_mixing)) ≈ I_n.
        # A large positive diagonal makes exp(stream_mixing) strongly diagonally
        # dominant, so the doubly stochastic projection starts near identity.
        self.stream_mixing.data.copy_(self.sinkhorn_bias_init * torch.eye(self.n))

        # Alpha gating factors: 0.01 per the mHC paper (Table 5).
        nn.init.constant_(self.alpha_read_in, 0.01)
        nn.init.constant_(self.alpha_write_out, 0.01)
        nn.init.constant_(self.alpha_stream_mixing, 0.01)

        # Projection weights: zero so the dynamic component starts negligible and
        # the initial behaviour is entirely determined by the static biases above.
        for proj in (self.proj_read_in, self.proj_write_out, self.proj_stream_mixing):
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _sinkhorn_knopp(self, x: torch.Tensor) -> torch.Tensor:
        x = x.exp()
        for _ in range(self.sinkhorn_iters):
            x = x / x.sum(dim=-1, keepdim=True)
            x = x / x.sum(dim=-2, keepdim=True)
        return x

    def compute_mixing_weights(self, x: torch.Tensor):
        B = x.shape[0]
        x_flat = x.view(B, -1)   # [B, input_dim] — flatten all streams
        # PERF EXPERIMENT (2026-07-15): was `self.norm(x_flat).float()`
        # ("float32 for linear layers under torch.compile") — under autocast the
        # Linears recast their input to bf16 anyway, so the .float() only
        # materialized the wide [B, input_dim] activation in fp32 (same waste as
        # cghc's old generator path). The numerics-sensitive Sinkhorn below now
        # upcasts its small [B, n, n] input instead. TODO: validate numerics
        # long-run, then make final.
        x_norm = self.norm(x_flat)

        h_read_in = self.proj_read_in(x_norm).reshape(B, self.n, self.m)         # [B, n, m]
        h_write_out = self.proj_write_out(x_norm).reshape(
            B,
            self.n,
            self.writeback_components,
            self.m,
        )
        h_stream_mixing = self.proj_stream_mixing(x_norm).reshape(B, self.n, self.n) # [B, n, n]

        # Scale by learnable alpha and add static bias
        h_read_in = self.alpha_read_in * h_read_in + self.read_in
        write_out_bias = self.write_out
        if self.writeback_components == 1:
            write_out_bias = write_out_bias.unsqueeze(1)
        h_write_out = self.alpha_write_out * h_write_out + write_out_bias
        h_stream_mixing = (
            self.alpha_stream_mixing * h_stream_mixing + self.stream_mixing
        )

        # Apply manifold constraints
        read_in = F.sigmoid(h_read_in).transpose(1, 2)
        write_out = 2 * F.sigmoid(h_write_out)
        if self.writeback_components == 1:
            write_out = write_out.squeeze(2)
        stream_mixing = self._sinkhorn_knopp(h_stream_mixing.float())

        return write_out, read_in, stream_mixing

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, *, input_dim]
        leading = x.shape[:-1]
        x = x.reshape(-1, self.n, self.block_size)  # [B*, n, block_size]
        B = x.shape[0]

        write_out, read_in, stream_mixing = self.compute_mixing_weights(x)
        write_out = write_out.to(x.dtype)
        read_in = read_in.to(x.dtype)
        stream_mixing = stream_mixing.to(x.dtype)
        # [B*, n, m], [B*, m, n], [B*, n, n]

        # Read in from the over-width space to backbone width
        x_read_in = einsum(read_in, x, "b m n, b n d -> b m d")

        # Process through the backbone module
        module_output = self.module(
            x_read_in.reshape(*leading, self.embed_dim),
            **kwargs,
        )

        if self.writeback is None:
            write_out = write_out.unsqueeze(2)
            writeback_components = module_output.reshape(
                B,
                1,
                self.m,
                self.block_size,
            )
        else:
            writeback_components = self.writeback(module_output).reshape(
                B,
                self.writeback_components,
                self.m,
                self.block_size,
            )

        source_update = einsum(
            write_out,
            writeback_components,
            "b n k m, b k m d -> b n d",
        )

        # Mix within the over-width space and add residual
        x = einsum(stream_mixing, x, "b n1 n2, b n2 d -> b n1 d")
        out = source_update + x
        return out.unflatten(0, leading).flatten(-2)
