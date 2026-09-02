import math

import torch
import torch.nn as nn
from einops import einsum
from timm.layers import trunc_normal_

from hyperconnections.temporal_writeback import TemporalWriteback


class IdentityHyperConnections(nn.Module):
    """Simplified hyperconnections using identity matrix for stream mixing.

    This variant strips away all learned stream mixing dynamics and uses
    a fixed identity matrix, focusing purely on read/write mechanics.
    """

    ### Static anchor matrices to exclude from weight decay (saturated-logit
    ### inits — see the weight-decay note in cghc.py). Collected by
    ### ContinuousGenHyperConnections.split_decay_param_groups via hasattr.
    NO_DECAY_PARAM_NAMES = frozenset({"read_in", "write_out"})

    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        bias: bool = False,
        elementwise_affine: bool = False,
        writeback_kernel_sizes: tuple[int, ...] = (),
        writeback_orthogonalize: bool = True,
    ) -> None:
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.writeback_components = len(writeback_kernel_sizes) + 1

        assert embed_dim % m == 0, (
            f"embed_dim ({embed_dim}) must be divisible by m ({m})"
        )
        assert input_dim == int((n / m) * embed_dim), (
            f"input_dim must be (n/m)*embed_dim, got {input_dim} vs {int((n / m) * embed_dim)}"
        )

        self.block_size = embed_dim // m

        # Read/write parameters following mHC convention
        self.read_in = nn.Parameter(torch.empty(n, m))
        self.alpha_read_in = nn.Parameter(torch.empty(1))
        write_out_shape = (
            (n, m)
            if self.writeback_components == 1
            else (n, self.writeback_components, m)
        )
        self.write_out = nn.Parameter(torch.empty(write_out_shape))
        self.alpha_write_out = nn.Parameter(torch.empty(1))

        self.proj_read_in = nn.Linear(input_dim, n * m, bias=bias)
        self.proj_write_out = nn.Linear(
            input_dim,
            n * m * self.writeback_components,
            bias=bias,
        )

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

        # Fixed identity matrix for stream mixing (no learning)
        self.register_buffer("stream_mixing", torch.eye(n))

        self.init_weights()

    def init_weights(self):
        # read_in: σ(bias) = 1/n  →  bias = log(1/(n-1))
        logit_1_over_n = math.log(1.0 / (self.n - 1)) if self.n > 1 else 10.0
        nn.init.constant_(self.read_in, logit_1_over_n)
        with torch.no_grad():
            self.read_in.add_(
                torch.randn_like(self.read_in) * 0.01
            )  # small noise for asymmetry breaking

        # write_out: 2·σ(0) = 1
        trunc_normal_(self.write_out, std=0.01)

        # Alpha gating: 0.01 so dynamic component starts negligible
        nn.init.constant_(self.alpha_read_in, 0.01)
        nn.init.constant_(self.alpha_write_out, 0.01)

        # Projections: small random init for weights, zero bias
        for proj in (self.proj_read_in, self.proj_write_out):
            trunc_normal_(proj.weight, std=0.01)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # RMSNorm weights: must be ones for proper normalization
        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)

    def compute_read_write_weights(self, x_norm: torch.Tensor):
        """Compute dynamic read/write weights from the current stream state.

        Args:
            x_norm: Normalized input of shape [B, input_dim]
        """
        B = x_norm.shape[0]

        h_read_in = self.proj_read_in(x_norm).reshape(B, self.n, self.m)
        h_write_out = self.proj_write_out(x_norm).reshape(
            B,
            self.n,
            self.writeback_components,
            self.m,
        )

        read_in = torch.sigmoid(
            self.alpha_read_in * h_read_in + self.read_in
        ).transpose(1, 2)  # [B, m, n]
        write_out_bias = self.write_out
        if self.writeback_components == 1:
            write_out_bias = write_out_bias.unsqueeze(1)
        write_out = 2 * torch.sigmoid(
            self.alpha_write_out * h_write_out + write_out_bias
        )
        if self.writeback_components == 1:
            write_out = write_out.squeeze(2)

        return write_out, read_in

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, *, input_dim]  (any number of leading dims, last dim = n * block_size)
        Returns:
            [B, *, input_dim]
        """
        ### x: [B, *, input_dim]
        leading = x.shape[:-1]
        x = x.reshape(-1, self.n, self.block_size)  ### [B*, n, block_size]
        B = x.shape[0]
        x_norm = self.norm(x.view(B, -1))  ### [B*, input_dim]

        write_out, read_in = self.compute_read_write_weights(x_norm)

        ### Source term Y = H^post F(H^pre X)  (read → compute → write)
        ### Read in from over-width space to backbone width
        x_read = einsum(read_in, x, "b m n, b n d -> b m d")

        ### Process through the backbone module
        module_output = self.module(
            x_read.reshape(*leading, self.embed_dim),
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

        ### Identity stream mixing: X_new = I @ X + Y = X + Y
        # Simply add the residual (identity matrix multiplication is a no-op)
        x_out = x + source_update  ### [B*, n, block_size]

        return x_out.unflatten(0, leading).flatten(-2)
