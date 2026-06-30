import math

import torch
import torch.nn as nn
from einops import einsum
from timm.layers import trunc_normal_

from hyperconnections.short_conv import DepthwiseShortConv1d


class IdentityHyperConnections(nn.Module):
    """Simplified hyperconnections using identity matrix for stream mixing.

    This variant strips away all learned stream mixing dynamics and uses
    a fixed identity matrix, focusing purely on read/write mechanics.
    """

    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        bias: bool = False,
        elementwise_affine: bool = False,
        shortconv_kernel_size: int = 0,
        shortconv_causal: bool = True,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim

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
        self.write_out = nn.Parameter(torch.empty(n, m))
        self.alpha_write_out = nn.Parameter(torch.empty(1))

        self.proj_read_in = nn.Linear(input_dim, n * m, bias=bias)
        self.proj_write_out = nn.Linear(input_dim, n * m, bias=bias)

        self.norm = nn.RMSNorm(input_dim, elementwise_affine=elementwise_affine)
        self.module = module
        # Optional over-width short conv on the read/source path (see forward).
        self.short_conv = (
            DepthwiseShortConv1d(input_dim, shortconv_kernel_size, causal=shortconv_causal)
            if shortconv_kernel_size > 0
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
        h_write_out = self.proj_write_out(x_norm).reshape(B, self.n, self.m)

        read_in = torch.sigmoid(
            self.alpha_read_in * h_read_in + self.read_in
        ).transpose(1, 2)  # [B, m, n]
        write_out = 2 * torch.sigmoid(
            self.alpha_write_out * h_write_out + self.write_out
        )  # [B, n, m]

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
        ### Optional over-width short conv on the read/source path only: it feeds the
        ### read-in and the x_norm-derived read/write weights, while the carried stream
        ### `x` that gets added back stays un-convolved.
        if self.short_conv is not None:
            src = self.short_conv(x.reshape(*leading, self.input_dim)).reshape(B, self.n, self.block_size)
        else:
            src = x
        x_norm = self.norm(src.view(B, -1))  ### [B*, input_dim]

        write_out, read_in = self.compute_read_write_weights(x_norm)

        ### Source term Y = H^post F(H^pre X)  (read → compute → write)
        ### Read in from over-width space to backbone width
        x_read = einsum(read_in, src, "b m n, b n d -> b m d")  ### [B*, m, block_size]

        ### Process through the backbone module
        out = self.module(x_read.reshape(*leading, self.embed_dim), **kwargs)

        ### Write out from backbone width back to the over-width space
        out = out.reshape(B, self.m, self.block_size)  ### [B*, m, block_size]
        Y = einsum(write_out, out, "b n m, b m d -> b n d")  ### [B*, n, block_size]

        ### Identity stream mixing: X_new = I @ X + Y = X + Y
        # Simply add the residual (identity matrix multiplication is a no-op)
        x_out = x + Y  ### [B*, n, block_size]

        return x_out.unflatten(0, leading).flatten(-2)
