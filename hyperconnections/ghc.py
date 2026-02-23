import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


# Input Dimension: d_in = (n / m) * embed_dim
# n / m is the expansion ratio of the embedding dimension.
# when m = 1, we get the regular hyperconnections, where the input dimension is n * embed_dim.
# when m = n, we get the regular connections, where the input dimension is embed_dim.
# when n > m > 1, we get a generalized version of hyperconnections, where the input dimension is (n / m) * embed_dim.
class GeneralizedHyperConnections(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        bias: bool = False,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        assert input_dim == int(
            (n / m) * embed_dim
        ), f"Input dimension must be (n / m) * embed_dim, but got {input_dim} and {(n / m) * embed_dim}"

        self.block_size = embed_dim // m  # the block size = d_in / n = embed_dim / m
        self.scaling_factor = math.sqrt(self.block_size)

        static_b = torch.zeros((m, n))  # [m, n]
        for j in range(static_b.shape[1]):
            static_b[j % m, j] = 1.0
        self.static_b = nn.Parameter(static_b.T.contiguous())  # [n, m]
        self.dynamic_scaling_b = nn.Parameter(torch.ones_like(self.static_b))  # [n, m]
        self.dynamic_scaling_weight_b = nn.Linear(
            self.block_size, m, bias=bias
        )  # [block_size, m]

        static_a = torch.zeros((n, n + m))  # [n, n + m]
        r = n - m
        static_a[:m, :m] = torch.eye(m)
        static_a[:m, m : m + m] = torch.eye(m)
        if r > 0:
            static_a[m:, 2 * m : 2 * m + r] = torch.eye(r)
        self.static_a = nn.Parameter(static_a.contiguous())  # [n, n + m]
        self.dynamic_scaling_a = nn.Parameter(
            torch.ones_like(self.static_a)
        )  # [n, n + m]
        self.dynamic_scaling_weight_a = nn.Linear(
            self.block_size, n + m, bias=bias
        )  # [block_size, n + m]

        self.norm = nn.RMSNorm(self.block_size, elementwise_affine=elementwise_affine)

        self.module = module

    def init_weights(self):
        # Initialize the dynamic scaling weights to be small, so that the initial behavior of the model is close to the static connections.
        nn.init.zeros_(self.dynamic_scaling_weight_b.weight)
        if self.dynamic_scaling_weight_b.bias is not None:
            nn.init.zeros_(self.dynamic_scaling_weight_b.bias)
        nn.init.zeros_(self.dynamic_scaling_weight_a.weight)
        if self.dynamic_scaling_weight_a.bias is not None:
            nn.init.zeros_(self.dynamic_scaling_weight_a.bias)

    def compute_mixing_weights(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n, block_size]

        x = self.norm(x)  # [B, n, block_size]

        # Computes the dynamic B matrix, this is referred to as the write_out_matrix. Because it writes out to the high-dimensional hyperconnection space.
        write_out_matrix = (
            self.dynamic_scaling_b
            * F.tanh(self.dynamic_scaling_weight_b(x) / self.scaling_factor)
            + self.static_b
        )  # [B, n, m]

        # Computes the dynamic A matrix, i.e., the read_in_matrix (\mathring{A}), and stream_mixing_matrix (\hat{A}). Because it reads in from the high-dimensional hyperconnection space.
        dynamic_A = (
            self.dynamic_scaling_a
            * F.tanh(self.dynamic_scaling_weight_a(x) / self.scaling_factor)
            + self.static_a
        )  # [B, n, n + m]
        read_in_matrix = dynamic_A[:, :, : self.m].transpose(1, 2)  # [B, m, n]
        stream_mixing_matrix = dynamic_A[:, :, self.m :]  # [B, n, n]

        return write_out_matrix, read_in_matrix, stream_mixing_matrix

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # X: [B, input_dim]
        B, _ = x.shape
        # Reshape the input to [B, n, block_size]
        x = x.reshape(B, self.n, self.block_size)  # [B, n, block_size]

        # We compute the dynamic mixing weights.
        write_out_matrix, read_in_matrix, stream_mixing_matrix = (
            self.compute_mixing_weights(x)
        )  # [B, n, m], [B, m, n], [B, n, n]

        # Read in from the hyperconnection space
        x_read_in = einsum(
            read_in_matrix, x, "b m n, b n d -> b m d"
        )  # [B, m, block_size]

        #  process the read-in information through the module.
        out = self.module(x_read_in.reshape(B, -1), **kwargs)

        # write out to the hyperconnection space
        out = out.reshape(B, self.m, self.block_size)  # [B, m, block_size]
        out = einsum(
            write_out_matrix, out, "b n m, b m d -> b n d"
        )  # [B, n, block_size]

        # mix the original hyperconnection space.
        x = einsum(
            stream_mixing_matrix, x, "b n1 n2, b n2 d -> b n1 d"
        )  # [B, n, block_size]
        out = out + x  # [B, n, block_size]
        return out.reshape(B, -1)
