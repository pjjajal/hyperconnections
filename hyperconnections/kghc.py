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