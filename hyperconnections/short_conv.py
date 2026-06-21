from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseShortConv1d(nn.Module):
    """Depthwise short convolution over the sequence dimension.

    Mixes a small local window per channel and preserves the sequence length.
    Operates on ``[B, T, C]`` (a single token/sequence axis is assumed, so it is
    meant for sequence-like data, not arbitrary leading dims).

    ``causal=True`` left-pads by ``kernel_size - 1`` so position ``t`` only sees
    positions ``<= t`` (autoregressive LM use). ``causal=False`` pads
    symmetrically — ``(kernel_size - 1) // 2`` left, ``kernel_size // 2`` right —
    so the window is centered, which is what non-autoregressive data (e.g. vision
    tokens) wants.

    Bias-free by design (matches the DDL / DeltaNet short-conv convention). The
    depthwise weight is held directly as ``[dim, kernel_size]`` and initialised
    ``uniform(-1/sqrt(k), 1/sqrt(k))`` — the same scale ``nn.Conv1d`` uses for a
    depthwise kernel, made explicit here.
    """

    def __init__(self, dim: int, kernel_size: int = 4, *, causal: bool = True) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        self.dim = dim
        self.kernel_size = kernel_size
        self.causal = causal
        if causal:
            self.padding = (kernel_size - 1, 0)
        else:
            self.padding = ((kernel_size - 1) // 2, kernel_size // 2)
        self.weight = nn.Parameter(torch.empty(dim, dim, kernel_size))
        self.init_weights()

    def init_weights(self) -> None:
        bound = 1.0 / math.sqrt(self.kernel_size)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, C, T]
        x = F.pad(x, self.padding)

        x = F.conv1d(x, self.weight)
        return x.transpose(1, 2)  # [B, T, C]
