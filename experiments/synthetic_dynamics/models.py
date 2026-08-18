"""Minimal shared-layer rollout for stream dynamics."""

import torch
import torch.nn as nn


class ZeroModule(nn.Module):
    def forward(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(state)


class StreamDynamics(nn.Module):
    def __init__(self, layer: nn.Module, n: int, d: int, depth: int):
        super().__init__()
        self.layer = layer
        self.n = n
        self.d = d
        self.depth = depth

    def forward(
        self, state: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        state = state.flatten(1)
        for step in range(self.depth):
            state = self.layer(state)
            if noise is not None:
                state = state.reshape(batch_size, self.n, self.d)
                state = state + noise[:, step]
                state = state.flatten(1)
        return state.reshape(batch_size, self.n, self.d)
