import torch
import torch.nn as nn

from hyperconnections.ghc import GeneralizedHyperConnections


class IdentityModule(nn.Module):
    """Passes input through unchanged. Accepts **kwargs for compatibility."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


class ZeroModule(nn.Module):
    """Always outputs zeros. Useful for isolating the stream_mixing path."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(x)


def make_ghc(n: int, m: int, embed_dim: int, module: nn.Module | None = None, **kwargs) -> GeneralizedHyperConnections:
    input_dim = (n * embed_dim) // m
    if module is None:
        module = IdentityModule()
    return GeneralizedHyperConnections(
        n=n, m=m, input_dim=input_dim, embed_dim=embed_dim, module=module, **kwargs
    )
