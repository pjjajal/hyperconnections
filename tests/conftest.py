import pytest
import torch
import torch.nn as nn

from hyperconnections.ghc import GeneralizedHyperConnections
from hyperconnections.ops import HAS_TRITON


# ---------------------------------------------------------------------------
# Device / custom-kernel dispatch
# ---------------------------------------------------------------------------
# Triton kernels require CUDA tensors. The model classes (CGHC / CGHCF) bind
# their kernel dispatch at construction time from ``use_triton and HAS_TRITON``
# *without* consulting the device, so a CPU run on a box where triton happens to
# be importable would route into CUDA-only kernels and raise at call time.
# These helpers make the intended device the single source of truth: tests pick
# a device, and the custom-kernel flag is toggled to match it.

# Devices the suite should exercise: always CPU, plus CUDA when present.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def should_use_triton(device) -> bool:
    """Whether the Triton custom kernels are usable on ``device``.

    True only for CUDA devices when triton is importable; CPU always returns
    False so callers fall back to the eager (pure-PyTorch) path.
    """
    return torch.device(device).type == "cuda" and HAS_TRITON


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    """Parametrized device fixture: yields "cpu" (always) and "cuda" (if available)."""
    return request.param


class IdentityModule(nn.Module):
    """Passes input through unchanged. Accepts **kwargs for compatibility."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


class ZeroModule(nn.Module):
    """Always outputs zeros. Useful for isolating the stream_mixing path."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros_like(x)


def make_ghc(
    n: int,
    m: int,
    embed_dim: int,
    module: nn.Module | None = None,
    device: str = "cpu",
    **kwargs,
) -> GeneralizedHyperConnections:
    input_dim = (n * embed_dim) // m
    if module is None:
        module = IdentityModule()
    # GHC has no Triton path, so only the device placement matters here.
    return GeneralizedHyperConnections(
        n=n, m=m, input_dim=input_dim, embed_dim=embed_dim, module=module, **kwargs
    ).to(device)
