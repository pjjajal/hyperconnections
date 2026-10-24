"""
Tests for device-driven custom-kernel (Triton) dispatch.

Triton kernels require CUDA tensors. A CPU model must fall back to the eager
PyTorch path while a CUDA model uses Triton (when importable). These tests
pin that contract:

  * ``should_use_triton`` resolves the flag from the device.
  * CPU models route to the eager path and run on CPU tensors.
  * CUDA models route to Triton (skipped when no GPU is present).
  * Forcing Triton on a CPU model is the failure the toggle exists to avoid.

Run with:
    pytest tests/test_kernel_device_dispatch.py -v
"""

import pytest
import torch
import torch._dynamo

from hyperconnections.ops import HAS_TRITON
from tests.conftest import DEVICES, should_use_triton
from tests.test_cghc import make_cghc
from tests.test_cghcf_smoke import make_cghcf


@pytest.fixture(autouse=True)
def disable_compile():
    # The forced-exp path uses torch.compiler.disable; keep dynamo off so the
    # eager fallback is exercised directly and assertions stay deterministic.
    torch._dynamo.config.disable = True
    yield
    torch._dynamo.config.disable = False


N, M, EMBED_DIM = 4, 2, 8


def _input(model, device, batch=3):
    return torch.randn(batch, model.input_dim, device=device)


# ---------------------------------------------------------------------------
# The resolver itself
# ---------------------------------------------------------------------------

class TestShouldUseTriton:
    def test_cpu_never_uses_triton(self):
        assert should_use_triton("cpu") is False
        assert should_use_triton(torch.device("cpu")) is False

    def test_cuda_follows_triton_availability(self):
        # Device intent is CUDA → flag tracks whether triton is importable.
        assert should_use_triton("cuda") == HAS_TRITON
        assert should_use_triton(torch.device("cuda")) == HAS_TRITON

    def test_devices_list_includes_cpu(self):
        assert "cpu" in DEVICES
        assert ("cuda" in DEVICES) == torch.cuda.is_available()


# ---------------------------------------------------------------------------
# CGHC: dispatch matches the device, and the model runs on it
# ---------------------------------------------------------------------------

class TestCGHCDispatch:
    @pytest.mark.parametrize("device", DEVICES)
    def test_dispatch_matches_device(self, device):
        model = make_cghc(N, M, EMBED_DIM, device=device)
        assert model._use_triton == should_use_triton(device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_forward_backward_runs_on_device(self, device):
        model = make_cghc(N, M, EMBED_DIM, device=device)
        x = _input(model, device).requires_grad_(True)
        out = model(x)
        assert out.shape == x.shape
        assert out.device.type == torch.device(device).type
        assert out.isfinite().all()
        out.sum().backward()
        assert x.grad is not None and x.grad.isfinite().all()

    def test_cpu_model_is_eager_regardless_of_triton_install(self):
        """The whole point: a CPU model never picks Triton, even where it's importable."""
        model = make_cghc(N, M, EMBED_DIM, device="cpu")
        assert model._use_triton is False


# ---------------------------------------------------------------------------
# CGHCF: the forced-exp block kernel toggles on the same flag
# ---------------------------------------------------------------------------

class TestCGHCFDispatch:
    @pytest.mark.parametrize("device", DEVICES)
    def test_block_kernel_matches_device(self, device):
        model = make_cghcf(N, M, EMBED_DIM, device=device)
        assert model._use_triton == should_use_triton(device)

    @pytest.mark.parametrize("device", DEVICES)
    def test_forward_runs_on_device(self, device):
        model = make_cghcf(N, M, EMBED_DIM, device=device)
        out = model(_input(model, device))
        assert out.device.type == torch.device(device).type
        assert out.isfinite().all()


# ---------------------------------------------------------------------------
# The failure the toggle prevents
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TRITON, reason="requires triton to be importable")
def test_forcing_triton_on_cpu_raises():
    """Overriding the toggle (use_triton=True on CPU) reproduces the CUDA-only error,
    documenting why device-driven toggling is necessary."""
    model = make_cghc(N, M, EMBED_DIM, device="cpu", use_triton=True)
    with pytest.raises(RuntimeError, match="CUDA"):
        model(_input(model, "cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_model_uses_triton_when_available():
    model = make_cghc(N, M, EMBED_DIM, device="cuda")
    assert model._use_triton == HAS_TRITON
    out = model(_input(model, "cuda"))
    assert out.is_cuda and out.isfinite().all()
