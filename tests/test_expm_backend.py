"""expm_backend: flash-hypercons' CUDA matrix exponential against the default Triton T18."""
import importlib.util

import pytest
import torch
from torch import nn

from hyperconnections.cghc import ContinuousGenHyperConnections, expm
from hyperconnections.cghcf import ContinuousGenHyperConnectionsForced

flash = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("flash_hypercons") is None,
    reason="needs CUDA and flash_hypercons")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        ContinuousGenHyperConnections(n=4, m=1, input_dim=64, embed_dim=16, module=nn.Identity(),
                                      expm_backend="cusolver")


def test_default_backend_follows_use_triton():
    model = ContinuousGenHyperConnections(n=4, m=1, input_dim=64, embed_dim=16, module=nn.Identity(),
                                          use_triton=False)
    assert model.expm_backend == "eager"


@flash
def test_flash_matches_triton_values_and_gradients():
    torch.manual_seed(0)
    A = (0.5 * torch.randn(4096, 4, 4, device="cuda")).requires_grad_()
    for backend in ("triton", "flash"):
        out = expm(A, backend)
        grad, = torch.autograd.grad(out, A, torch.ones_like(out))
        if backend == "triton":
            ref_out, ref_grad = out.detach(), grad
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=1e-4)


@flash
@pytest.mark.parametrize("cls", [ContinuousGenHyperConnections, ContinuousGenHyperConnectionsForced])
def test_flash_backend_in_module(cls):
    torch.manual_seed(0)
    kwargs = dict(n=4, m=1, input_dim=64, embed_dim=16, module=nn.Identity(), projection="mean")
    reference = cls(**kwargs).cuda()
    flash_model = cls(**kwargs, expm_backend="flash").cuda()
    flash_model.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for p in (reference.input_proj.fused.weight, flash_model.input_proj.fused.weight):
            torch.manual_seed(1)
            p.normal_(std=0.1)
    x = torch.randn(3, 8, 64, device="cuda")
    torch.testing.assert_close(flash_model(x), reference(x), rtol=1e-4, atol=1e-4)
