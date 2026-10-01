import pytest
import torch
import torch.nn as nn

from hyperconnections.cghc import FusedInputProjection


@pytest.mark.parametrize("elementwise_affine", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_forward_rms_normalized_matches_norm_then_project(elementwise_affine, bias):
    torch.manual_seed(0)
    n, d = 4, 64
    proj = FusedInputProjection(
        n * d, n=n, m=1, n_dt=n, bias=bias, conserv=True, diss_dim=n, proj_v=True
    ).double()
    with torch.no_grad():
        proj.fused.weight.normal_(std=0.1)
        for b in proj.biases.values():
            b.normal_()
    norm = nn.RMSNorm(n * d, elementwise_affine=elementwise_affine).double()
    if elementwise_affine:
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)

    x = (3.0 * torch.randn(7, n * d, dtype=torch.float64)).requires_grad_()
    reference = proj(norm(x))
    fused = proj.forward_rms_normalized(x, norm)

    for key in reference._fields:
        expected, actual = getattr(reference, key), getattr(fused, key)
        assert (expected is None) == (actual is None), key
        if expected is not None:
            torch.testing.assert_close(actual, expected)

    params = [x, proj.fused.weight, *proj.biases.values()]
    if elementwise_affine:
        params.append(norm.weight)
    reference_out = [t for t in reference if t is not None]
    fused_out = [t for t in fused if t is not None]
    weights = [torch.randn_like(t) for t in reference_out]
    grads_ref = torch.autograd.grad(sum((t * w).sum() for t, w in zip(reference_out, weights)), params)
    grads_fused = torch.autograd.grad(sum((t * w).sum() for t, w in zip(fused_out, weights)), params)
    for g_ref, g_fused in zip(grads_ref, grads_fused):
        torch.testing.assert_close(g_fused, g_ref)
