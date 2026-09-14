"""Regression checks for scaling beyond the former fixed/capped ranges."""
import pytest
import torch

from hyperconnections.ops import HAS_TRITON, expm_t18, expm_t18_triton


@pytest.mark.parametrize('backend', ['eager', 'triton'])
@pytest.mark.parametrize('norm', [0.1, 48., 780., 10000.])
@pytest.mark.parametrize('upstream_scale', [0.001, 1., 1000.])
def test_rotation_and_batch_outlier(backend, norm, upstream_scale):
    if backend == 'triton' and not (HAS_TRITON and torch.cuda.is_available()):
        pytest.skip('CUDA Triton required')
    device = 'cuda' if backend == 'triton' else 'cpu'
    fn = expm_t18_triton if backend == 'triton' else expm_t18
    small = torch.tensor([[[0., -.1], [.1, 0.]]], device=device)
    a = torch.cat([small, small * (norm/.1)]).requires_grad_()
    ref_a = a.detach().double().requires_grad_()
    ref = torch.matrix_exp(ref_a)
    out = fn(a)
    torch.testing.assert_close(out.double(), ref, atol=3e-3, rtol=3e-3)
    g = torch.tensor([[[.2, -.7], [.9, .3]]], device=device).expand_as(a)*upstream_scale
    grad, = torch.autograd.grad(out, a, g)
    rg, = torch.autograd.grad(ref, ref_a, g.double())
    torch.testing.assert_close(grad.double()/upstream_scale, rg/upstream_scale,
                               atol=5e-3, rtol=5e-3)
    alone = small.requires_grad_()
    result = fn(alone)
    ag, = torch.autograd.grad(result, alone, g[:1])
    torch.testing.assert_close(out[:1], result, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad[:1], ag, atol=1e-6*upstream_scale, rtol=1e-6)


def test_insufficient_eager_budget_is_explicit():
    a = torch.tensor([[[0., -780.], [780., 0.]]])
    with pytest.raises(ValueError, match='needs 9 squarings'):
        expm_t18(a, S=8)
