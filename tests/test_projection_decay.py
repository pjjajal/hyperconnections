from types import SimpleNamespace

import pytest
import torch

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.ops import HAS_TRITON


@pytest.mark.parametrize('backend', ['eager', 'small', 'big'])
@pytest.mark.parametrize('decay', [0., .99, 1.])
def test_decay_forward_and_all_gradients(backend, decay):
    if backend != 'eager' and not (HAS_TRITON and torch.cuda.is_available()):
        pytest.skip('CUDA Triton required')
    device = 'cpu' if backend == 'eager' else 'cuda'
    n = 16 if backend == 'big' else 4
    torch.manual_seed(19)
    t = (torch.randn(2, n, n, device=device)*.1).requires_grad_()
    x = torch.randn(2, n, 32, device=device, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    v = torch.nn.functional.normalize(torch.randn(2, n, device=device), dim=-1).requires_grad_()
    if backend == 'eager':
        actual = ContinuousGenHyperConnections._stream_mix(
            SimpleNamespace(projection_variant='one_sided', projection_decay=decay, _use_triton=False),
            x, t, y, v)
    else:
        from hyperconnections.ops.stream_mix_small_nb import stream_mix_add_small_nb
        from hyperconnections.ops.stream_mix_big_nb import stream_mix_add_big_nb
        fn = stream_mix_add_small_nb if backend == 'small' else stream_mix_add_big_nb
        actual = fn(t, x, y, v, decay)
    p = v.unsqueeze(-1)*v.unsqueeze(-2)
    reference = (decay*p+t@(torch.eye(n, device=device)-p))@x+y
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
    g = torch.randn_like(actual)
    ag = torch.autograd.grad(actual, (t, x, y, v), g, retain_graph=True)
    rg = torch.autograd.grad(reference, (t, x, y, v), g)
    for a, r in zip(ag, rg):
        torch.testing.assert_close(a, r, atol=1e-4, rtol=1e-4)


def test_preserved_direction_decays():
    hc = ContinuousGenHyperConnections(n=4, m=1, input_dim=32, embed_dim=8,
        module=torch.nn.Identity(), use_triton=False, projection='v', projection_decay=.99)
    v = torch.ones(1, 4)/2
    x = v.unsqueeze(-1).expand(1, 4, 8)
    out = hc._stream_mix(x, torch.randn(1, 4, 4), torch.zeros_like(x), v)
    torch.testing.assert_close(out, .99*x)
