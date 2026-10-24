"""Full residual mixing and explicit branch readouts."""
import pytest
import torch
from torch import nn

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.cghcf import ContinuousGenHyperConnectionsForced


class BranchModule(nn.Module):
    def forward(self, x):
        self.received = x
        return x[..., 0, :] * torch.sigmoid(x[..., -1, :])


@pytest.mark.parametrize('cls', [ContinuousGenHyperConnections, ContinuousGenHyperConnectionsForced])
@pytest.mark.parametrize('num_readouts', [1, 3])
@pytest.mark.parametrize('projection', ['none', 'mean', 'v'])
@pytest.mark.parametrize('variant', ['one_sided', 'two_sided', 'generator'])
def test_full_mixer_and_branch_readouts(cls, projection, variant, num_readouts):
    torch.manual_seed(5)
    module = BranchModule()
    model = cls(n=2, m=1, input_dim=16, embed_dim=8, module=module,
                stream_subdivisions=2, num_readouts=num_readouts, use_triton=False,
                projection=projection, projection_variant=variant, vec_dt=True)
    x = torch.randn(2, 3, 16, requires_grad=True)
    out = model(x)
    assert model.read_in.shape == (num_readouts, 2)
    assert module.received.shape == (2, 3, num_readouts, 8)
    streams = x.reshape(-1, 2, 8)
    projections = model.input_proj(model.norm(streams.flatten(1)))
    write, read = model.compute_read_write_weights(projections)
    expected_read = torch.einsum('bkn,bnd->bkd', read, streams)
    torch.testing.assert_close(module.received.reshape(-1, num_readouts, 8), expected_read)
    source = torch.einsum('bnm,bmd->bnd', write, module(module.received).reshape(-1, 1, 8))
    source = source.reshape(-1, 4, 4)
    A = model.compute_generator(projections)
    assert A.shape == (6, 4, 4)
    T = torch.matrix_exp(A)
    if cls is ContinuousGenHyperConnectionsForced:
        augmented = torch.zeros(6, 8, 8)
        augmented[:, :4, :4] = A
        augmented[:, :4, 4:] = torch.eye(4)
        source = torch.matrix_exp(augmented)[:, :4, 4:] @ source
    expected = model._stream_mix(streams.reshape(-1, 4, 4), T, source,
                                 model.compute_projection(projections))
    torch.testing.assert_close(out, expected.reshape_as(x), atol=2e-6, rtol=2e-5)
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert model.read_in.grad is not None
    assert torch.isfinite(model.input_proj.fused.weight.grad).all()


@pytest.mark.parametrize('kwargs', [{'stream_subdivisions': 3}, {'stream_subdivisions': 0}, {'num_readouts': 0}])
def test_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        ContinuousGenHyperConnections(2, 1, 16, 8, nn.Identity(), **kwargs)
