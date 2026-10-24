"""Interfaces for fused callers: mixing_matrix (CGHC) and weights_from_logits (identity, mHC)."""
import pytest
import torch
from torch import nn

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.identity_hc import IdentityHyperConnections
from hyperconnections.mhc import ManifoldHyperConnections

from tests.conftest import should_use_triton


@pytest.mark.parametrize('projection', ['none', 'mean', 'v'])
@pytest.mark.parametrize('variant', ['one_sided', 'two_sided', 'inflow', 'generator'])
def test_mixing_matrix_matches_stream_mix(device, projection, variant):
    if variant == 'inflow' and projection == 'none':
        pytest.skip('inflow needs a projection')
    torch.manual_seed(0)
    model = ContinuousGenHyperConnections(
        n=4, m=1, input_dim=64, embed_dim=16, module=nn.Identity(), projection=projection,
        projection_variant=variant,
        projection_decay=0.9 if projection != 'none' and variant != 'generator' else 1.0,
        projection_inflow=0.3,
        use_triton=should_use_triton(device)).to(device)
    with torch.no_grad():
        model.input_proj.fused.weight.normal_(std=0.1)
    streams = torch.randn(6, 4, 16, device=device)
    source = torch.randn(6, 4, 16, device=device)
    projections = model.input_proj(model.norm(streams.flatten(1)))
    transition = model.compute_transition(projections).float()
    direction = None if variant == 'generator' else model.compute_projection(projections)
    want = model._stream_mix(streams, transition, source, direction, inflow=model.compute_inflow(projections))
    mixing = model.mixing_matrix(transition, projections)
    torch.testing.assert_close(torch.einsum('bij,bjd->bid', mixing, streams) + source, want,
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('writeback', [(), (2, 4)])
def test_identity_weights_from_logits(writeback):
    torch.manual_seed(0)
    model = IdentityHyperConnections(n=4, m=1, input_dim=64, embed_dim=16, module=nn.Identity(),
                                     writeback_kernel_sizes=writeback)
    x_norm = torch.randn(5, 64)
    want = model.compute_read_write_weights(x_norm)
    k = len(writeback) + 1
    got = model.weights_from_logits(model.proj_read_in(x_norm).reshape(5, 4, 1),
                                    model.proj_write_out(x_norm).reshape(5, 4, k, 1))
    for a, b in zip(got, want):
        assert torch.equal(a, b)


@pytest.mark.parametrize('writeback', [(), (2, 4)])
def test_mhc_weights_from_logits(writeback):
    torch.manual_seed(0)
    model = ManifoldHyperConnections(n=4, m=1, input_dim=64, embed_dim=16, module=nn.Identity(),
                                     writeback_kernel_sizes=writeback)
    x = torch.randn(5, 4, 16)
    want = model.compute_mixing_weights(x)
    x_norm = model.norm(x.view(5, -1))
    k = len(writeback) + 1
    got = model.weights_from_logits(model.proj_read_in(x_norm).reshape(5, 4, 1),
                                    model.proj_write_out(x_norm).reshape(5, 4, k, 1),
                                    model.proj_stream_mixing(x_norm).reshape(5, 4, 4))
    for a, b in zip(got, want):
        assert torch.equal(a, b)
