import pytest
import torch
import torch.nn as nn

from hyperconnections.identity_hc import IdentityHyperConnections
from hyperconnections.mhc import ManifoldHyperConnections


VARIANTS = [ManifoldHyperConnections, IdentityHyperConnections]
N = 4
M = 2
EMBED_DIM = 8
INPUT_DIM = N * EMBED_DIM // M


def make_variant(cls, **kwargs):
    return cls(
        n=N,
        m=M,
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        module=nn.Identity(),
        **kwargs,
    )


def compute_write_weights(model, x):
    streams = x.reshape(-1, model.n, model.block_size)
    if isinstance(model, ManifoldHyperConnections):
        write_weights, _, _ = model.compute_mixing_weights(streams)
    else:
        x_norm = model.norm(streams.flatten(1))
        write_weights, _ = model.compute_read_write_weights(x_norm)
    return write_weights


@pytest.mark.parametrize("variant", VARIANTS)
def test_default_writeback_shapes_are_unchanged(variant):
    model = make_variant(variant)
    write_weights = compute_write_weights(model, torch.randn(2, INPUT_DIM))

    assert model.writeback is None
    assert model.write_out.shape == (N, M)
    assert model.proj_write_out.out_features == N * M
    assert write_weights.shape == (2, N, M)


@pytest.mark.parametrize("variant", VARIANTS)
def test_enriched_writeback_shapes_and_gradients(variant):
    model = make_variant(variant, writeback_kernel_sizes=(2, 4, 6))
    x = torch.randn(2, 7, INPUT_DIM, requires_grad=True)
    write_weights = compute_write_weights(model, x)

    output = model(x)
    output.sum().backward()

    assert model.write_out.shape == (N, 4, M)
    assert model.proj_write_out.out_features == N * M * 4
    assert write_weights.shape == (14, N, 4, M)
    assert output.shape == x.shape
    assert output.isfinite().all()
    assert x.grad is not None and x.grad.isfinite().all()
    assert all(weight.grad is not None for weight in model.writeback.weights)


@pytest.mark.parametrize("variant", VARIANTS)
def test_enriched_writeback_is_causal(variant):
    model = make_variant(variant, writeback_kernel_sizes=(2, 4))
    x = torch.randn(2, 7, INPUT_DIM)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])

    before = model(x)
    after = model(changed)

    assert torch.allclose(before[:, :4], after[:, :4], atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_enriched_writeback_requires_sequence_dimension(variant):
    model = make_variant(variant, writeback_kernel_sizes=(2,))

    with pytest.raises(ValueError, match="requires a sequence dimension"):
        model(torch.randn(3, INPUT_DIM))
