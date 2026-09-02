import pytest
import torch

from hyperconnections.temporal_writeback import TemporalWriteback
from tests.test_cghc import make_cghc


def test_component_shape_and_original_output():
    augmentation = TemporalWriteback(8, kernel_sizes=(2, 4, 6))
    output = torch.randn(2, 3, 5, 8)

    components = augmentation(output)

    assert components.shape == (2, 3, 5, 4, 8)
    assert torch.equal(components[..., 0, :], output)


def test_components_are_orthogonal_per_token():
    augmentation = TemporalWriteback(16, kernel_sizes=(2, 4, 6))
    components = augmentation(torch.randn(2, 7, 16))
    gram = torch.einsum("bskd,bsjd->bskj", components, components)
    off_diagonal = gram.masked_select(
        ~torch.eye(augmentation.num_components, dtype=torch.bool)
    )

    assert torch.allclose(off_diagonal, torch.zeros_like(off_diagonal), atol=1e-5)


def test_augmentation_is_causal():
    augmentation = TemporalWriteback(8, kernel_sizes=(2, 4))
    output = torch.randn(2, 7, 8)
    changed = output.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])

    before = augmentation(output)
    after = augmentation(changed)

    assert torch.allclose(before[:, :4], after[:, :4])


def test_sequence_dimension_is_required():
    augmentation = TemporalWriteback(8, kernel_sizes=(2,))

    with pytest.raises(ValueError, match="requires a sequence dimension"):
        augmentation(torch.randn(3, 8))


def test_cghc_expands_writeback_projection_and_preserves_output_shape():
    model = make_cghc(4, 2, 8, writeback_kernel_sizes=(2, 4, 6))
    x = torch.randn(2, 7, model.input_dim, requires_grad=True)
    flat_x = x.reshape(-1, model.input_dim)
    projections = model.input_proj(model.norm(flat_x))

    write_weights, _ = model.compute_read_write_weights(projections)
    output = model(x)
    output.sum().backward()

    assert model.write_out.shape == (model.n, 4, model.m)
    assert write_weights.shape == (14, model.n, 4, model.m)
    assert output.shape == x.shape
    assert all(weight.grad is not None for weight in model.writeback.weights)


def test_cghc_enriched_writeback_is_causal():
    model = make_cghc(4, 2, 8, writeback_kernel_sizes=(2, 4))
    x = torch.randn(2, 7, model.input_dim)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])

    before = model(x)
    after = model(changed)

    assert torch.allclose(before[:, :4], after[:, :4], atol=1e-6)


def test_cghc_enriched_writeback_requires_sequence_dimension():
    model = make_cghc(4, 2, 8, writeback_kernel_sizes=(2,))

    with pytest.raises(ValueError, match="requires a sequence dimension"):
        model(torch.randn(3, model.input_dim))
