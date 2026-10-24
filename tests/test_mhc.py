import pytest
import torch
import torch.nn as nn

from hyperconnections.mhc import ManifoldHyperConnections
from tests.conftest import IdentityModule


def make_mhc(
    n: int = 4,
    m: int = 1,
    embed_dim: int = 8,
    module: nn.Module | None = None,
    **kwargs,
) -> ManifoldHyperConnections:
    return ManifoldHyperConnections(
        n=n,
        m=m,
        input_dim=n * (embed_dim // m),
        embed_dim=embed_dim,
        module=module or IdentityModule(),
        **kwargs,
    )


class TestInitialization:
    @pytest.mark.parametrize("layer_id", range(6))
    def test_paper_read_bias_rotates_with_layer(self, layer_id):
        mhc = make_mhc(layer_id=layer_id)
        expected = torch.full((4, 1), -3.0)
        expected[layer_id % 4, 0] = 3.0
        assert torch.equal(mhc.read_in.detach(), expected)

    def test_generalized_read_bias_assigns_each_fraction(self):
        mhc = make_mhc(n=4, m=2, layer_id=1)
        expected = torch.full((4, 2), -3.0)
        expected[2, 0] = 3.0
        expected[3, 1] = 3.0
        assert torch.equal(mhc.read_in.detach(), expected)

    def test_paper_residual_logit_gap_and_zero_dynamic_weights(self):
        mhc = make_mhc()
        assert torch.equal(mhc.stream_mixing.detach(), 6.0 * torch.eye(4))
        assert torch.count_nonzero(mhc.proj_read_in.weight) == 0
        assert torch.count_nonzero(mhc.proj_write_out.weight) == 0
        assert torch.count_nonzero(mhc.proj_stream_mixing.weight) == 0
        assert mhc.alpha_read_in.item() == pytest.approx(0.01)
        assert mhc.alpha_write_out.item() == pytest.approx(0.01)
        assert mhc.alpha_stream_mixing.item() == pytest.approx(0.01)

    def test_norm_uses_paper_epsilon(self):
        assert make_mhc().norm.eps == pytest.approx(1e-20)

    def test_nonintegral_expansion_is_supported(self):
        mhc = ManifoldHyperConnections(
            n=3,
            m=2,
            input_dim=12,
            embed_dim=8,
            module=IdentityModule(),
        )
        assert mhc(torch.randn(2, 12)).shape == (2, 12)


class TestSinkhorn:
    def test_is_doubly_stochastic(self):
        mhc = make_mhc()
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(7, 4, 4, generator=generator) * 2.0
        mixing = mhc._sinkhorn_knopp(logits)
        ones = torch.ones(7, 4)
        # Finite Sinkhorn iterations make the first-normalized axis approximate;
        # the final-normalized axis is tighter.
        assert torch.allclose(mixing.sum(dim=-1), ones, atol=5e-3)
        assert torch.allclose(mixing.sum(dim=-2), ones, atol=2e-5)
        assert torch.isfinite(mixing).all()

    def test_large_logits_do_not_overflow(self):
        mhc = make_mhc()
        logits = torch.tensor(
            [[[1e4, -1e4, 0.0, 1.0]] * 4],
            requires_grad=True,
        )
        mixing = mhc._sinkhorn_knopp(logits)
        assert torch.isfinite(mixing).all()
        mixing.square().sum().backward()
        assert torch.isfinite(logits.grad).all()


class TestTraining:
    def test_dynamic_projections_receive_gradients_from_zero_init(self):
        mhc = make_mhc(module=nn.Linear(8, 8, bias=False))
        x = torch.randn(2, 3, 32, requires_grad=True)
        mhc(x).square().mean().backward()
        for projection in (
            mhc.proj_read_in,
            mhc.proj_write_out,
            mhc.proj_stream_mixing,
        ):
            assert projection.weight.grad is not None
            assert torch.isfinite(projection.weight.grad).all()
            assert projection.weight.grad.abs().sum() > 0

    def test_no_decay_names_cover_all_static_anchors(self):
        assert ManifoldHyperConnections.NO_DECAY_PARAM_NAMES == {
            "read_in",
            "write_out",
            "stream_mixing",
        }
