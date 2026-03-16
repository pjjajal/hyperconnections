"""
Tests for GeneralizedHyperConnections (ghc.py).

Configurations used throughout: (n, m, embed_dim)
  - (2, 1, 8)  : regular hyperconnections (m=1)
  - (2, 2, 8)  : regular connections (m=n)
  - (4, 2, 8)  : generalized, r=2
  - (4, 4, 8)  : regular connections (m=n)
  - (6, 2, 8)  : generalized, r=4
  - (6, 3, 12)  : generalized, r=3
"""

import math

import pytest
import torch
import torch.nn as nn

from hyperconnections.ghc import GeneralizedHyperConnections
from tests.conftest import IdentityModule, ZeroModule, make_ghc

CONFIGS = [
    (2, 1, 8),
    (2, 2, 8),
    (4, 2, 8),
    (4, 4, 8),
    (6, 2, 8),
    (6, 3, 12),
]


class TestInit:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_parameter_shapes(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        assert ghc.write_out.shape == (n, m)
        assert ghc.dynamic_scaling_write_out.shape == (n, m)
        assert ghc.read_in.shape == (n, m)
        assert ghc.dynamic_scaling_read_in.shape == (n, m)
        assert ghc.stream_mixing.shape == (n, n)
        assert ghc.dynamic_scaling_stream_mixing.shape == (n, n)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_linear_layer_shapes(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        block_size = embed_dim // m
        assert ghc.dynamic_scaling_weight_write_out.in_features == block_size
        assert ghc.dynamic_scaling_weight_write_out.out_features == m
        assert ghc.dynamic_scaling_weight_read_in.in_features == block_size
        assert ghc.dynamic_scaling_weight_read_in.out_features == m
        assert ghc.dynamic_scaling_weight_stream_mixing.in_features == block_size
        assert ghc.dynamic_scaling_weight_stream_mixing.out_features == n

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_block_size_and_scaling_factor(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        assert ghc.block_size == embed_dim // m
        assert ghc.scaling_factor == math.sqrt(embed_dim // m)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_write_out_cyclic_init(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        w = ghc.write_out.data
        for i in range(n):
            for j in range(m):
                expected = 1.0 if j == i % m else 0.0
                assert w[i, j].item() == expected, f"write_out[{i},{j}]: expected {expected}"

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_read_in_identity_block_init(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        r = ghc.read_in.data
        assert torch.allclose(r[:m, :m], torch.eye(m))
        if n > m:
            assert torch.all(r[m:, :] == 0)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_stream_mixing_identity_init(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        assert torch.allclose(ghc.stream_mixing.data, torch.eye(n))

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_dynamic_scaling_ones_init(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        assert torch.all(ghc.dynamic_scaling_write_out == 1.0)
        assert torch.all(ghc.dynamic_scaling_read_in == 1.0)
        assert torch.all(ghc.dynamic_scaling_stream_mixing == 1.0)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_dynamic_weight_linear_zero_init(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        assert torch.all(ghc.dynamic_scaling_weight_write_out.weight == 0)
        assert torch.all(ghc.dynamic_scaling_weight_read_in.weight == 0)
        assert torch.all(ghc.dynamic_scaling_weight_stream_mixing.weight == 0)

    def test_invalid_input_dim_raises(self):
        with pytest.raises(AssertionError):
            GeneralizedHyperConnections(
                n=4, m=2, input_dim=99, embed_dim=8, module=IdentityModule()
            )

    def test_embed_dim_not_divisible_by_m_raises(self):
        with pytest.raises(AssertionError):
            GeneralizedHyperConnections(
                n=6, m=3, input_dim=16, embed_dim=8, module=IdentityModule()
            )

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_init_weights_is_idempotent(self, n, m, embed_dim):
        """Calling init_weights() again should reset to the same initial state."""
        ghc = make_ghc(n, m, embed_dim)
        # Corrupt params
        ghc.write_out.data.fill_(99.0)
        ghc.stream_mixing.data.fill_(99.0)
        ghc.dynamic_scaling_weight_write_out.weight.data.fill_(99.0)
        # Re-init and verify
        ghc.init_weights()
        assert torch.allclose(ghc.stream_mixing.data, torch.eye(n))
        assert torch.all(ghc.dynamic_scaling_weight_write_out.weight == 0)


class TestComputeMixingWeightsShapes:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_output_shapes(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, n, embed_dim // m)
        write_out, read_in, stream_mixing = ghc.compute_mixing_weights(x)
        assert write_out.shape == (B, n, m)
        assert read_in.shape == (B, m, n)
        assert stream_mixing.shape == (B, n, n)


class TestStaticBehavior:
    """
    With dynamic_scaling_weight linear layers zeroed (initial state),
    tanh(W·x / τ) = tanh(0) = 0, so every computed matrix collapses to
    its static value: dynamic_scaling * 0 + static = static.
    """

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_write_out_equals_static(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, n, embed_dim // m)
        write_out, _, _ = ghc.compute_mixing_weights(x)
        expected = ghc.write_out.unsqueeze(0).expand(B, -1, -1)
        assert torch.allclose(write_out, expected)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_read_in_equals_static_transposed(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, n, embed_dim // m)
        _, read_in, _ = ghc.compute_mixing_weights(x)
        expected = ghc.read_in.T.unsqueeze(0).expand(B, -1, -1)
        assert torch.allclose(read_in, expected)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_stream_mixing_equals_static(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, n, embed_dim // m)
        _, _, stream_mixing = ghc.compute_mixing_weights(x)
        expected = ghc.stream_mixing.unsqueeze(0).expand(B, -1, -1)
        assert torch.allclose(stream_mixing, expected)


class TestForwardShape:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_output_shape_matches_input(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        B = 3
        input_dim = (n * embed_dim) // m
        x = torch.randn(B, input_dim)
        assert ghc(x).shape == (B, input_dim)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_batch_size_one(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        x = torch.randn(1, (n * embed_dim) // m)
        assert ghc(x).shape == (1, (n * embed_dim) // m)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_extra_leading_dims(self, n, m, embed_dim):
        """Input [B, S, input_dim] should produce output of the same shape."""
        ghc = make_ghc(n, m, embed_dim)
        B, S, input_dim = 2, 7, (n * embed_dim) // m
        x = torch.randn(B, S, input_dim)
        assert ghc(x).shape == (B, S, input_dim)


class TestForwardBehavior:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_zero_module_output_equals_input(self, n, m, embed_dim):
        """
        ZeroModule outputs 0, so write_out @ 0 = 0.
        stream_mixing = I_n, so the residual passes x through unchanged.
        Expected: output == input.
        """
        ghc = make_ghc(n, m, embed_dim, module=ZeroModule())
        x = torch.randn(3, (n * embed_dim) // m)
        assert torch.allclose(ghc(x), x, atol=1e-6)

    @pytest.mark.parametrize("n,m,embed_dim", [(2, 2, 8), (4, 4, 8)])
    def test_identity_module_m_equals_n_doubles_input(self, n, m, embed_dim):
        """
        When m=n: write_out=I_n, read_in^T=I_n, stream_mixing=I_n.
        IdentityModule returns x unchanged.
        x_read_in = x, out_written = x, x_mixed = x → output = x + x = 2x.
        """
        ghc = make_ghc(n, m, embed_dim, module=IdentityModule())
        x = torch.randn(3, embed_dim)  # input_dim = embed_dim when m=n
        assert torch.allclose(ghc(x), 2 * x, atol=1e-6)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_module_receives_correct_input_shape(self, n, m, embed_dim):
        """Inner module should receive a tensor of shape [B, embed_dim]."""
        received_shapes = []

        class ShapeCapture(nn.Module):
            def forward(self, x, **kwargs):
                received_shapes.append(x.shape)
                return torch.zeros_like(x)

        B = 5
        ghc = make_ghc(n, m, embed_dim, module=ShapeCapture())
        ghc(torch.randn(B, (n * embed_dim) // m))
        assert len(received_shapes) == 1
        assert received_shapes[0] == (B, embed_dim)


class TestGradientFlow:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_backward_produces_gradients(self, n, m, embed_dim):
        ghc = make_ghc(n, m, embed_dim)
        x = torch.randn(3, (n * embed_dim) // m, requires_grad=True)
        ghc(x).sum().backward()
        assert x.grad is not None
        assert ghc.write_out.grad is not None
        assert ghc.read_in.grad is not None
        assert ghc.stream_mixing.grad is not None
        assert ghc.dynamic_scaling_write_out.grad is not None
        assert ghc.dynamic_scaling_read_in.grad is not None
        assert ghc.dynamic_scaling_stream_mixing.grad is not None
