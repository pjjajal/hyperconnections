"""
Tests for ContinuousGenHyperConnections (cghc.py).

Configurations used throughout: (n, m, embed_dim)
  - (2, 1, 8)  : hyperconnections (m=1)
  - (4, 2, 8)  : generalized, r=2
  - (4, 4, 8)  : regular connections (m=n)
  - (6, 3, 12) : generalized, r=3
"""

import math

import pytest
import torch
import torch.nn as nn

from hyperconnections.cghc import ContinuousGenHyperConnections
from tests.conftest import IdentityModule, ZeroModule, should_use_triton


CONFIGS = [
    (2, 1, 8),
    (4, 2, 8),
    (4, 4, 8),
    (6, 3, 12),
]

ALL_GENERATOR_TYPES = [
    "conservative",
    "psd_diss",
    "diagonal_diss",
    "laplacian",
    "conservative_diag_diss",
    "conservative_psd_diss",
    "conservative_laplacian",
]


def make_cghc(
    n: int,
    m: int,
    embed_dim: int,
    module: nn.Module | None = None,
    device: str = "cpu",
    use_triton: bool | None = None,
    **kwargs,
) -> ContinuousGenHyperConnections:
    input_dim = (n * embed_dim) // m
    if module is None:
        module = IdentityModule()
    # Toggle the custom-kernel flag to match the device unless explicitly set:
    # Triton kernels need CUDA tensors, so a CPU model must run the eager path.
    if use_triton is None:
        use_triton = should_use_triton(device)
    return ContinuousGenHyperConnections(
        n=n, m=m, input_dim=input_dim, embed_dim=embed_dim, module=module,
        use_triton=use_triton, **kwargs
    ).to(device)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_parameter_shapes(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        assert cghc.read_in.shape == (n, m)
        assert cghc.write_out.shape == (n, m)
        assert cghc.alpha_read_in.shape == (1,)
        assert cghc.alpha_write_out.shape == (1,)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_block_size(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        assert cghc.block_size == embed_dim // m

    def test_invalid_embed_dim_not_divisible_by_m_raises(self):
        with pytest.raises(AssertionError):
            make_cghc(n=6, m=4, embed_dim=10)

    def test_invalid_input_dim_raises(self):
        with pytest.raises(AssertionError):
            ContinuousGenHyperConnections(
                n=4, m=2, input_dim=99, embed_dim=8, module=IdentityModule()
            )

    # Regression test for the conservative_laplacian bug fix
    def test_conservative_laplacian_creates_conserv_params(self):
        cghc = make_cghc(4, 2, 8, generator_type="conservative_laplacian")
        assert hasattr(cghc, "conserv_A"), "conserv_A must exist for conservative_laplacian"
        assert "conserv" in cghc.input_proj._slices, (
            "fused projection must have a conserv segment for conservative_laplacian"
        )
        assert hasattr(cghc, "laplacian_A"), "laplacian_A must exist for conservative_laplacian"

    @pytest.mark.parametrize("generator_type", ALL_GENERATOR_TYPES)
    def test_generator_type_constructs_without_error(self, generator_type):
        make_cghc(4, 2, 8, generator_type=generator_type)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_read_in_init_value(self, n, m, embed_dim):
        """m=1: all read_in entries ≈ logit(1/n). m>1: block-diagonal (on ≈5, off ≈-5)."""
        cghc = make_cghc(n, m, embed_dim)
        # [n, m] stored; transpose to [m, n] semantic view
        W = cghc.read_in.detach()  # [n, m]
        if m == 1:
            expected_logit = math.log(1.0 / (n - 1)) if n > 1 else 10.0
            assert torch.allclose(W, torch.full_like(W, expected_logit), atol=0.05)
        else:
            for j in range(m):
                assert W[j, j].item() == pytest.approx(5.0, abs=0.05), f"on-diag [{j},{j}]"
                for k in range(m):
                    if k != j:
                        assert W[j, k].item() == pytest.approx(-5.0, abs=0.05), f"off-diag [{j},{k}]"

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_write_out_init_value(self, n, m, embed_dim):
        """Round-robin: on-position bias ≈ 0 (2·σ(0)=1), off-position bias ≈ -5."""
        cghc = make_cghc(n, m, embed_dim)
        W = cghc.write_out.detach()  # [n, m]
        for i in range(n):
            on = i % m
            assert W[i, on].item() == pytest.approx(0.0, abs=0.05), f"on [{i},{on}]"
            for j in range(m):
                if j != on:
                    assert W[i, j].item() == pytest.approx(-5.0, abs=0.05), f"off [{i},{j}]"

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_alpha_negligible_at_init(self, n, m, embed_dim):
        """Alpha gating should be 0.01 so dynamic component starts negligible."""
        cghc = make_cghc(n, m, embed_dim)
        assert cghc.alpha_read_in.item() == pytest.approx(0.01)
        assert cghc.alpha_write_out.item() == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Generator structural properties
# ---------------------------------------------------------------------------


class TestGeneratorStructure:
    """After perturbing parameters, verify each generator type maintains its structural guarantee."""

    def _perturb(self, cghc: ContinuousGenHyperConnections):
        """Add random noise to generator params so we don't just test at zero."""
        with torch.no_grad():
            if hasattr(cghc, "conserv_A"):
                cghc.conserv_A.add_(torch.randn_like(cghc.conserv_A) * 0.1)
            if hasattr(cghc, "diss_A"):
                cghc.diss_A.add_(torch.randn_like(cghc.diss_A) * 0.1)
            if hasattr(cghc, "diss_diag"):
                cghc.diss_diag.add_(torch.randn_like(cghc.diss_diag) * 0.5)
            if hasattr(cghc, "laplacian_A"):
                # log-scale starts at log(1e-3) so the Laplacian is negligible;
                # crank the scale to 1 so the perturbed generator has a visible
                # dissipative part (anchor is already 0 → softplus(0)≈0.693 adjacency)
                cghc.laplacian_log_scale.fill_(0.0)
                cghc.laplacian_A.add_(torch.randn_like(cghc.laplacian_A) * 0.1)

    def _proj(self, cghc, B=3):
        return cghc.input_proj(torch.randn(B, cghc.input_dim))

    def test_conservative_is_skew_symmetric(self):
        cghc = make_cghc(4, 2, 8, generator_type="conservative")
        self._perturb(cghc)
        A = cghc.compute_generator(self._proj(cghc))
        assert torch.allclose(A + A.transpose(-1, -2), torch.zeros_like(A), atol=1e-5)

    def test_psd_diss_is_negative_semidefinite(self):
        cghc = make_cghc(4, 2, 8, generator_type="psd_diss")
        self._perturb(cghc)
        A = cghc.compute_generator(self._proj(cghc))
        eigvals = torch.linalg.eigvalsh(A)
        assert (eigvals <= 1e-4).all(), "psd_diss generator must be NSD"

    def test_diagonal_diss_has_nonpositive_diagonal_and_zero_offdiag(self):
        cghc = make_cghc(4, 2, 8, generator_type="diagonal_diss")
        self._perturb(cghc)
        A = cghc.compute_generator(self._proj(cghc))
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        off_diag_mask = ~torch.eye(cghc.n, dtype=torch.bool)
        assert (diag <= 0).all(), "diagonal entries must be non-positive"
        assert torch.allclose(
            A[:, off_diag_mask], torch.zeros(3, cghc.n * cghc.n - cghc.n), atol=1e-6
        ), "off-diagonal entries must be zero"

    def test_laplacian_A_small_at_init(self):
        """laplacian_log_scale=log(1e-3) at init gives ~7e-4 adjacency → A near zero but not exact."""
        cghc = make_cghc(4, 2, 8, generator_type="laplacian")
        A = cghc.compute_generator(self._proj(cghc))
        assert torch.allclose(A, torch.zeros_like(A), atol=1e-2), "laplacian A must be near zero at init"

    def test_conservative_laplacian_is_neither_symmetric_nor_skew(self):
        """Combined generator should have both conservative and dissipative parts."""
        cghc = make_cghc(4, 2, 8, generator_type="conservative_laplacian")
        self._perturb(cghc)
        A = cghc.compute_generator(self._proj(cghc))
        # Not purely skew-symmetric (laplacian adds symmetric negative part)
        assert not torch.allclose(A + A.transpose(-1, -2), torch.zeros_like(A), atol=1e-3)


# ---------------------------------------------------------------------------
# Transition matrix at initialization
# ---------------------------------------------------------------------------


class TestInitialTransition:
    """At initialization the dynamic predictors are zeroed, so A depends only on static params.
    Static params have small noise at init, so A is small (not exactly zero) and Phi≈I.
    diagonal_diss/laplacian start with log_scale=log(1e-3) → ~7e-4 dissipation.
    """

    ALL_TYPES = [
        "conservative",
        "psd_diss",
        "diagonal_diss",
        "laplacian",
        "conservative_diag_diss",
        "conservative_psd_diss",
        "conservative_laplacian",
    ]

    @pytest.mark.parametrize("generator_type", ALL_TYPES)
    def test_transition_is_near_identity_at_init(self, generator_type):
        """All types start with small A (noise * dt ≈ 1e-4) so Phi is near I."""
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type)
        x = torch.randn(3, (n * embed_dim) // m)
        Phi = cghc.compute_transition(cghc.input_proj(x))
        I = torch.eye(n).unsqueeze(0).expand_as(Phi)
        assert torch.allclose(Phi, I, atol=1e-2), (
            f"Transition should be near-I at init for generator_type='{generator_type}'"
        )

    @pytest.mark.parametrize("generator_type", ["conservative", "conservative_psd_diss", "conservative_laplacian"])
    def test_transition_is_orthogonal_after_perturbation(self, generator_type):
        """exp(skew-sym) is orthogonal; mixed types with conservative component should satisfy Phi Phi^T ≈ I only when purely conservative."""
        if generator_type != "conservative":
            pytest.skip("orthogonality only holds for purely conservative generator")
        cghc = make_cghc(4, 2, 8, generator_type=generator_type)
        with torch.no_grad():
            cghc.conserv_A.add_(torch.randn_like(cghc.conserv_A) * 0.5)
        x = torch.randn(3, cghc.input_dim)
        Phi = cghc.compute_transition(cghc.input_proj(x))
        I = torch.eye(4).unsqueeze(0).expand_as(Phi)
        assert torch.allclose(Phi @ Phi.transpose(-1, -2), I, atol=1e-4)


# ---------------------------------------------------------------------------
# Read/write weights
# ---------------------------------------------------------------------------


class TestReadWriteWeights:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_output_shapes(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, (n * embed_dim) // m)
        write_out, read_in = cghc.compute_read_write_weights(cghc.input_proj(x))
        assert write_out.shape == (B, n, m)
        assert read_in.shape == (B, m, n)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_read_in_static_at_init(self, n, m, embed_dim):
        """With alpha=0.01, read_in ≈ sigmoid(bias). m=1: all ≈ 1/n. m>1: block-diagonal."""
        cghc = make_cghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, (n * embed_dim) // m)
        _, read_in = cghc.compute_read_write_weights(cghc.input_proj(x))  # [B, m, n]
        if m == 1:
            expected = 1.0 / n
            assert torch.allclose(read_in, torch.full_like(read_in, expected), atol=0.02)
        else:
            # Block-diagonal: read_in[b, j, j] ≈ sigmoid(5) ≈ 0.993, off ≈ sigmoid(-5) ≈ 0.007
            for j in range(m):
                assert read_in[:, j, j].mean().item() == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item(), abs=0.05)
                for k in range(n):
                    if k != j:
                        assert read_in[:, j, k].mean().item() == pytest.approx(torch.sigmoid(torch.tensor(-5.0)).item(), abs=0.05)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_write_out_static_at_init(self, n, m, embed_dim):
        """With alpha=0.01, write_out ≈ 2*sigmoid(bias). On-position bias=0 → 1. Off-position bias=-5 → ~0.013."""
        cghc = make_cghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, (n * embed_dim) // m)
        write_out, _ = cghc.compute_read_write_weights(cghc.input_proj(x))  # [B, n, m]
        for i in range(n):
            on = i % m
            assert write_out[:, i, on].mean().item() == pytest.approx(1.0, abs=0.05)
            for j in range(m):
                if j != on:
                    assert write_out[:, i, j].mean().item() == pytest.approx(2 * torch.sigmoid(torch.tensor(-5.0)).item(), abs=0.05)


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


class TestForwardShape:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_output_shape_matches_input(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        B = 3
        input_dim = (n * embed_dim) // m
        x = torch.randn(B, input_dim)
        assert cghc(x).shape == (B, input_dim)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_batch_size_one(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        input_dim = (n * embed_dim) // m
        x = torch.randn(1, input_dim)
        assert cghc(x).shape == (1, input_dim)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_extra_leading_dims(self, n, m, embed_dim):
        """Input [B, S, input_dim] should produce output of the same shape."""
        cghc = make_cghc(n, m, embed_dim)
        B, S, input_dim = 2, 7, (n * embed_dim) // m
        x = torch.randn(B, S, input_dim)
        assert cghc(x).shape == (B, S, input_dim)

    @pytest.mark.parametrize("generator_type", ALL_GENERATOR_TYPES)
    def test_all_generator_types_produce_correct_shape(self, generator_type):
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type)
        x = torch.randn(3, (n * embed_dim) // m)
        assert cghc(x).shape == x.shape

    @pytest.mark.parametrize("projection", ["mean", "v", "none"])
    def test_all_projection_modes_produce_correct_shape(self, projection):
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, projection=projection)
        x = torch.randn(3, (n * embed_dim) // m)
        assert cghc(x).shape == x.shape


# ---------------------------------------------------------------------------
# Forward behavior
# ---------------------------------------------------------------------------


class TestForwardBehavior:
    ALL_GENERATOR_TYPES = [
        "conservative",
        "psd_diss",
        "diagonal_diss",
        "laplacian",
        "conservative_diag_diss",
        "conservative_psd_diss",
        "conservative_laplacian",
    ]

    @pytest.mark.parametrize("generator_type", ALL_GENERATOR_TYPES)
    def test_zero_module_output_near_input_at_init(self, generator_type):
        """
        At init, static params have small noise so Phi≈I (A~O(noise*dt)~1e-4).
        ZeroModule gives Y=0, so output = Phi @ x ≈ x.
        """
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type, module=ZeroModule())
        x = torch.randn(3, (n * embed_dim) // m)
        assert torch.allclose(cghc(x), x, atol=1e-2)

    @pytest.mark.parametrize("generator_type", ["diagonal_diss", "conservative_diag_diss"])
    def test_zero_module_output_near_input_at_init_diag_diss(self, generator_type):
        """diagonal_diss has Phi≈I (not exact) so output≈x with ZeroModule."""
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type, module=ZeroModule())
        x = torch.randn(3, (n * embed_dim) // m)
        assert torch.allclose(cghc(x), x, atol=0.02)

    def test_module_receives_correct_embed_dim(self):
        """Inner module receives [B, embed_dim] when input is [B, input_dim]."""
        received = []

        class ShapeCapture(nn.Module):
            def forward(self, x, **kwargs):
                received.append(x.shape)
                return torch.zeros_like(x)

        n, m, embed_dim = 4, 2, 8
        B = 5
        cghc = make_cghc(n, m, embed_dim, module=ShapeCapture())
        cghc(torch.randn(B, (n * embed_dim) // m))
        assert len(received) == 1
        assert received[0] == (B, embed_dim)

    def test_module_receives_correct_shape_with_leading_dims(self):
        """Inner module receives [B, S, embed_dim] when input is [B, S, input_dim]."""
        received = []

        class ShapeCapture(nn.Module):
            def forward(self, x, **kwargs):
                received.append(x.shape)
                return torch.zeros_like(x)

        n, m, embed_dim = 4, 2, 8
        B, S = 2, 7
        cghc = make_cghc(n, m, embed_dim, module=ShapeCapture())
        cghc(torch.randn(B, S, (n * embed_dim) // m))
        assert len(received) == 1
        assert received[0] == (B, S, embed_dim)

    @pytest.mark.parametrize("projection", ["mean", "v", "none"])
    def test_zero_module_with_projections_at_init(self, projection):
        """Projection doesn't change the near-identity result with ZeroModule."""
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(
            n, m, embed_dim, projection=projection, module=ZeroModule(), generator_type="conservative_psd_diss"
        )
        x = torch.randn(3, (n * embed_dim) // m)
        assert torch.allclose(cghc(x), x, atol=1e-2)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_backward_produces_gradients(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        x = torch.randn(3, (n * embed_dim) // m, requires_grad=True)
        cghc(x).sum().backward()
        assert x.grad is not None
        assert cghc.read_in.grad is not None
        assert cghc.write_out.grad is not None

    @pytest.mark.parametrize("generator_type", ALL_GENERATOR_TYPES)
    def test_backward_through_all_generator_types(self, generator_type):
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type)
        x = torch.randn(3, (n * embed_dim) // m, requires_grad=True)
        cghc(x).sum().backward()
        assert x.grad is not None
