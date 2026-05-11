"""
Tests for hyperconnections/ops/expm_block.py

Validates expm_t18_augmented_sparse against:
  1. expm_t18 applied to the full 2n×2n block-upper-triangular matrix
  2. torch.linalg.matrix_exp (float64) as ground truth

Run with:
    pytest tests/test_expm_block.py -v
"""

import pytest
import torch
import torch._dynamo

from hyperconnections.ops.expm import expm_t18
from hyperconnections.ops.expm_block import expm_t18_augmented_sparse


@pytest.fixture(autouse=True, scope="module")
def disable_compile():
    torch._dynamo.config.disable = True
    yield
    torch._dynamo.config.disable = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_block_matrix(A: torch.Tensor) -> torch.Tensor:
    """Return Z = [[A, I], [0, 0]] as a 2n×2n matrix."""
    *batch, n, _ = A.shape
    eye  = torch.eye(n, dtype=A.dtype, device=A.device).expand(*batch, n, n)
    zero = torch.zeros_like(A)
    top  = torch.cat([A, eye],  dim=-1)
    bot  = torch.cat([zero, zero], dim=-1)
    return torch.cat([top, bot], dim=-2)


def _ref_expm_block(A: torch.Tensor):
    """Float64 matrix_exp on the 2n×2n block matrix — ground truth."""
    Z  = _build_block_matrix(A).double()
    EZ = torch.linalg.matrix_exp(Z).float()
    n  = A.shape[-1]
    return EZ[..., :n, :n], EZ[..., :n, n:]


def _t18_on_block(A: torch.Tensor):
    """expm_t18 applied to the full 2n×2n block matrix."""
    EZ = expm_t18(_build_block_matrix(A))
    n  = A.shape[-1]
    return EZ[..., :n, :n], EZ[..., :n, n:]


def _make_A(shape, scale=1.0, seed=0):
    torch.manual_seed(seed)
    return scale * torch.randn(*shape)


# ---------------------------------------------------------------------------
# Correctness vs float64 reference
# ---------------------------------------------------------------------------

class TestVsReference:
    """structured block impl vs torch.linalg.matrix_exp (float64)."""

    atol = 1e-4   # float32 T18 accuracy against float64 ground truth
    rtol = 1e-4

    @pytest.mark.parametrize("n,scale,seed", [
        (4,  0.5,  0),   # small norm
        (4,  1.0,  1),   # moderate norm
        (8,  0.5,  2),
        (16, 1.0,  3),
        (32, 1.0,  4),
    ])
    def test_E_small_norm(self, n, scale, seed):
        A = _make_A((n, n), scale=scale, seed=seed)
        E_blk, _   = expm_t18_augmented_sparse(A)
        E_ref, _   = _ref_expm_block(A)
        assert torch.allclose(E_blk, E_ref, atol=self.atol, rtol=self.rtol), \
            f"E max-diff={( E_blk - E_ref).abs().max():.2e}"

    @pytest.mark.parametrize("n,scale,seed", [
        (4,  0.5,  0),
        (4,  1.0,  1),
        (8,  0.5,  2),
        (16, 1.0,  3),
        (32, 1.0,  4),
    ])
    def test_psi_small_norm(self, n, scale, seed):
        A = _make_A((n, n), scale=scale, seed=seed)
        _, psi_blk = expm_t18_augmented_sparse(A)
        _, psi_ref = _ref_expm_block(A)
        assert torch.allclose(psi_blk, psi_ref, atol=self.atol, rtol=self.rtol), \
            f"psi max-diff={(psi_blk - psi_ref).abs().max():.2e}"

    @pytest.mark.parametrize("n,scale,seed", [
        (4,  5.0,  10),   # large norm → triggers scaling
        (8,  8.0,  11),
        (16, 4.0,  12),
    ])
    def test_E_large_norm(self, n, scale, seed):
        A = _make_A((n, n), scale=scale, seed=seed)
        E_blk, _   = expm_t18_augmented_sparse(A)
        E_ref, _   = _ref_expm_block(A)
        assert torch.allclose(E_blk, E_ref, atol=self.atol, rtol=self.rtol), \
            f"E max-diff={( E_blk - E_ref).abs().max():.2e}"

    @pytest.mark.parametrize("n,scale,seed", [
        (4,  5.0,  10),
        (8,  8.0,  11),
        (16, 4.0,  12),
    ])
    def test_psi_large_norm(self, n, scale, seed):
        A = _make_A((n, n), scale=scale, seed=seed)
        _, psi_blk = expm_t18_augmented_sparse(A)
        _, psi_ref = _ref_expm_block(A)
        assert torch.allclose(psi_blk, psi_ref, atol=self.atol, rtol=self.rtol), \
            f"psi max-diff={(psi_blk - psi_ref).abs().max():.2e}"

    @pytest.mark.parametrize("batch,n,scale,seed", [
        ((2,),  8,  1.0, 20),
        ((3,),  4,  2.0, 21),
        ((2, 3), 4, 1.0, 22),
    ])
    def test_batched(self, batch, n, scale, seed):
        A = _make_A((*batch, n, n), scale=scale, seed=seed)
        E_blk, psi_blk = expm_t18_augmented_sparse(A)
        E_ref,  psi_ref = _ref_expm_block(A)
        assert torch.allclose(E_blk,   E_ref,   atol=self.atol, rtol=self.rtol), \
            f"E   max-diff={(E_blk   - E_ref  ).abs().max():.2e}"
        assert torch.allclose(psi_blk, psi_ref, atol=self.atol, rtol=self.rtol), \
            f"psi max-diff={(psi_blk - psi_ref).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Consistency vs expm_t18 on the full 2n×2n block matrix
# ---------------------------------------------------------------------------

class TestVsT18Block:
    """structured block impl should match expm_t18 on the 2n×2n matrix closely
    (both are the same polynomial; differences are only floating-point reordering)."""

    atol = 1e-5
    rtol = 1e-5

    @pytest.mark.parametrize("n,scale,seed", [
        (4,  0.5,  0),
        (4,  1.0,  1),
        (8,  1.0,  2),
        (16, 1.0,  3),
        (4,  5.0,  4),   # large norm
    ])
    def test_E_matches_t18_block(self, n, scale, seed):
        A = _make_A((n, n), scale=scale, seed=seed)
        E_blk, _   = expm_t18_augmented_sparse(A)
        E_t18, _   = _t18_on_block(A)
        assert torch.allclose(E_blk, E_t18, atol=self.atol, rtol=self.rtol), \
            f"E max-diff={(E_blk - E_t18).abs().max():.2e}"

    @pytest.mark.parametrize("n,scale,seed", [
        (4,  0.5,  0),
        (4,  1.0,  1),
        (8,  1.0,  2),
        (16, 1.0,  3),
        (4,  5.0,  4),
    ])
    def test_psi_matches_t18_block(self, n, scale, seed):
        A = _make_A((n, n), scale=scale, seed=seed)
        _, psi_blk = expm_t18_augmented_sparse(A)
        _, psi_t18 = _t18_on_block(A)
        assert torch.allclose(psi_blk, psi_t18, atol=self.atol, rtol=self.rtol), \
            f"psi max-diff={(psi_blk - psi_t18).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

class TestSanity:
    def test_E_equals_matrix_exp(self):
        """E must equal matrix_exp(A) directly."""
        torch.manual_seed(99)
        A = torch.randn(16, 16)
        E_blk, _ = expm_t18_augmented_sparse(A)
        E_ref     = torch.linalg.matrix_exp(A.double()).float()
        assert torch.allclose(E_blk, E_ref, atol=1e-4, rtol=1e-4), \
            f"max-diff={(E_blk - E_ref).abs().max():.2e}"

    def test_psi_identity_input(self):
        """phi_1(0) = I (exp of zero matrix gives identity, phi_1(0) = I)."""
        n = 8
        A = torch.zeros(n, n)
        _, psi = expm_t18_augmented_sparse(A)
        assert torch.allclose(psi, torch.eye(n), atol=1e-5), \
            f"phi_1(0) should be I, max-diff={(psi - torch.eye(n)).abs().max():.2e}"

    def test_output_shapes(self):
        n = 12
        A = torch.randn(n, n)
        E, psi = expm_t18_augmented_sparse(A)
        assert E.shape   == (n, n)
        assert psi.shape == (n, n)

