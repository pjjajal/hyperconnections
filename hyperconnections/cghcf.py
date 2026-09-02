import torch
from einops import einsum

from hyperconnections.cghc import CGHCProjections, ContinuousGenHyperConnections
from hyperconnections.ops import expm_t18_augmented_sparse, expm_t18_block_triton


class ContinuousGenHyperConnectionsForced(ContinuousGenHyperConnections):
    """CGHC with exact integration of a constant source over each step."""

    def compute_transition_and_psi(
        self, projections: CGHCProjections
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (exp(A), φ₁(A)), both of shape [B, n, n].

        Args:
            projections: Fused input projection (from self.input_proj).
        """
        A = self.compute_generator(projections).float()
        if self._use_triton:
            transition_matrix, psi = expm_t18_block_triton(A)
        else:
            transition_matrix, psi = expm_t18_augmented_sparse(A)
        dtype = projections.read_in.dtype
        return transition_matrix.to(dtype), psi.to(dtype)

    def compute_transition(self, projections: CGHCProjections) -> torch.Tensor:
        """Return exp(A), delegating to compute_transition_and_psi."""
        transition, _ = self.compute_transition_and_psi(projections)
        return transition

    def _transition_and_source(
        self, projections: CGHCProjections, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition, psi = self.compute_transition_and_psi(projections)
        source = einsum(psi, source, "b n1 n2, b n2 d -> b n1 d")
        return transition, source
