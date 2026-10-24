import math

import torch
from einops import einsum

from hyperconnections.cghc import CGHCProjections, ContinuousGenHyperConnections
from hyperconnections.ops import expm_t18_augmented_sparse, expm_t18_block_triton


def _flash_transition_and_psi(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(exp(A), phi_1(A)) from one exponential of the augmented [[A, I], [0, 0]] (2n x 2n)."""
    import flash_hypercons

    n = A.shape[-1]
    augmented = A.new_zeros(*A.shape[:-2], 2 * n, 2 * n)
    augmented[..., :n, :n] = A
    augmented[..., :n, n:] = torch.eye(n, device=A.device, dtype=A.dtype)
    block = flash_hypercons.matrix_exp(augmented)
    return block[..., :n, :n], block[..., :n, n:]


class ContinuousGenHyperConnectionsForced(ContinuousGenHyperConnections):
    """CGHC with exact integration of a constant source over each step."""

    def compute_transition_and_psi(
        self, projections: CGHCProjections
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (exp(A), φ₁(A)), both of shape [B, num_mixing_streams, num_mixing_streams].

        Args:
            projections: Fused input projection (from self.input_proj).
        """
        A = self.compute_generator(projections).float()
        if self.expm_backend == "flash":
            transition_matrix, psi = _flash_transition_and_psi(A)
        elif self.expm_backend == "triton":
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
        psi = self._projected_psi(psi, projections)
        source = einsum(psi, source, "b n1 n2, b n2 d -> b n1 d")
        return transition, source

    def _projected_psi(self, psi: torch.Tensor, projections: CGHCProjections) -> torch.Tensor:
        """Constant-forcing operator for the projected transitions.

        The projected variants replace T = exp(A) by d P + (gamma P + Q) T Q (gamma = 0
        two-sided, 1 one-sided, input-dependent for inflow). That is the exact flow of a split
        system: the v part decays as s' = λ s with d = e^λ, the v-perp part evolves under the
        full A, and (gamma P + Q) is read out at the end. Splitting the source the same way and
        integrating it exactly gives φ₁(λ) P + gamma P φ₁(A) Q + Q φ₁(A) Q.
        """
        if self.projection_variant == "generator":
            return psi  # φ₁(QAQ) is already the exact operator for exp(QAQ).
        projection_dir = self.compute_projection(projections)
        if projection_dir is None:
            return psi
        gamma = {"one_sided": 1.0, "two_sided": 0.0}.get(self.projection_variant)
        if gamma is None:
            gamma = self.compute_inflow(projections)
        # The v channel decays by d = e^λ, so its constant-forcing gain is φ₁(λ) = (d - 1) / ln d.
        d = self.projection_decay
        source_decay = 1.0 if d == 1.0 else 0.0 if d == 0.0 else (d - 1.0) / math.log(d)
        return self._inflow_transition(psi, projection_dir, gamma, decay=source_decay)
