# Constant-forcing variant of CGHC.
#
# Exact integration of dx/dt = Ax + f over one step gives:
#   x(1) = exp(A) x(0) + φ₁(A) f
# where φ₁(A) = ∫₀¹ exp(θA) dθ.
# Both exp(A) and φ₁(A) are computed together without materializing the 2n×2n
# augmented system (see expm_t18_augmented_sparse in ops).
#
# Weight decay note: this class adds no parameters of its own — the exclusions
# listed at the top of cghc.py apply unchanged (read_in, write_out, conserv_A,
# diss_A, laplacian_A, and all 1-D params). The inherited
# split_decay_param_groups(model, wd) classmethod builds the right groups.

from typing import Literal

import torch
from einops import einsum

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.ops import HAS_TRITON, expm_t18_augmented_sparse, expm_t18_block_triton


class ContinuousGenHyperConnectionsForced(ContinuousGenHyperConnections):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: torch.nn.Module,
        dt: float = 0.01,
        generator_type: Literal[
            "conservative",
            "psd_diss",
            "diagonal_diss",
            "laplacian",
            "conservative_diag_diss",
            "conservative_psd_diss",
            "conservative_laplacian",
        ] = "conservative_psd_diss",
        projection: Literal["mean", "v", "none"] = "none",
        learn_dt: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 1.0,
        bias: bool = False,
        elementwise_affine: bool = False,
        use_triton: bool = True,
        vec_dt: bool = False,
        shortconv_kernel_size: int = 0,
        shortconv_causal: bool = True,
    ):
        super().__init__(
            n=n,
            m=m,
            input_dim=input_dim,
            embed_dim=embed_dim,
            module=module,
            dt=dt,
            generator_type=generator_type,
            projection=projection,
            learn_dt=learn_dt,
            dt_min=dt_min,
            dt_max=dt_max,
            bias=bias,
            elementwise_affine=elementwise_affine,
            use_triton=use_triton,
            vec_dt=vec_dt,
            shortconv_kernel_size=shortconv_kernel_size,
            shortconv_causal=shortconv_causal,
        )
        ### Custom-kernel path for the forced exponential (exp(A), φ₁(A)), gated on the
        ### same use_triton flag as the base class's _matrix_exp / _stream_mix dispatch.
        self._use_block_triton = bool(use_triton and HAS_TRITON)

    # Manual graph break: an outer torch.compile treats the forced exp unit (Triton kernel
    # or max-autotune eager) as opaque instead of tracing into it.
    @torch.compiler.disable(recursive=False)
    def _expm_block(self, A: torch.Tensor):
        """Return (exp(A), φ₁(A)) via the Triton forced kernel when available, else eager."""
        if self._use_block_triton:
            return expm_t18_block_triton(A)
        return expm_t18_augmented_sparse(A)

    def compute_transition_expm_t18(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (exp(A), φ₁(A)), both of shape [B, n, n].

        Args:
            x_norm: Normalized input of shape [B, input_dim]
        """
        A = self.compute_generator(x_norm)
        transition_matrix, psi = self._expm_block(A.float())
        return transition_matrix.to(x_norm.dtype), psi.to(x_norm.dtype)

    def compute_transition(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Return exp(A) only, shape [B, n, n]. Delegates to compute_transition_expm_t18."""
        transition, _ = self.compute_transition_expm_t18(x_norm)
        return transition

    def _transition_and_source(
        self, x_norm: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition, psi = self.compute_transition_expm_t18(x_norm)
        return transition, einsum(psi, Y, "b n1 n2, b n2 d -> b n1 d")
