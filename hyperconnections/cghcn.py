# Nilpotent-chain variant of CGHC.
#
# The generator is a weighted shift: learned constants on the first lower
# subdiagonal,
#   N = subdiag(w),  w in R^{n-1},  N^n = 0,
# scaled by the inherited dt machinery (log-space learned dt + dt_proj
# conditioning; the vec_dt sandwich is a diagonal congruence, which preserves
# strict lower-triangularity and hence exact nilpotency). Because A = dt·N is
# nilpotent, exp(A) is the exact finite polynomial
#   exp(A) = I + A + A²/2! + ... + A^{n-1}/(n-1)!,
# so no general matrix exponential (expm_t18) is needed — compute_transition
# evaluates the polynomial directly. Stream i receives x scaled by the length-i
# path product Δ^i/i! · w_i···w_1: an ordered predictor/corrector chain.
#
# Weight decay note: this class adds only `chain_w` (1-D, caught by the
# ndim <= 1 rule in split_decay_param_groups); the exclusions listed at the top
# of cghc.py otherwise apply unchanged. The unused parent generator params
# (diss_*) and the dissipative dt params are deleted at construction so they
# never appear in the state dict, the optimizer, or DDP's unused-param checks.

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from hyperconnections.cghc import ContinuousGenHyperConnections


class ContinuousGenHyperConnectionsNilpotent(ContinuousGenHyperConnections):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        dt: float = 0.01,
        projection: str = "none",
        learn_dt: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 1.0,
        bias: bool = False,
        elementwise_affine: bool = False,
        use_triton: bool = True,
        vec_dt: bool = False,
        shortconv_kernel_size: int = 0,
        shortconv_causal: bool = True,
        chain_init: float = 1.0,
    ):
        assert n >= 2, f"nilpotent chain needs at least 2 streams, got n={n}"
        self.chain_init = chain_init
        super().__init__(
            n=n,
            m=m,
            input_dim=input_dim,
            embed_dim=embed_dim,
            module=module,
            dt=dt,
            # Smallest parent generator; its params are deleted just below.
            generator_type="diagonal_diss",
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
        self.generator_type = "nilpotent_chain"
        # Drop the parent's generator and dissipative-dt params: they take no
        # part in the nilpotent generator, and dead requires_grad params would
        # trip DDP's unused-parameter check.
        del self.diss_diag
        del self.diss_log_scale
        del self.diss_pred
        del self.log_dt_diss
        del self.dt_proj_diss

        # The chain edge weights (semantic anchor: dt·chain_init sets the
        # initial one-step transfer strength; the note's ρ ∈ [0.05, 0.2] regime
        # corresponds to chain_init=1.0 at dt=0.1).
        self.chain_w = nn.Parameter(torch.empty(n - 1))
        self._init_chain_w()

    def _init_chain_w(self) -> None:
        nn.init.constant_(self.chain_w, self.chain_init)
        with torch.no_grad():
            # small noise for asymmetry breaking
            noise = torch.empty_like(self.chain_w)
            trunc_normal_(noise, std=0.01)
            self.chain_w.add_(noise)

    def init_weights(self):
        # Called once inside super().__init__() (before chain_w exists) and
        # again post-construction by the LM muP init sweep — restore chain_w
        # only on the latter.
        super().init_weights()
        if hasattr(self, "chain_w"):
            self._init_chain_w()

    def _compute_generator(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Return A = dt-scaled subdiag(chain_w), shape [B, n, n].

        Scalar dt:  A = dt(x) * subdiag(w).
        vec_dt:     A = D^{1/2} subdiag(w) D^{1/2}, D = diag(dt(x)) — the edge
        i→i+1 is scaled by sqrt(dt_{i+1} dt_i). Either way A stays strictly
        lower-bidiagonal, so A^n = 0 exactly for every input.
        """
        logit = self.log_dt_conserv + self.dt_proj_conserv(x_norm)  # [B, n_dt]
        dt = torch.exp(
            self.log_dt_min + self.log_dt_range * torch.sigmoid(logit)
        )
        if not self.vec_dt:
            w_dt = dt * self.chain_w  # [B, 1] * [n-1] -> [B, n-1]
        else:
            sqrt_dt = dt.sqrt()  # [B, n]
            w_dt = sqrt_dt[:, 1:] * self.chain_w * sqrt_dt[:, :-1]  # [B, n-1]
        return torch.diag_embed(w_dt, offset=-1)  # [B, n, n]

    def _nilpotent_expm(self, A: torch.Tensor) -> torch.Tensor:
        """Exact exp(A) for nilpotent A: the Taylor series terminates at A^{n-1}."""
        E = torch.eye(self.n, device=A.device, dtype=A.dtype).expand_as(A)
        term = E
        for k in range(1, self.n):
            term = A @ term / k
            E = E + term
        return E

    def compute_transition(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Return Phi = exp(A) via the exact terminating polynomial, [B, n, n].

        Args:
            x_norm: Normalized input of shape [B, input_dim]
        """
        A = self.compute_generator(x_norm)  # fp32
        with torch.autocast(device_type=A.device.type, enabled=False):
            Phi = self._nilpotent_expm(A)
        return Phi.to(x_norm.dtype)
