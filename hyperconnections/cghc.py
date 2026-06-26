import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from timm.layers import trunc_normal_

from hyperconnections.ops import HAS_TRITON, expm_t18, expm_t18_triton, stream_mix_add
from hyperconnections.short_conv import DepthwiseShortConv1d


class ContinuousGenHyperConnections(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
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
        use_expm_t18: bool = False,
        shortconv_kernel_size: int = 0,
        shortconv_causal: bool = True,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.generator_type = generator_type
        self.projection = projection

        assert embed_dim % m == 0, (
            f"embed_dim ({embed_dim}) must be divisible by m ({m})"
        )
        assert input_dim == int((n / m) * embed_dim), (
            f"input_dim must be (n/m)*embed_dim, got {input_dim} vs {int((n / m) * embed_dim)}"
        )

        self.block_size = embed_dim // m

        # Read/write parameters following mHC convention
        self.read_in = nn.Parameter(torch.empty(n, m))
        self.alpha_read_in = nn.Parameter(torch.empty(1))
        self.write_out = nn.Parameter(torch.empty(n, m))
        self.alpha_write_out = nn.Parameter(torch.empty(1))

        self.proj_read_in = nn.Linear(input_dim, n * m, bias=bias)
        self.proj_write_out = nn.Linear(input_dim, n * m, bias=bias)

        # dt parameters
        assert dt_min < dt < dt_max, (
            f"Initial dt ({dt}) must lie strictly in (dt_min, dt_max) = ({dt_min}, {dt_max})"
        )
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.log_dt_init = math.log(dt)
        self.vec_dt = vec_dt
        n_dt = n if vec_dt else 1
        self.log_dt_conserv = nn.Parameter(torch.empty(n_dt), requires_grad=learn_dt)
        self.log_dt_diss = nn.Parameter(torch.empty(n_dt), requires_grad=learn_dt)
        self.dt_proj_conserv = nn.Linear(input_dim, n_dt, bias=True)
        self.dt_proj_diss = nn.Linear(input_dim, n_dt, bias=True)

        # Generator parameters — boolean flags drive which components are created
        conserv = generator_type in {
            "conservative",
            "conservative_diag_diss",
            "conservative_psd_diss",
            "conservative_laplacian",
        }
        psd_diss = generator_type in {"psd_diss", "conservative_psd_diss"}
        diag_diss = generator_type in {"diagonal_diss", "conservative_diag_diss"}
        laplacian = generator_type in {"laplacian", "conservative_laplacian"}

        if conserv:
            self.conserv_A = nn.Parameter(torch.eye(n, n))
            self.conserv_pred = nn.Linear(input_dim, n * n, bias=False)
        if psd_diss:
            self.diss_A = nn.Parameter(torch.zeros(n, n))
            self.diss_pred = nn.Linear(input_dim, n * n, bias=False)
        if diag_diss:
            self.diss_diag = nn.Parameter(torch.full((n,), -8.0))
            self.diss_pred = nn.Linear(input_dim, n, bias=False)
        if laplacian:
            self.laplacian_A = nn.Parameter(torch.zeros(n, n))
            self.laplacian_pred = nn.Linear(input_dim, n * n, bias=False)

        # Projection Direction
        if projection == "mean":
            self.register_buffer("projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "v":
            self.register_buffer("base_projection_dir", torch.ones(n) / math.sqrt(n))
            self.projection_dir = nn.Linear(input_dim, n, bias=False)
        elif projection == "none":
            self.projection_dir = None

        self.norm = nn.RMSNorm(input_dim, elementwise_affine=elementwise_affine)
        self.module = module
        # Optional over-width short conv on the read/source path (see forward).
        self.short_conv = (
            DepthwiseShortConv1d(input_dim, shortconv_kernel_size, causal=shortconv_causal)
            if shortconv_kernel_size > 0
            else None
        )
        self._stream_mix = (
            self._stream_mix_triton
            if use_triton and HAS_TRITON
            else self._stream_mix_eager
        )
        self._matrix_exp = (
            expm_t18_triton if use_triton and HAS_TRITON else self._matrix_exp_eager
        )
        self.init_weights()

    def _init_log_dt(self, param: nn.Parameter) -> None:
        target = (math.exp(self.log_dt_init) - self.dt_min) / (self.dt_max - self.dt_min)
        target = min(max(target, 1e-4), 1 - 1e-4)
        nn.init.constant_(param, math.log(target / (1 - target)))

    def init_weights(self):
        # read_in (semantic [m, n], stored as [n, m]):
        #   m == 1 (mHC case): preserve mean-read convention — σ(log(1/(n-1))) = 1/n
        #     so each of n streams contributes 1/n to the single chunk's read.
        #   m > 1 (GHC case): top-left m×m identity. Chunk j reads predominantly from
        #     stream j; streams [m, n) start dormant. σ(+5) ≈ 0.993 / σ(-5) ≈ 0.007
        #     saturates GHC's 1/0 values.
        if self.m == 1:
            logit_1_over_n = math.log(1.0 / (self.n - 1)) if self.n > 1 else 10.0
            nn.init.constant_(self.read_in, logit_1_over_n)
        else:
            read_in_init = torch.full((self.m, self.n), -5.0)
            for j in range(self.m):
                read_in_init[j, j] = 5.0
            self.read_in.data.copy_(read_in_init.T)

        # write_out (semantic [n, m]): round-robin, matching GHC. Stream i writes
        # predominantly from chunk (i % m). 2·σ(0) = 1 exactly; 2·σ(-5) ≈ 0.013
        # for off-positions. For m == 1 every entry collapses to 0, recovering
        # the original 2·σ(0) = 1 mHC convention.
        write_out_init = torch.full((self.n, self.m), -5.0)
        for i in range(self.n):
            write_out_init[i, i % self.m] = 0.0
        self.write_out.data.copy_(write_out_init)

        with torch.no_grad():
            # small noise for asymmetry breaking
            self.read_in.add_(torch.randn_like(self.read_in) * 0.01)
            self.write_out.add_(torch.randn_like(self.write_out) * 0.01)
        # Alpha gating: 0.01 so dynamic component starts negligible
        nn.init.constant_(self.alpha_read_in, 0.01)
        nn.init.constant_(self.alpha_write_out, 0.01)

        # Generator Static Parameters
        if hasattr(self, "conserv_A"):
            nn.init.eye_(self.conserv_A)
            # Small asymmetry so skew-sym part is non-zero at init
            with torch.no_grad():
                noise = torch.empty_like(self.conserv_A)
                trunc_normal_(noise, std=0.01)
                self.conserv_A.add_(noise)

        if hasattr(self, "diss_A"):
            trunc_normal_(self.diss_A, std=0.01)

        if hasattr(self, "diss_diag"):
            nn.init.constant_(self.diss_diag, -8.0)

        if hasattr(self, "laplacian_A"):
            nn.init.constant_(self.laplacian_A, -8.0)

        # Generator Dynamic Parameters
        if hasattr(self, "conserv_pred"):
            nn.init.zeros_(self.conserv_pred.weight)

        if hasattr(self, "diss_pred"):
            nn.init.zeros_(self.diss_pred.weight)

        if hasattr(self, "laplacian_pred"):
            nn.init.zeros_(self.laplacian_pred.weight)

        self._init_log_dt(self.log_dt_conserv)
        self._init_log_dt(self.log_dt_diss)

        # dt_proj: small random init for weights, zero bias for centered initial dt with input-dependent variation
        trunc_normal_(self.dt_proj_conserv.weight, std=0.01)
        nn.init.zeros_(self.dt_proj_conserv.bias)
        trunc_normal_(self.dt_proj_diss.weight, std=0.01)
        nn.init.zeros_(self.dt_proj_diss.bias)

        # Projections: small random init for weights, zero bias so initial mean behaviour matches static biases
        for proj in (self.proj_read_in, self.proj_write_out):
            trunc_normal_(proj.weight, std=0.01)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        # mean projection: set to mean direction.
        # small noise for asymmetry breaking so projection isn't exactly static at init, but normalised to keep initial scale consistent.
        if self.projection == "mean":
            self.projection_dir.fill_(1.0 / math.sqrt(self.n))
            with torch.no_grad():
                self.projection_dir.add_(torch.randn_like(self.projection_dir) * 0.01)
                self.projection_dir.div_(self.projection_dir.norm())
        # proj_v: zero init for weights to start at base_projection_dir with input-dependent variation
        if self.projection == "v":
            self.base_projection_dir.fill_(1.0 / math.sqrt(self.n))
            nn.init.zeros_(self.projection_dir.weight)

        # RMSNorm weights: must be ones for proper normalization
        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)

    def compute_generator(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Return the effective generator A of shape [B, n, n].

        Args:
            x_norm: Normalized input of shape [B, input_dim]

        When vec_dt=True, each stream has its own time scale and the generator is
        built via a symmetric congruence sandwich:

            A = D_S^{1/2} (S) D_S^{1/2}  -  D_Q^{1/2} (Q) D_Q^{1/2}

        where D_S = diag(dt_conserv), D_Q = diag(dt_diss), each with shape [B, n]
        and entries in (dt_min, dt_max). The sandwich preserves skew-symmetry of S
        and PSD-ness of Q, so the Lyapunov stability argument carries through.
        For the diagonal dissipation case, D_Q^{1/2} diag(d) D_Q^{1/2} = diag(dt_diss * d).

        When vec_dt=False, dt_conserv and dt_diss are scalars (shape [B, 1]) shared
        across all streams, reducing the sandwich to a simple scalar scaling:

            A = dt_conserv * S  -  dt_diss * Q
        """
        B = x_norm.shape[0]
        A = torch.zeros(B, self.n, self.n, device=x_norm.device, dtype=x_norm.dtype)

        # --- Conservative branch ---
        if hasattr(self, "conserv_A"):
            M = self.conserv_A + self.conserv_pred(x_norm).reshape(B, self.n, self.n)
            logit_conserv = self.log_dt_conserv + self.dt_proj_conserv(x_norm)  # [B, n]
            dt_conserv = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(logit_conserv)
            skew = 0.5 * (M - M.transpose(-1, -2))  # [B, n, n], skew-symmetric
            if not self.vec_dt:
                # Scalar dt: equivalent to the sandwich but avoids unnecessary sqrt
                skew_dt = dt_conserv.unsqueeze(-1) * skew
            else:
                # Per-stream sandwich: (D^{1/2} skew D^{1/2})_{ij} = sqrt_dt_i * skew_{ij} * sqrt_dt_j
                sqrt_dt_conserv = dt_conserv.sqrt()  # [B, n]
                skew_dt = (
                    sqrt_dt_conserv[:, :, None] * skew * sqrt_dt_conserv[:, None, :]
                )

            A = A + skew_dt

        # --- Shared dissipative dt ---
        if (
            hasattr(self, "diss_A")
            or hasattr(self, "diss_diag")
            or hasattr(self, "laplacian_A")
        ):
            logit_diss = self.log_dt_diss + self.dt_proj_diss(x_norm)  # [B, n]
            dt_diss = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(logit_diss)
            sqrt_dt_diss = dt_diss.sqrt()  # [B, n]

        # --- PSD dissipative (Gram matrix) branch ---
        if hasattr(self, "diss_A"):
            R = self.diss_A + self.diss_pred(x_norm).reshape(B, self.n, self.n)
            K = R @ R.transpose(-1, -2) / (self.n**0.5)  # [B, n, n], PSD
            if not self.vec_dt:
                # Scalar dt: equivalent to the sandwich but avoids unnecessary sqrt
                diss_dt = dt_diss.unsqueeze(-1) * K
            else:
                # Per-stream sandwich: (D^{1/2} K D^{1/2})_{ij} = sqrt_dt_i * K_{ij} * sqrt_dt_j
                diss_dt = sqrt_dt_diss[:, :, None] * K * sqrt_dt_diss[:, None, :]
            A = A - diss_dt

        # --- Diagonal dissipative branch ---
        if hasattr(self, "diss_diag"):
            d = F.softplus(self.diss_diag + self.diss_pred(x_norm))  # [B, n], positive
            # Sandwich of a diagonal reduces to elementwise product: diag(sqrt_dt * d * sqrt_dt)
            # = diag(dt_diss * d)
            A = A - torch.diag_embed(
                dt_diss * d
            )  # dt_diss [B,1] * d [B,n] broadcasts correctly

        # --- Laplacian dissipative branch ---
        if hasattr(self, "laplacian_A"):
            scores = self.laplacian_A + self.laplacian_pred(x_norm).reshape(B, self.n, self.n)
            scores = 0.5 * (scores + scores.transpose(-1, -2))  # symmetrize
            adjacency = F.softplus(scores)
            adjacency = adjacency - torch.diag_embed(
                torch.diagonal(adjacency, dim1=-2, dim2=-1)
            )
            degree = torch.diag_embed(adjacency.sum(dim=-1))
            laplacian = degree - adjacency  # PSD
            if not self.vec_dt:
                # Scalar dt: equivalent to the sandwich but avoids unnecessary sqrt
                laplacian_dt = dt_diss.unsqueeze(-1) * laplacian
            else:
                laplacian_dt = (
                    sqrt_dt_diss[:, :, None] * laplacian * sqrt_dt_diss[:, None, :]
                )
            A = A - laplacian_dt

        return A

    def compute_transition(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Return Phi = exp(dt * A), shape [B, n, n].

        Args:
            x_norm: Normalized input of shape [B, input_dim]
        """
        A = self.compute_generator(x_norm)
        return self._matrix_exp(A)

    def _expm_t18(self, A: torch.Tensor) -> torch.Tensor:
        """Compute matrix exponential using expm_t18 approximation."""
        return expm_t18(A)

    def _matrix_exp_eager(self, A: torch.Tensor) -> torch.Tensor:
        return self._expm_t18(A.float()).to(A.dtype)

    def compute_read_write_weights(self, x_norm: torch.Tensor):
        """Compute dynamic read/write weights from the current stream state.

        Args:
            x_norm: Normalized input of shape [B, input_dim]
        """
        B = x_norm.shape[0]

        h_read_in = self.proj_read_in(x_norm).reshape(B, self.n, self.m)
        h_write_out = self.proj_write_out(x_norm).reshape(B, self.n, self.m)

        read_in = torch.sigmoid(
            self.alpha_read_in * h_read_in + self.read_in
        ).transpose(1, 2)  # [B, m, n]
        write_out = 2 * torch.sigmoid(
            self.alpha_write_out * h_write_out + self.write_out
        )  # [B, n, m]

        return write_out, read_in

    def compute_projection(self, x_norm: torch.Tensor):
        """Compute projection direction.

        Args:
            x_norm: Normalized input of shape [B, input_dim]
        """
        if self.projection == "mean":
            return self.projection_dir.unsqueeze(0)  # [1, n]
        elif self.projection == "v":
            v = self.projection_dir(x_norm) + self.base_projection_dir  # [B, n]
            return F.normalize(v, dim=-1)  # [B, n], unit norm
        else:
            return None

    def _stream_mix_triton(
        self,
        x: torch.Tensor,
        transition_matrix: torch.Tensor,
        Y: torch.Tensor,
        projection_dir: torch.Tensor | None,
    ) -> torch.Tensor:
        if projection_dir is not None:
            projection_dir = projection_dir.expand(
                x.shape[0], -1
            )  ### [1, N] ("mean" mode) --> [B, N]
        return stream_mix_add(transition_matrix, x, Y, projection_dir)

    def _stream_mix_eager(
        self,
        x: torch.Tensor,
        transition_matrix: torch.Tensor,
        Y: torch.Tensor,
        projection_dir: torch.Tensor | None,
    ) -> torch.Tensor:
        if projection_dir is None:
            x_mixed = einsum(
                transition_matrix, x, "b n1 n2, b n2 d -> b n1 d"
            )  # [B*, n, block_size]
        else:
            x_proj = einsum(
                projection_dir,
                projection_dir,
                x,
                "b n1, b n2, b n2 d -> b n1 d",
            )  # [b, n, block_size]
            x_orth = x - x_proj
            x_mixed = x_proj + einsum(
                transition_matrix, x_orth, "b n1 n2, b n2 d -> b n1 d"
            )  # [B*, n, block_size]
        return x_mixed + Y

    def _transition_and_source(
        self, x_norm: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (transition_matrix, Y) for stream mixing.

        Subclasses override this to apply φ₁(A) to Y (forced/exact integration).
        """
        return self.compute_transition(x_norm), Y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, *, input_dim]  (any number of leading dims, last dim = n * block_size)
        Returns:
            [B, *, input_dim]
        """
        ### x: [B, *, input_dim]
        leading = x.shape[:-1]
        x = x.reshape(-1, self.n, self.block_size)  ### [B*, n, block_size]
        B = x.shape[0]
        ### Optional over-width short conv on the read/source path only: it feeds the
        ### read-in and every x_norm-derived weight (read/write/generator/projection),
        ### while the carried stream `x` that gets stream-mixed stays un-convolved.
        if self.short_conv is not None:
            src = self.short_conv(x.reshape(*leading, self.input_dim)).reshape(B, self.n, self.block_size)
        else:
            src = x
        x_norm = self.norm(src.view(B, -1))  ### [B*, input_dim]

        write_out, read_in = self.compute_read_write_weights(x_norm)

        ### Source term Y = H^post F(H^pre X)  (read → compute → write)
        ### Read in from over-width space to backbone width
        x_read = einsum(read_in, src, "b m n, b n d -> b m d")  ### [B*, m, block_size]

        ### Process through the backbone module
        out = self.module(x_read.reshape(*leading, self.embed_dim), **kwargs)

        ### Write out from backbone width back to the over-width space
        out = out.reshape(B, self.m, self.block_size)  ### [B*, m, block_size]
        Y = einsum(write_out, out, "b n m, b m d -> b n d")  ### [B*, n, block_size]

        ### Compute transition matrix and (optionally) apply φ₁ to Y
        transition_matrix, Y = self._transition_and_source(x_norm, Y)

        projection_dir = self.compute_projection(x_norm)  ### [B, n] or None

        return (
            self._stream_mix(
                x=x,
                transition_matrix=transition_matrix,
                Y=Y,
                projection_dir=projection_dir,
            )
            .unflatten(0, leading)
            .flatten(-2)
        )
