import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from timm.models.layers import trunc_normal_

from hyperconnections.ops import stream_mix_add, expm_t18


class ContinuousGenHyperConnections(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        dt: float = 1.0,
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
        vec_dt: bool = False
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
        self.log_dt_conserv = nn.Parameter(
            torch.empty(n_dt), requires_grad=learn_dt
        )
        self.log_dt_diss = nn.Parameter(
            torch.empty(n_dt), requires_grad=learn_dt
        )
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
            self.conv_pred = nn.Linear(input_dim, n * n, bias=True)
        if psd_diss:
            self.diss_A = nn.Parameter(torch.zeros(n, n))
            self.diss_pred = nn.Linear(input_dim, n * n, bias=True)
        if diag_diss:
            # Diagonal dissipation: store only the diagonal entries for efficiency
            # Initialised to -2.0 in init_weights, giving softplus(-2.0) ≈ 0.127
            self.diss_diag = nn.Parameter(torch.full((n,), -5.0, requires_grad=True))
            self.diss_pred = nn.Linear(input_dim, n, bias=True)
        if laplacian:
            self.laplacian_A = nn.Parameter(torch.zeros(n, n))
            self.laplacian_q = nn.Linear(self.block_size, self.block_size // 4, bias=True)
            self.laplacian_k = nn.Linear(self.block_size, self.block_size // 4, bias=True)
            self.laplacian_scale = (self.block_size // 4)**-0.5
            self.norm_lap = nn.RMSNorm(self.block_size, elementwise_affine=True)

        # Projection Direction
        if projection == "mean":
            self.register_buffer("projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "v":
            self.projection_dir = nn.Linear(input_dim, n, bias=True)
        elif projection == "none":
            self.projection_dir = None

        self.norm = nn.RMSNorm(input_dim, elementwise_affine=elementwise_affine)
        self.module = module
        self._stream_mix = (
            self._stream_mix_triton if use_triton else self._stream_mix_eager
        )
        self.init_weights()

    def init_weights(self):
        # read_in: σ(bias) = 1/n  →  bias = log(1/(n-1))
        logit_1_over_n = math.log(1.0 / (self.n - 1)) if self.n > 1 else 10.0
        nn.init.constant_(self.read_in, logit_1_over_n)
        with torch.no_grad():
            self.read_in.add_(torch.randn_like(self.read_in) * 0.01)  # small noise for asymmetry breaking
        # write_out: 2·σ(0) = 1
        trunc_normal_(self.write_out, std=0.01)
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
            nn.init.constant_(self.diss_diag, -2.0)

        if hasattr(self, "laplacian_A"):
            nn.init.zeros_(self.laplacian_A)

        # Generator Dynamic Parameters
        if hasattr(self, "conv_pred"):
            trunc_normal_(self.conv_pred.weight, std=0.01)
            nn.init.zeros_(self.conv_pred.bias)

        if hasattr(self, "diss_pred"):
            trunc_normal_(self.diss_pred.weight, std=0.01)
            nn.init.zeros_(self.diss_pred.bias)

        if hasattr(self, "laplacian_q"):
            trunc_normal_(self.laplacian_q.weight, std=0.01)
            nn.init.zeros_(self.laplacian_q.bias)
            trunc_normal_(self.laplacian_k.weight, std=0.01)
            nn.init.zeros_(self.laplacian_k.bias)

        # Initialize log_dt so that sigmoid(log_dt) * (dt_max - dt_min) + dt_min = dt_init.
        # log_dt has length n_dt (1 when vec_dt=False, n when vec_dt=True).
        dt_init = math.exp(self.log_dt_init)
        target = (dt_init - self.dt_min) / (self.dt_max - self.dt_min)
        target = min(max(target, 1e-4), 1 - 1e-4)  # guard against endpoint singularities
        bias_init = math.log(target / (1 - target))
        nn.init.constant_(self.log_dt_conserv, bias_init)
        nn.init.constant_(self.log_dt_diss, bias_init)

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
        # proj_v: small random weight + scaled ones bias → starts near mean direction with input-dependent variation
        if self.projection == "v":
            trunc_normal_(self.projection_dir.weight, std=0.01)
            nn.init.ones_(self.projection_dir.bias)
            self.projection_dir.bias.data /= math.sqrt(self.n)

        # RMSNorm weights: must be ones for proper normalization
        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)
        if (
            hasattr(self, "norm_lap")
            and hasattr(self.norm_lap, "weight")
            and self.norm_lap.weight is not None
        ):
            nn.init.ones_(self.norm_lap.weight)

    def compute_generator(self, x: torch.Tensor) -> torch.Tensor:
        """Return the effective generator A of shape [B, n, n].

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
        B = x.shape[0]
        if hasattr(self, "laplacian_A"):
            x_lap_norm = self.norm_lap(x)
        x_norm = self.norm(x.view(B, -1))  # [B, input_dim]
        A = torch.zeros(B, self.n, self.n, device=x.device, dtype=x.dtype)

        # --- Conservative branch ---
        if hasattr(self, "conserv_A"):
            M = self.conserv_A + self.conv_pred(x_norm).reshape(B, self.n, self.n)
            logit_conserv = self.log_dt_conserv + self.dt_proj_conserv(x_norm)  # [B, n]
            dt_conserv = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(logit_conserv)
            skew = M - M.transpose(-1, -2)  # [B, n, n], skew-symmetric
            if not self.vec_dt:
                # Scalar dt: equivalent to the sandwich but avoids unnecessary sqrt
                skew_dt = dt_conserv * skew
            else:
                # Per-stream sandwich: (D^{1/2} skew D^{1/2})_{ij} = sqrt_dt_i * skew_{ij} * sqrt_dt_j
                sqrt_dt_conserv = dt_conserv.sqrt()  # [B, n]
                skew_dt = sqrt_dt_conserv[:, :, None] * skew * sqrt_dt_conserv[:, None, :]

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
            K = R @ R.transpose(-1, -2)  # [B, n, n], PSD
            if not self.vec_dt:
                # Scalar dt: equivalent to the sandwich but avoids unnecessary sqrt
                diss_dt = dt_diss * K
            else:
                # Per-stream sandwich: (D^{1/2} K D^{1/2})_{ij} = sqrt_dt_i * K_{ij} * sqrt_dt_j
                diss_dt = sqrt_dt_diss[:, :, None] * K * sqrt_dt_diss[:, None, :]
            A = A - diss_dt

        # --- Diagonal dissipative branch ---
        if hasattr(self, "diss_diag"):
            d = F.softplus(self.diss_diag + self.diss_pred(x_norm))  # [B, n], positive
            # Sandwich of a diagonal reduces to elementwise product: diag(sqrt_dt * d * sqrt_dt)
            # = diag(dt_diss * d)
            A = A - torch.diag_embed(dt_diss * d)

        # --- Laplacian dissipative branch ---
        if hasattr(self, "laplacian_A"):
            score_bias = self.laplacian_A
            lap_q = self.laplacian_q(x_lap_norm)
            lap_k = self.laplacian_k(x_lap_norm)
            scores = lap_q @ lap_k.transpose(-1, -2) * self.laplacian_scale
            scores = score_bias + scores
            scores = 0.5 * (scores + scores.transpose(-1, -2))  # symmetrize
            adjacency = F.softplus(scores) - math.log(2)  # zero scores -> zero adjacency
            adjacency = adjacency - torch.diag_embed(
                torch.diagonal(adjacency, dim1=-2, dim2=-1)
            )
            degree = torch.diag_embed(adjacency.sum(dim=-1))
            laplacian = degree - adjacency  # PSD
            if not self.vec_dt:
                # Scalar dt: equivalent to the sandwich but avoids unnecessary sqrt
                laplacian_dt = dt_diss * laplacian
            else:
                laplacian_dt = sqrt_dt_diss[:, :, None] * laplacian * sqrt_dt_diss[:, None, :]
            A = A - laplacian_dt

        return A

    def compute_transition(self, x: torch.Tensor) -> torch.Tensor:
        """Return Phi = exp(dt * A), shape [B, n, n]."""
        A = self.compute_generator(x)
        # return expm_t18(A).to(x.dtype)
        return torch.linalg.matrix_exp(A)

    def compute_read_write_weights(self, x: torch.Tensor):
        """Compute dynamic read/write weights from the current stream state."""
        B = x.shape[0]
        x_flat = x.view(B, -1)  # [B, input_dim]
        x_norm = self.norm(x_flat)  # [B, input_dim]

        h_read_in = self.proj_read_in(x_norm).reshape(B, self.n, self.m)
        h_write_out = self.proj_write_out(x_norm).reshape(B, self.n, self.m)

        read_in = torch.sigmoid(
            self.alpha_read_in * h_read_in + self.read_in
        ).transpose(1, 2)  # [B, m, n]
        write_out = 2 * torch.sigmoid(
            self.alpha_write_out * h_write_out + self.write_out
        )  # [B, n, m]

        return write_out, read_in

    def compute_projection(self, x: torch.Tensor):
        if self.projection == "mean":
            return self.projection_dir.unsqueeze(0)  # [1, n]
        elif self.projection == "v":
            B = x.shape[0]
            x_flat = x.view(B, -1)
            v = self.projection_dir(self.norm(x_flat))  # [B, n]
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
            proj_matrix = einsum(
                projection_dir, projection_dir, "b n1, b n2 -> b n1 n2"
            )  # [b, n, n]
            orthogonal_proj = (
                torch.eye(self.n, device=x.device, dtype=x.dtype) - proj_matrix
            )  # [b, n, n]
            x_proj = einsum(
                proj_matrix, x, "b n1 n2, b n2 d -> b n1 d"
            )  # [b, n, block_size]
            x_orth = einsum(
                orthogonal_proj, x, "b n1 n2, b n2 d -> b n1 d"
            )  # [b, n, block_size]
            x_mixed = x_proj + einsum(
                transition_matrix, x_orth, "b n1 n2, b n2 d -> b n1 d"
            )  # [B*, n, block_size]
        return x_mixed + Y

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

        write_out, read_in = self.compute_read_write_weights(x)

        ### Source term Y = H^post F(H^pre X)  (read → compute → write)
        ### Read in from over-width space to backbone width
        x_read = einsum(read_in, x, "b m n, b n d -> b m d")  ### [B*, m, block_size]

        ### Process through the backbone module
        out = self.module(x_read.reshape(*leading, self.embed_dim), **kwargs)

        ### Write out from backbone width back to the over-width space
        out = out.reshape(B, self.m, self.block_size)  ### [B*, m, block_size]
        Y = einsum(write_out, out, "b n m, b m d -> b n d")  ### [B*, n, block_size]

        ### Steam Mixing
        ### Mixing: X_new_mix = Phi @ X  (or protected variant)
        transition_matrix = self.compute_transition(x)  ### [B, n, n]

        ### compute projection direction for projected mixing
        projection_dir = self.compute_projection(x)  ### [B, n] or None

        # if projection_dir is None:
        #     x_mixed = einsum(
        #         transition_matrix, x, "b n1 n2, b n2 d -> b n1 d"
        #     )  # [B*, n, block_size]
        # else:
        #     proj_matrix = einsum(
        #         projection_dir, projection_dir, "b n1, b n2 -> b n1 n2"
        #     )  # [b, n, n]
        #     orthogonal_proj = (
        #         torch.eye(self.n, device=x.device, dtype=x.dtype) - proj_matrix
        #     )  # [b, n, n]
        #     x_proj = einsum(
        #         proj_matrix, x, "b n1 n2, b n2 d -> b n1 d"
        #     )  # [b, n, block_size]
        #     x_orth = einsum(
        #         orthogonal_proj, x, "b n1 n2, b n2 d -> b n1 d"
        #     )  # [b, n, block_size]
        #     x_mixed = x_proj + einsum(
        #         transition_matrix, x_orth, "b n1 n2, b n2 d -> b n1 d"
        #     )  # [B*, n, block_size]
        # return (x_mixed + Y).unflatten(0, leading).flatten(-2)

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
