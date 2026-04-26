import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from timm.models.layers import trunc_normal_

from hyperconnections.ops import HAS_TRITON, stream_mix_add


class ContinuousGenHyperConnections(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        n_heads: int = 1,
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
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.projection = projection

        assert embed_dim % m == 0, (
            f"embed_dim ({embed_dim}) must be divisible by m ({m})"
        )
        assert input_dim == int((n / m) * embed_dim), (
            f"input_dim must be (n/m)*embed_dim, got {input_dim} vs {int((n / m) * embed_dim)}"
        )

        self.block_size = embed_dim // m

        assert self.block_size % n_heads == 0, (
            f"block_size ({self.block_size}) must be divisible by n_heads ({n_heads})"
        )
        self.head_dim = self.block_size // n_heads
        # features seen by each head: n streams × head_dim features each
        self.head_input_dim = n * self.head_dim

        # Read/write parameters following mHC convention.
        # Predictors take head_input_dim; outputs are [B*n_heads, n, m] (heads in batch).
        self.read_in = nn.Parameter(torch.empty(n, m))
        self.alpha_read_in = nn.Parameter(torch.empty(1))
        self.write_out = nn.Parameter(torch.empty(n, m))
        self.alpha_write_out = nn.Parameter(torch.empty(1))

        self.proj_read_in = nn.Linear(self.head_input_dim, n * m, bias=bias)
        self.proj_write_out = nn.Linear(self.head_input_dim, n * m, bias=bias)

        # dt parameters — per-head static bias, shared input-dependent predictor
        assert dt_min < dt < dt_max, (
            f"Initial dt ({dt}) must lie strictly in (dt_min, dt_max) = ({dt_min}, {dt_max})"
        )
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt
        self.vec_dt = vec_dt
        n_dt = n if vec_dt else 1
        self.log_dt_conserv = nn.Parameter(torch.empty(n_heads, n_dt), requires_grad=learn_dt)
        self.log_dt_diss = nn.Parameter(torch.empty(n_heads, n_dt), requires_grad=learn_dt)
        self.dt_proj_conserv = nn.Linear(self.head_input_dim, n_dt, bias=True)
        self.dt_proj_diss = nn.Linear(self.head_input_dim, n_dt, bias=True)

        # Generator parameters — per-head static bias [n_heads, n, n], shared dynamic predictor
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
            self.conserv_A = nn.Parameter(torch.empty(n_heads, n, n))
            self.conv_pred = nn.Linear(self.head_input_dim, n * n, bias=True)
        if psd_diss:
            self.diss_A = nn.Parameter(torch.zeros(n_heads, n, n))
            self.diss_pred = nn.Linear(self.head_input_dim, n * n, bias=True)
        if diag_diss:
            self.diss_diag = nn.Parameter(torch.full((n_heads, n), -2.0, requires_grad=True))
            self.diss_pred = nn.Linear(self.head_input_dim, n, bias=True)
        if laplacian:
            self.laplacian_A = nn.Parameter(torch.zeros(n_heads, n, n))
            self.laplacian_pred = nn.Linear(self.head_input_dim, n * n, bias=True)

        # Projection Direction.
        # "mean": single shared direction [n], broadcast over all heads.
        # "v": per-head predictor taking head_input_dim -> n; outputs [B*n_heads, n].
        if projection == "mean":
            self.register_buffer("projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "v":
            self.projection_dir = nn.Linear(self.head_input_dim, n, bias=True)
        elif projection == "none":
            self.projection_dir = None

        # Single norm over head_input_dim — each head's features are normalised independently.
        # When n_heads=1, head_input_dim == input_dim, recovering the original behaviour.
        self.norm = nn.RMSNorm(self.head_input_dim, elementwise_affine=elementwise_affine)
        self.module = module
        # Triton kernel expects a single [B, n, n] matrix — only use it for n_heads=1
        self._stream_mix = (
            self._stream_mix_triton
            if use_triton and HAS_TRITON and n_heads == 1
            else self._stream_mix_eager
        )
        self.init_weights()

    def init_weights(self):
        # read_in: σ(bias) = 1/n  →  bias = log(1/(n-1))
        logit_1_over_n = math.log(1.0 / (self.n - 1)) if self.n > 1 else 10.0
        nn.init.constant_(self.read_in, logit_1_over_n)
        with torch.no_grad():
            self.read_in.add_(
                torch.randn_like(self.read_in) * 0.01
            )  # small noise for asymmetry breaking
        # write_out: 2·σ(0) = 1
        trunc_normal_(self.write_out, std=0.01)
        # Alpha gating: 0.01 so dynamic component starts negligible
        nn.init.constant_(self.alpha_read_in, 0.01)
        nn.init.constant_(self.alpha_write_out, 0.01)

        # Generator Static Parameters — each head initialised independently
        if hasattr(self, "conserv_A"):
            with torch.no_grad():
                for h in range(self.n_heads):
                    nn.init.eye_(self.conserv_A[h])
                noise = torch.empty_like(self.conserv_A)
                trunc_normal_(noise, std=0.01)
                self.conserv_A.add_(noise)

        if hasattr(self, "diss_A"):
            trunc_normal_(self.diss_A, std=0.01)

        if hasattr(self, "diss_diag"):
            nn.init.constant_(self.diss_diag, -2.0)

        if hasattr(self, "laplacian_A"):
            nn.init.constant_(self.laplacian_A, -2.0)

        # Generator Dynamic Parameters
        for pred_name in ("conv_pred", "diss_pred", "laplacian_pred"):
            if hasattr(self, pred_name):
                pred = getattr(self, pred_name)
                trunc_normal_(pred.weight, std=0.01)
                nn.init.zeros_(pred.bias)

        # Initialize log_dt so that sigmoid(log_dt) * (dt_max - dt_min) + dt_min = dt_init.
        # log_dt has shape [n_heads, n_dt]; all heads start at the same value.
        target = (self.dt_init - self.dt_min) / (self.dt_max - self.dt_min)
        target = min(max(target, 1e-4), 1 - 1e-4)
        bias_init = math.log(target / (1 - target))
        nn.init.constant_(self.log_dt_conserv, bias_init)
        nn.init.constant_(self.log_dt_diss, bias_init)

        for proj in (self.dt_proj_conserv, self.dt_proj_diss):
            trunc_normal_(proj.weight, std=0.01)
            nn.init.zeros_(proj.bias)

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

    def compute_generator(self, x_norm_h: torch.Tensor) -> torch.Tensor:
        """Return the effective generator A of shape [B*n_heads, n, n].

        x_norm_h: [B*n_heads, head_input_dim] — already normalised per-head features.

        Each head sees only its own feature slice and produces an independent [n, n]
        generator via the same conservative/dissipative construction as the single-head
        case. The static bias is per-head ([n_heads, n, n]); the dynamic predictor is
        shared across heads. Stability (exp(A_h) contractive) holds per-head and
        therefore for the concatenated map.

        When vec_dt=True, the generator is built via a symmetric congruence sandwich:

            A = D_S^{1/2} (S) D_S^{1/2}  -  D_Q^{1/2} (Q) D_Q^{1/2}

        where D_S = diag(dt_conserv), D_Q = diag(dt_diss), each with shape [B*n_heads, n].
        When vec_dt=False, dt_conserv and dt_diss are scalars (shape [B*n_heads, 1]).
        """
        Bh = x_norm_h.shape[0]
        B = Bh // self.n_heads
        A = torch.zeros(Bh, self.n, self.n, device=x_norm_h.device, dtype=x_norm_h.dtype)

        # --- Conservative branch ---
        if hasattr(self, "conserv_A"):
            # static: [n_heads, n, n] -> [Bh, n, n]
            static_conserv = self.conserv_A.unsqueeze(0).expand(B, -1, -1, -1).reshape(Bh, self.n, self.n)
            M = static_conserv + self.conv_pred(x_norm_h).reshape(Bh, self.n, self.n)
            # log_dt: [n_heads, n_dt] -> [Bh, n_dt]
            logit_conserv = (
                self.log_dt_conserv.unsqueeze(0).expand(B, -1, -1).reshape(Bh, -1)
                + self.dt_proj_conserv(x_norm_h)
            )
            dt_conserv = self.dt_min + (
                self.dt_max - self.dt_min
            ) * torch.sigmoid(logit_conserv)
            skew = M - M.transpose(-1, -2)  # [Bh, n, n], skew-symmetric
            if not self.vec_dt:
                skew_dt = dt_conserv.unsqueeze(-1) * skew
            else:
                sqrt_dt_conserv = dt_conserv.sqrt()  # [Bh, n]
                skew_dt = sqrt_dt_conserv[:, :, None] * skew * sqrt_dt_conserv[:, None, :]
            A = A + skew_dt

        # --- Shared dissipative dt ---
        if (
            hasattr(self, "diss_A")
            or hasattr(self, "diss_diag")
            or hasattr(self, "laplacian_A")
        ):
            logit_diss = (
                self.log_dt_diss.unsqueeze(0).expand(B, -1, -1).reshape(Bh, -1)
                + self.dt_proj_diss(x_norm_h)
            )
            dt_diss = self.dt_min + (
                self.dt_max - self.dt_min
            ) * torch.sigmoid(logit_diss)
            sqrt_dt_diss = dt_diss.sqrt()  # [Bh, n_dt]

        # --- PSD dissipative (Gram matrix) branch ---
        if hasattr(self, "diss_A"):
            static_diss = self.diss_A.unsqueeze(0).expand(B, -1, -1, -1).reshape(Bh, self.n, self.n)
            R = static_diss + self.diss_pred(x_norm_h).reshape(Bh, self.n, self.n)
            K = R @ R.transpose(-1, -2) / (self.n**0.5)  # [Bh, n, n], PSD
            if not self.vec_dt:
                diss_dt = dt_diss.unsqueeze(-1) * K
            else:
                diss_dt = sqrt_dt_diss[:, :, None] * K * sqrt_dt_diss[:, None, :]
            A = A - diss_dt

        # --- Diagonal dissipative branch ---
        if hasattr(self, "diss_diag"):
            static_diag = self.diss_diag.unsqueeze(0).expand(B, -1, -1).reshape(Bh, self.n)
            d = F.softplus(static_diag + self.diss_pred(x_norm_h))  # [Bh, n], positive
            A = A - torch.diag_embed(dt_diss * d)

        # --- Laplacian dissipative branch ---
        if hasattr(self, "laplacian_A"):
            static_lap = self.laplacian_A.unsqueeze(0).expand(B, -1, -1, -1).reshape(Bh, self.n, self.n)
            scores = static_lap + self.laplacian_pred(x_norm_h).reshape(Bh, self.n, self.n)
            scores = 0.5 * (scores + scores.transpose(-1, -2))  # symmetrize
            adjacency = F.softplus(scores)
            adjacency = adjacency - torch.diag_embed(
                torch.diagonal(adjacency, dim1=-2, dim2=-1)
            )
            degree = torch.diag_embed(adjacency.sum(dim=-1))
            laplacian = degree - adjacency  # PSD
            if not self.vec_dt:
                laplacian_dt = dt_diss.unsqueeze(-1) * laplacian
            else:
                laplacian_dt = sqrt_dt_diss[:, :, None] * laplacian * sqrt_dt_diss[:, None, :]
            A = A - laplacian_dt

        return A

    def compute_transition(self, x_norm_h: torch.Tensor) -> torch.Tensor:
        """x_norm_h: [B*n_heads, head_input_dim] -> Phi: [B*n_heads, n, n]."""
        A = self.compute_generator(x_norm_h)
        return torch.linalg.matrix_exp(A.float()).to(x_norm_h.dtype)

    def compute_read_write_weights(self, x_norm_h: torch.Tensor):
        """x_norm_h: [B*n_heads, head_input_dim] -> ([B*n_heads, n, m], [B*n_heads, m, n])."""
        Bh = x_norm_h.shape[0]
        h_read_in = self.proj_read_in(x_norm_h).reshape(Bh, self.n, self.m)
        h_write_out = self.proj_write_out(x_norm_h).reshape(Bh, self.n, self.m)
        # static biases [n, m] broadcast over Bh
        read_in = torch.sigmoid(
            self.alpha_read_in * h_read_in + self.read_in
        ).transpose(1, 2)  # [Bh, m, n]
        write_out = 2 * torch.sigmoid(
            self.alpha_write_out * h_write_out + self.write_out
        )  # [Bh, n, m]
        return write_out, read_in

    def compute_projection(self, x_norm_h: torch.Tensor):
        """x_norm_h: [B*n_heads, head_input_dim] -> [B*n_heads, n] or [1, n] or None."""
        if self.projection == "mean":
            return self.projection_dir.unsqueeze(0)  # [1, n], broadcasts over B*n_heads
        elif self.projection == "v":
            v = self.projection_dir(x_norm_h)  # [B*n_heads, n]
            return F.normalize(v, dim=-1)
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
            )
        else:
            proj_matrix = einsum(
                projection_dir, projection_dir, "b n1, b n2 -> b n1 n2"
            )
            orthogonal_proj = (
                torch.eye(self.n, device=x.device, dtype=x.dtype) - proj_matrix
            )
            x_proj = einsum(proj_matrix, x, "b n1 n2, b n2 d -> b n1 d")
            x_orth = einsum(orthogonal_proj, x, "b n1 n2, b n2 d -> b n1 d")
            x_mixed = x_proj + einsum(
                transition_matrix, x_orth, "b n1 n2, b n2 d -> b n1 d"
            )
        return x_mixed + Y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, *, input_dim]  (any number of leading dims, last dim = n * block_size)
        Returns:
            [B, *, input_dim]
        """
        leading = x.shape[:-1]
        x = x.reshape(-1, self.n, self.block_size)  # [B, n, block_size]
        B = x.shape[0]
        Bh = B * self.n_heads

        # Fold heads into batch and normalise once
        x_h = (
            x.reshape(B, self.n, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(Bh, self.n, self.head_dim)
        )  # [Bh, n, head_dim]
        x_norm_h = self.norm(x_h.reshape(Bh, self.head_input_dim))  # [Bh, head_input_dim]

        write_out_h, read_in_h = self.compute_read_write_weights(x_norm_h)

        ### Source term Y = H^post F(H^pre X)  (read → compute → write)
        ### Read: [Bh, m, n] x [Bh, n, head_dim] -> [Bh, m, head_dim]
        x_read_h = einsum(read_in_h, x_h, "b m n, b n d -> b m d")

        ### Reshape for module: [Bh, m, head_dim] -> [B, m, n_heads, head_dim] -> [*, embed_dim]
        out = self.module(
            x_read_h.reshape(B, self.n_heads, self.m, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(*leading, self.embed_dim),
            **kwargs,
        )

        ### Reshape module output back: [B, embed_dim] -> [Bh, m, head_dim]
        out_h = (
            out.reshape(B, self.m, self.n_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(Bh, self.m, self.head_dim)
        )

        ### Write: [Bh, n, m] x [Bh, m, head_dim] -> [Bh, n, head_dim]
        Y_h = einsum(write_out_h, out_h, "b n m, b m d -> b n d")

        ### Stream Mixing
        transition_matrix = self.compute_transition(x_norm_h)  # [Bh, n, n]
        projection_dir = self.compute_projection(x_norm_h)     # [Bh, n] or [1, n] or None

        mixed_h = self._stream_mix(
            x=x_h,
            transition_matrix=transition_matrix,
            Y=Y_h,
            projection_dir=projection_dir,
        )  # [Bh, n, head_dim]

        ### Unfold heads from batch: [Bh, n, head_dim] -> [B, n, block_size]
        result = (
            mixed_h.reshape(B, self.n_heads, self.n, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(B, self.n, self.block_size)
        )
        return result.unflatten(0, leading).flatten(-2)
