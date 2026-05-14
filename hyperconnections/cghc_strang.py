import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from timm.models.layers import trunc_normal_

from hyperconnections.ops import HAS_TRITON, stream_mix_add


class ContinuousGenHyperConnectionsStrang(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        dt: float = 0.01,
        generator_type: Literal["conservative_diag_diss"] = "conservative_diag_diss",
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
        self.dt_min_cons = dt_min
        self.dt_max_cons = dt_max
        self.dt_min_diss = dt_min
        self.dt_max_diss = dt_max
        self.log_dt_init_cons = math.log(dt)
        self.log_dt_init_diss = math.log(dt)
        self.vec_dt = vec_dt
        n_dt = n if vec_dt else 1
        self.log_dt_conserv = nn.Parameter(torch.empty(n_dt), requires_grad=learn_dt)
        self.log_dt_diss = nn.Parameter(torch.empty(n_dt), requires_grad=learn_dt)
        self.dt_proj_conserv = nn.Linear(input_dim, n_dt, bias=True)
        self.dt_proj_diss = nn.Linear(input_dim, n_dt, bias=True)

        # Generator parameters: conservative + diagonal dissipation
        self.conserv_A = nn.Parameter(torch.eye(n, n))
        self.conv_pred = nn.Linear(input_dim, n * n, bias=False)
        self.diss_diag = nn.Parameter(torch.full((n,), -8.0, requires_grad=True))
        self.diss_pred = nn.Linear(input_dim, n, bias=False)

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
        self._stream_mix = (
            self._stream_mix_triton
            if use_triton and HAS_TRITON
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

        # Generator Static Parameters
        nn.init.eye_(self.conserv_A)
        # Small asymmetry so skew-sym part is non-zero at init
        with torch.no_grad():
            noise = torch.empty_like(self.conserv_A)
            trunc_normal_(noise, std=0.01)
            self.conserv_A.add_(noise)

        nn.init.constant_(self.diss_diag, -8.0)

        # Generator Dynamic Parameters
        nn.init.zeros_(self.conv_pred.weight)
        nn.init.zeros_(self.diss_pred.weight)

        # Initialize log_dt so that sigmoid(log_dt) * (dt_max - dt_min) + dt_min = dt_init.
        # log_dt has length n_dt (1 when vec_dt=False, n when vec_dt=True).
        dt_init_cons = math.exp(self.log_dt_init_cons)
        target = (dt_init_cons - self.dt_min_cons) / (
            self.dt_max_cons - self.dt_min_cons
        )
        target = min(max(target, 1e-4), 1 - 1e-4)
        bias_init = math.log(target / (1 - target))
        nn.init.constant_(self.log_dt_conserv, bias_init)

        dt_init_diss = math.exp(self.log_dt_init_diss)
        target = (dt_init_diss - self.dt_min_diss) / (
            self.dt_max_diss - self.dt_min_diss
        )
        target = min(max(target, 1e-4), 1 - 1e-4)
        bias_init = math.log(target / (1 - target))
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
        # proj_v: zero init for weights to start at base_projection_dir with input-dependent variation
        if self.projection == "v":
            nn.init.zeros_(self.projection_dir.weight)

        # RMSNorm weights: must be ones for proper normalization
        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)

    def compute_conservative(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Compute the conservative (skew-symmetric) part of the generator.

        Args:
            x_norm: Normalized input of shape [B, input_dim]

        Returns:
            Skew-symmetric matrix of shape [B, n, n] representing the conservative dynamics
        """
        B = x_norm.shape[0]
        M = self.conserv_A + self.conv_pred(x_norm).reshape(B, self.n, self.n)
        skew = 0.5 * (M - M.transpose(-1, -2))  # [B, n, n], skew-symmetric

        # dt scaling for conservative part
        logit_conserv = self.log_dt_conserv + self.dt_proj_conserv(x_norm)  # [B, n]
        dt_conserv = self.dt_min_cons + (
            self.dt_max_cons - self.dt_min_cons
        ) * torch.sigmoid(logit_conserv)
        if not self.vec_dt:
            skew = dt_conserv.unsqueeze(-1) * skew
        else:
            sqrt_dt_conserv = dt_conserv.sqrt()  # [B, n]
            skew = sqrt_dt_conserv[:, :, None] * skew * sqrt_dt_conserv[:, None, :]

        return skew

    def compute_dissipative(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Compute the dissipative (diagonal) part of the generator.

        Args:
            x_norm: Normalized input of shape [B, input_dim]

        Returns:
            Diagonal elements of shape [B, n] representing dissipative dynamics (negative values)
        """
        d = F.softplus(self.diss_diag + self.diss_pred(x_norm))  # [B, n], positive

        logit_diss = self.log_dt_diss + self.dt_proj_diss(x_norm)  # [B, n]
        dt_diss = self.dt_min_diss + (
            self.dt_max_diss - self.dt_min_diss
        ) * torch.sigmoid(logit_diss)

        # Negative since dissipation corresponds to negative eigenvalues
        return -dt_diss * d

    def compute_transition(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Compute the transition matrix Phi using Strang splitting: exp(0.5*D) exp(S) exp(0.5*D).

        Where S is the conservative (skew-symmetric) part and D is the dissipative (diagonal) part.

        Args:
            x_norm: Normalized input of shape [B, input_dim]

        Returns:
            Transition matrix of shape [B, n, n]
        """
        dtype = x_norm.dtype
        device = x_norm.device

        S = self.compute_conservative(x_norm)  # [B, n, n] skew-symmetric
        D = self.compute_dissipative(x_norm)   # [B, n] diagonal elements (negative)

        # Strang splitting: Phi = exp(0.5*D) exp(S) exp(0.5*D)
        # For diagonal D: exp(0.5*D) is element-wise exponential
        exp_half_D = torch.exp(0.5 * D)  # [B, n]

        # For skew-symmetric S: use Cayley transform exp(S) = (I - S)^{-1} (I + S)
        # Use float32 for numerical stability and solve_ex to avoid CPU sync
        identity = torch.eye(self.n, device=device, dtype=torch.float32).unsqueeze(0)  # [1, n, n]
        S_f32 = S.float()
        exp_S, _ = torch.linalg.solve_ex(identity - S_f32, identity + S_f32)  # [B, n, n]
        exp_S = exp_S.to(dtype)

        # Combine: diag(exp_half_D) @ exp_S @ diag(exp_half_D)
        return exp_half_D[:, :, None] * exp_S * exp_half_D[:, None, :]

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
        x_norm = self.norm(x.view(B, -1))  ### [B*, input_dim]

        write_out, read_in = self.compute_read_write_weights(x_norm)

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
        transition_matrix = self.compute_transition(x_norm)  ### [B, n, n]

        ### compute projection direction for projected mixing
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
