import math
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from timm.layers import trunc_normal_

from hyperconnections.ops import HAS_TRITON, expm_t18, expm_t18_triton, stream_mix_add
from hyperconnections.temporal_writeback import TemporalWriteback


GeneratorType = Literal[
    "conservative",
    "psd_diss",
    "diagonal_diss",
    "laplacian",
    "conservative_diag_diss",
    "conservative_psd_diss",
    "conservative_laplacian",
]
ProjectionType = Literal["mean", "v", "none"]

GENERATOR_COMPONENTS: dict[GeneratorType, tuple[bool, str | None]] = {
    "conservative": (True, None),
    "psd_diss": (False, "psd"),
    "diagonal_diss": (False, "diagonal"),
    "laplacian": (False, "laplacian"),
    "conservative_diag_diss": (True, "diagonal"),
    "conservative_psd_diss": (True, "psd"),
    "conservative_laplacian": (True, "laplacian"),
}


class CGHCProjections(NamedTuple):
    """Per-map slices of the fused input projection (see FusedInputProjection).

    read_in/write_out are always present. The flattened write_out segment also
    contains the temporal-component axis when enriched write-back is enabled.
    A CGHC has at most one conservative part and one dissipative part, so `diss`
    carries whichever dissipative predictor is active. Disabled components are
    None.
    """

    read_in: torch.Tensor
    write_out: torch.Tensor
    conserv: torch.Tensor | None = None
    dt_conserv: torch.Tensor | None = None
    diss: torch.Tensor | None = None
    dt_diss: torch.Tensor | None = None
    proj_v: torch.Tensor | None = None


class FusedInputProjection(nn.Module):
    """Compute every input-dependent CGHC projection with one linear layer."""

    def __init__(
        self,
        input_dim: int,
        *,
        n: int,
        m: int,
        n_dt: int,
        bias: bool,
        conserv: bool,
        diss_dim: int | None,
        proj_v: bool,
        writeback_components: int = 1,
    ) -> None:
        super().__init__()
        segments: list[tuple[str, int, bool, str]] = [
            ("read_in", n * m, bias, "noise"),
            ("write_out", n * m * writeback_components, bias, "noise"),
        ]
        if conserv:
            segments += [
                ("conserv", n * n, False, "zero"),
                ("dt_conserv", n_dt, True, "noise"),
            ]
        if diss_dim is not None:
            segments += [
                ("diss", diss_dim, False, "zero"),
                ("dt_diss", n_dt, True, "noise"),
            ]
        if proj_v:
            segments += [("proj_v", n, False, "zero")]

        offset = 0
        self._slices: dict[str, tuple[int, int]] = {}
        self._init: dict[str, str] = {}
        for key, dim, _has_bias, init_kind in segments:
            self._slices[key] = (offset, offset + dim)
            self._init[key] = init_kind
            offset += dim
        self.fused = nn.Linear(input_dim, offset, bias=False)
        self.biases = nn.ParameterDict(
            {key: nn.Parameter(torch.zeros(dim)) for key, dim, has_bias, _ in segments if has_bias}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            W = self.fused.weight  # [total_out, input_dim]
            for key, (s, e) in self._slices.items():
                if self._init[key] == "zero":
                    nn.init.zeros_(W[s:e])
                else:
                    trunc_normal_(W[s:e], std=0.01)
        for b in self.biases.values():
            nn.init.zeros_(b)

    def forward(self, x_norm: torch.Tensor) -> CGHCProjections:
        h = self.fused(x_norm)

        def seg(key: str) -> torch.Tensor | None:
            bounds = self._slices.get(key)
            if bounds is None:
                return None
            out = h[..., bounds[0] : bounds[1]]
            if key in self.biases:
                out = out + self.biases[key].to(out.dtype)
            return out

        return CGHCProjections(
            read_in=seg("read_in"),
            write_out=seg("write_out"),
            conserv=seg("conserv"),
            dt_conserv=seg("dt_conserv"),
            diss=seg("diss"),
            dt_diss=seg("dt_diss"),
            proj_v=seg("proj_v"),
        )


class ContinuousGenHyperConnections(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        dt: float = 0.01,
        generator_type: GeneratorType = "conservative_psd_diss",
        projection: ProjectionType = "none",
        learn_dt: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 1.0,
        bias: bool = False,
        elementwise_affine: bool = False,
        use_triton: bool = True,
        vec_dt: bool = False,
        sat_c: float | None = 2.0,
        writeback_kernel_sizes: tuple[int, ...] = (),
        writeback_orthogonalize: bool = True,
    ) -> None:
        super().__init__()

        if generator_type not in GENERATOR_COMPONENTS:
            raise ValueError(f"Unknown generator type: {generator_type!r}")
        if projection not in {"mean", "v", "none"}:
            raise ValueError(f"Unknown projection type: {projection!r}")

        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.generator_type = generator_type
        self.projection = projection
        self.has_conservative, self.dissipation_type = GENERATOR_COMPONENTS[
            generator_type
        ]
        self.writeback_components = len(writeback_kernel_sizes) + 1

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
        write_out_shape = (
            (n, m)
            if self.writeback_components == 1
            else (n, self.writeback_components, m)
        )
        self.write_out = nn.Parameter(torch.empty(write_out_shape))
        self.alpha_write_out = nn.Parameter(torch.empty(1))

        assert dt_min > 0, (
            f"dt_min must be positive (dt is parameterized in log space), got {dt_min}"
        )
        assert dt_min < dt < dt_max, (
            f"Initial dt ({dt}) must lie strictly in (dt_min, dt_max) = ({dt_min}, {dt_max})"
        )
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.log_dt_min = math.log(dt_min)
        self.log_dt_range = math.log(dt_max) - math.log(dt_min)
        self.log_dt_init = math.log(dt)
        self.vec_dt = vec_dt
        n_dt = n if vec_dt else 1
        self.log_dt_conserv = nn.Parameter(torch.empty(n_dt), requires_grad=learn_dt)
        self.log_dt_diss = nn.Parameter(torch.empty(n_dt), requires_grad=learn_dt)
        self.sat_c = sat_c

        if self.has_conservative:
            self.conserv_A = nn.Parameter(torch.eye(n, n))
        if self.dissipation_type == "psd":
            self.diss_A = nn.Parameter(torch.zeros(n, n))
        if self.dissipation_type == "diagonal":
            self.diss_diag = nn.Parameter(torch.zeros(n))
            self.diss_log_scale = nn.Parameter(torch.empty(1))
        if self.dissipation_type == "laplacian":
            self.laplacian_A = nn.Parameter(torch.zeros(n, n))
            self.laplacian_log_scale = nn.Parameter(torch.empty(1))

        # Projection Direction
        if projection == "mean":
            self.register_buffer("projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "v":
            self.register_buffer("base_projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "none":
            self.projection_dir = None

        diss_dim = {
            None: None,
            "diagonal": n,
            "psd": n * n,
            "laplacian": n * n,
        }[self.dissipation_type]
        self.input_proj = FusedInputProjection(
            input_dim,
            n=n,
            m=m,
            n_dt=n_dt,
            bias=bias,
            conserv=self.has_conservative,
            diss_dim=diss_dim,
            proj_v=(projection == "v"),
            writeback_components=self.writeback_components,
        )

        self.norm = nn.RMSNorm(input_dim, elementwise_affine=elementwise_affine)
        self.module = module
        self.writeback = (
            TemporalWriteback(
                embed_dim,
                writeback_kernel_sizes,
                orthogonalize=writeback_orthogonalize,
            )
            if writeback_kernel_sizes
            else None
        )
        self._use_triton = use_triton and HAS_TRITON
        self.init_weights()

    def _init_log_dt(self, param: nn.Parameter) -> None:
        target = (self.log_dt_init - self.log_dt_min) / self.log_dt_range
        target = min(max(target, 1e-4), 1 - 1e-4)
        nn.init.constant_(param, math.log(target / (1 - target)))

    def init_weights(self) -> None:
        # Stored as [n, m], although reads use the transpose [m, n].
        if self.m == 1:
            logit_1_over_n = math.log(1.0 / (self.n - 1)) if self.n > 1 else 10.0
            nn.init.constant_(self.read_in, logit_1_over_n)
        else:
            read_in_init = torch.full((self.m, self.n), -5.0)
            for j in range(self.m):
                read_in_init[j, j % self.n] = 5.0
            with torch.no_grad():
                self.read_in.copy_(read_in_init.T)

        # Stream i initially writes from module chunk i % m.
        write_out_init = torch.full((self.n, self.m), -5.0)
        for i in range(self.n):
            write_out_init[i, i % self.m] = 0.0
        if self.writeback_components > 1:
            write_out_init = write_out_init[:, None, :].expand_as(self.write_out)

        with torch.no_grad():
            self.write_out.copy_(write_out_init)
            self.read_in.add_(torch.randn_like(self.read_in) * 0.01)
            self.write_out.add_(torch.randn_like(self.write_out) * 0.01)

        nn.init.constant_(self.alpha_read_in, 0.01)
        nn.init.constant_(self.alpha_write_out, 0.01)

        if self.has_conservative:
            nn.init.eye_(self.conserv_A)
            with torch.no_grad():
                noise = torch.empty_like(self.conserv_A)
                trunc_normal_(noise, std=0.01)
                self.conserv_A.add_(noise)

        if self.dissipation_type == "psd":
            trunc_normal_(self.diss_A, std=0.01)

        if self.dissipation_type == "diagonal":
            nn.init.zeros_(self.diss_diag)
            nn.init.constant_(self.diss_log_scale, math.log(1e-3))

        if self.dissipation_type == "laplacian":
            nn.init.zeros_(self.laplacian_A)
            nn.init.constant_(self.laplacian_log_scale, math.log(1e-3))

        self._init_log_dt(self.log_dt_conserv)
        self._init_log_dt(self.log_dt_diss)

        self.input_proj.reset_parameters()

        if self.projection == "mean":
            self.projection_dir.fill_(1.0 / math.sqrt(self.n))
            with torch.no_grad():
                self.projection_dir.add_(torch.randn_like(self.projection_dir) * 0.01)
                self.projection_dir.div_(self.projection_dir.norm())
        if self.projection == "v":
            self.base_projection_dir.fill_(1.0 / math.sqrt(self.n))

        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)

    NO_DECAY_PARAM_NAMES = frozenset(
        {"read_in", "write_out", "conserv_A", "diss_A", "laplacian_A"}
    )

    @staticmethod
    def split_decay_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
        """Optimizer param groups with weight decay off where it corrupts semantics.

        Anchor names are collected from every submodule class that declares
        NO_DECAY_PARAM_NAMES, so models mixing HC variants are covered. No decay
        for those anchors and for all ndim <= 1 parameters; everything else keeps
        `weight_decay`.

        Usage:
            optimizer = torch.optim.AdamW(
                ContinuousGenHyperConnections.split_decay_param_groups(model, 0.05),
                lr=lr,
            )
        """
        anchor_leafs = frozenset().union(
            *(
                m.NO_DECAY_PARAM_NAMES
                for m in model.modules()
                if hasattr(m, "NO_DECAY_PARAM_NAMES")
            )
        )
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            leaf = name.rsplit(".", 1)[-1]
            if p.ndim <= 1 or leaf in anchor_leafs:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def compute_generator(self, projections: CGHCProjections) -> torch.Tensor:
        """Assemble the effective generator A in float32.

        With vector-valued time steps, the generator uses congruence sandwiches:

            A = D_S^{1/2} (S) D_S^{1/2}  -  D_Q^{1/2} (Q) D_Q^{1/2}

        Scalar time steps reduce this to:

            A = dt_conserv * S  -  dt_diss * Q
        """
        batch_size = projections.read_in.shape[0]
        device = projections.read_in.device
        A = torch.zeros(
            batch_size,
            self.n,
            self.n,
            device=device,
            dtype=torch.float32,
        )

        if self.has_conservative:
            M = self.conserv_A + projections.conserv.reshape(
                batch_size, self.n, self.n
            ).float()
            logit_conserv = self.log_dt_conserv + projections.dt_conserv.float()
            dt_conserv = torch.exp(
                self.log_dt_min + self.log_dt_range * torch.sigmoid(logit_conserv)
            )
            skew = 0.5 * (M - M.transpose(-1, -2))
            if self.sat_c is not None:
                skew = self.sat_c * torch.tanh(skew / self.sat_c)
            if not self.vec_dt:
                skew_dt = dt_conserv.unsqueeze(-1) * skew
            else:
                sqrt_dt_conserv = dt_conserv.sqrt()
                skew_dt = (
                    sqrt_dt_conserv[:, :, None] * skew * sqrt_dt_conserv[:, None, :]
                )

            A = A + skew_dt

        if self.dissipation_type is not None:
            logit_diss = self.log_dt_diss + projections.dt_diss.float()
            dt_diss = torch.exp(
                self.log_dt_min + self.log_dt_range * torch.sigmoid(logit_diss)
            )
            sqrt_dt_diss = dt_diss.sqrt()

        if self.dissipation_type == "psd":
            R = self.diss_A + projections.diss.reshape(
                batch_size, self.n, self.n
            ).float()
            if self.sat_c is not None:
                R = self.sat_c * torch.tanh(R / self.sat_c)
            with torch.autocast(device_type=device.type, enabled=False):
                K = R @ R.transpose(-1, -2) / (self.n**0.5)
            if not self.vec_dt:
                diss_dt = dt_diss.unsqueeze(-1) * K
            else:
                diss_dt = sqrt_dt_diss[:, :, None] * K * sqrt_dt_diss[:, None, :]
            A = A - diss_dt

        if self.dissipation_type == "diagonal":
            d = torch.exp(self.diss_log_scale) * F.softplus(
                self.diss_diag + projections.diss.float()
            )
            if self.sat_c is not None:
                d = d / (1 + d / self.sat_c)
            A = A - torch.diag_embed(dt_diss * d)

        if self.dissipation_type == "laplacian":
            scores = self.laplacian_A + projections.diss.reshape(
                batch_size, self.n, self.n
            ).float()
            scores = 0.5 * (scores + scores.transpose(-1, -2))
            adjacency = torch.exp(self.laplacian_log_scale) * F.softplus(scores)
            if self.sat_c is not None:
                adjacency = adjacency / (1 + adjacency / self.sat_c)
            adjacency = adjacency - torch.diag_embed(
                torch.diagonal(adjacency, dim1=-2, dim2=-1)
            )
            degree = torch.diag_embed(adjacency.sum(dim=-1))
            laplacian = degree - adjacency
            if not self.vec_dt:
                laplacian_dt = dt_diss.unsqueeze(-1) * laplacian
            else:
                laplacian_dt = (
                    sqrt_dt_diss[:, :, None] * laplacian * sqrt_dt_diss[:, None, :]
                )
            A = A - laplacian_dt

        return A

    def compute_transition(self, projections: CGHCProjections) -> torch.Tensor:
        """Return exp(A), where the time step is already folded into A."""
        A = self.compute_generator(projections)
        transition = (
            expm_t18_triton(A) if self._use_triton else expm_t18(A.float())
        )
        return transition.to(projections.read_in.dtype)

    def compute_read_write_weights(
        self, projections: CGHCProjections
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dynamic read/write weights."""
        batch_size = projections.read_in.shape[0]

        h_read_in = projections.read_in.reshape(batch_size, self.n, self.m)
        h_write_out = projections.write_out.reshape(
            batch_size,
            self.n,
            self.writeback_components,
            self.m,
        )

        read_in = torch.sigmoid(
            self.alpha_read_in * h_read_in + self.read_in
        ).transpose(1, 2)  # [B, m, n]
        write_out_bias = self.write_out
        if self.writeback_components == 1:
            write_out_bias = write_out_bias.unsqueeze(1)
        write_out = 2 * torch.sigmoid(
            self.alpha_write_out * h_write_out + write_out_bias
        )
        if self.writeback_components == 1:
            write_out = write_out.squeeze(2)

        return write_out, read_in

    def compute_projection(
        self, projections: CGHCProjections
    ) -> torch.Tensor | None:
        """Compute the preserved stream direction, if configured."""
        if self.projection == "mean":
            return self.projection_dir.unsqueeze(0)
        if self.projection == "v":
            v = projections.proj_v + self.base_projection_dir
            return F.normalize(v, dim=-1)
        return None

    def _stream_mix(
        self,
        x: torch.Tensor,
        transition_matrix: torch.Tensor,
        Y: torch.Tensor,
        projection_dir: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._use_triton:
            if projection_dir is not None:
                projection_dir = projection_dir.expand(x.shape[0], -1)
            return stream_mix_add(transition_matrix, x, Y, projection_dir)

        if projection_dir is None:
            x_mixed = einsum(
                transition_matrix, x, "b n1 n2, b n2 d -> b n1 d"
            )
        else:
            x_proj = einsum(
                projection_dir,
                projection_dir,
                x,
                "b n1, b n2, b n2 d -> b n1 d",
            )
            x_orth = x - x_proj
            x_mixed = x_proj + einsum(
                transition_matrix, x_orth, "b n1 n2, b n2 d -> b n1 d"
            )
        return x_mixed + Y

    def _transition_and_source(
        self, projections: CGHCProjections, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allow forced CGHC to apply φ₁(A) to the source."""
        return self.compute_transition(projections), source

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, *, input_dim]  (any number of leading dims, last dim = n * block_size)
        Returns:
            [B, *, input_dim]
        """
        leading_shape = x.shape[:-1]
        streams = x.reshape(-1, self.n, self.block_size)
        batch_size = streams.shape[0]

        x_norm = self.norm(streams.flatten(1))
        projections: CGHCProjections = self.input_proj(x_norm)

        write_weights, read_weights = self.compute_read_write_weights(projections)

        module_input = einsum(
            read_weights,
            streams,
            "b m n, b n d -> b m d",
        )
        module_output = self.module(
            module_input.reshape(*leading_shape, self.embed_dim),
            **kwargs,
        )

        if self.writeback is None:
            write_weights = write_weights.unsqueeze(2)
            writeback_components = module_output.reshape(
                batch_size,
                1,
                self.m,
                self.block_size,
            )
        else:
            writeback_components = self.writeback(module_output).reshape(
                batch_size,
                self.writeback_components,
                self.m,
                self.block_size,
            )

        source_update = einsum(
            write_weights,
            writeback_components,
            "b n k m, b k m d -> b n d",
        )

        transition, source_update = self._transition_and_source(
            projections,
            source_update,
        )
        projection_dir = self.compute_projection(projections)

        updated_streams = self._stream_mix(
            x=streams,
            transition_matrix=transition,
            Y=source_update,
            projection_dir=projection_dir,
        )

        return updated_streams.reshape(*leading_shape, self.input_dim)
