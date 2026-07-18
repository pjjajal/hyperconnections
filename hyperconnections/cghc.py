# Weight decay note: several parameters are semantic anchors whose init values
# encode structure — decaying them toward 0 changes the dynamics rather than
# regularizing a feature transform. Turn weight decay OFF for:
#   - read_in, write_out             (saturated ±5 logits / round-robin pattern)
#   - conserv_A, diss_A, laplacian_A (static generator anchors)
#   - every 1-D parameter: alpha_*, log_dt_*, diss_diag, the *_log_scale
#     magnitudes, norm gains, biases. Decay is actively harmful for the
#     log-scales: dragging log(scale) from log(1e-3) toward 0 pushes the scale
#     toward 1, *increasing* dissipation by orders of magnitude.
# Keep decay on the read/write projections, generator predictors, dt projections,
# and the wrapped module's weights. Use
# ContinuousGenHyperConnections.split_decay_param_groups(model, wd) to build
# optimizer param groups with exactly this split.

import math
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
from timm.layers import trunc_normal_

from hyperconnections.ops import HAS_TRITON, expm_t18, expm_t18_triton, stream_mix_add
from hyperconnections.short_conv import DepthwiseShortConv1d


class CGHCProjections(NamedTuple):
    """Per-map slices of the fused input projection (see FusedInputProjection).

    read_in/write_out are always present. A CGHC has at most one conservative
    (skew) part and at most one dissipative part — psd/diagonal/laplacian are
    mutually exclusive — so `diss` carries whichever dissipative predictor is
    active. Each tensor is [B, seg_dim] in the ambient (autocast) dtype; disabled
    components are None.
    """

    read_in: torch.Tensor
    write_out: torch.Tensor
    conserv: torch.Tensor | None = None
    dt_conserv: torch.Tensor | None = None
    diss: torch.Tensor | None = None
    dt_diss: torch.Tensor | None = None
    proj_v: torch.Tensor | None = None


class FusedInputProjection(nn.Module):
    """Every x_norm-consuming linear map concatenated into ONE weight.

    The wide [B, input_dim] activation is read from HBM once and a single GEMM
    produces all projections, replacing one skinny GEMM per map. The output is
    sliced into a CGHCProjections bundle.

    The generator predictors come in symmetric (predictor, dt) pairs: a
    conservative pair when `conserv`, and a dissipative pair when `diss_dim` is
    set (its width is n·n for psd/laplacian, n for diagonal — the caller picks).

    Init / decay semantics carried over from the per-Linear version:
      - Predictors (conserv/diss/proj_v): bias-free, zero-init, so the generator/
        projection starts exactly at its static anchor.
      - Read/write gates and dt segments: small trunc_normal noise + their own
        1-D bias (zero-init) for input-dependent variation.
      - The fused weight is 2-D → keeps weight decay, identical to the separate
        weights it replaces; biases are 1-D → no decay (see split_decay_param_groups).
    """

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
    ):
        super().__init__()
        # (key, out_dim, has_bias, init) with init ∈ {"noise", "zero"}.
        # Predictor+dt come as a pair; "noise"/bias go on the gates and dt only.
        segments: list[tuple[str, int, bool, str]] = [
            ("read_in", n * m, bias, "noise"),
            ("write_out", n * m, bias, "noise"),
        ]
        if conserv:
            segments += [("conserv", n * n, False, "zero"), ("dt_conserv", n_dt, True, "noise")]
        if diss_dim is not None:
            segments += [("diss", diss_dim, False, "zero"), ("dt_diss", n_dt, True, "noise")]
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
        h = self.fused(x_norm)  # [B, total_out] — one GEMM

        def seg(key: str) -> torch.Tensor | None:
            bounds = self._slices.get(key)
            if bounds is None:
                return None
            out = h[..., bounds[0] : bounds[1]]
            if key in self.biases:
                # cast bias to the GEMM output dtype so a bias'd segment stays in
                # the ambient (autocast) dtype, matching the old fused-Linear op.
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
        # The read/write dynamic components come from the fused input projection
        # (self.proj_fused) built below, once all component flags are known.

        # dt parameters — log-space sigmoid interpolation:
        #   dt = exp(log dt_min + (log dt_max - log dt_min) * σ(θ))
        # Hard-bounded like a linear sigmoid, but with uniform *relative*
        # sensitivity across the range, so a small dt init sits on the responsive
        # part of the curve instead of the saturated tail (dt=0.01 in (0.001, 1.0)
        # lands at logit ≈ -0.69 rather than -4.7).
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
        if psd_diss:
            self.diss_A = nn.Parameter(torch.zeros(n, n))
        # Dissipation magnitude is factored out of the softplus argument:
        #   d = exp(log_scale) * softplus(anchor + pred(x))
        # The anchor sits at 0 (softplus'(0) = 0.5, responsive) instead of deep in
        # the saturated tail; the shared log-scale carries the "starts tiny"
        # magnitude and learns multiplicatively, with its gradient pooled across
        # streams and tokens.
        if diag_diss:
            self.diss_diag = nn.Parameter(torch.zeros(n))
            self.diss_log_scale = nn.Parameter(torch.empty(1))
        if laplacian:
            self.laplacian_A = nn.Parameter(torch.zeros(n, n))
            self.laplacian_log_scale = nn.Parameter(torch.empty(1))

        # Projection Direction
        if projection == "mean":
            self.register_buffer("projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "v":
            self.register_buffer("base_projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "none":
            self.projection_dir = None

        # Fused input projection: one GEMM for every x_norm-consuming linear map
        # (read/write gates, generator predictors, dt, proj_v). The single active
        # dissipative predictor is n·n wide for psd/laplacian, n for diagonal. See
        # FusedInputProjection for init/decay semantics.
        diss_dim = n if diag_diss else (n * n if (psd_diss or laplacian) else None)
        self.input_proj = FusedInputProjection(
            input_dim,
            n=n,
            m=m,
            n_dt=n_dt,
            bias=bias,
            conserv=conserv,
            diss_dim=diss_dim,
            proj_v=(projection == "v"),
        )

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
        target = (self.log_dt_init - self.log_dt_min) / self.log_dt_range
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
                read_in_init[j, j % self.n] = 5.0
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

        # Dissipation anchors at 0; magnitude in the log-scale, so dissipation
        # starts at ~1e-3 * softplus(0) ≈ 7e-4 (matching the old -8 tail init in
        # value, but on the responsive part of the softplus).
        if hasattr(self, "diss_diag"):
            nn.init.zeros_(self.diss_diag)
            nn.init.constant_(self.diss_log_scale, math.log(1e-3))

        if hasattr(self, "laplacian_A"):
            nn.init.zeros_(self.laplacian_A)
            nn.init.constant_(self.laplacian_log_scale, math.log(1e-3))

        self._init_log_dt(self.log_dt_conserv)
        self._init_log_dt(self.log_dt_diss)

        # Fused input projection self-initializes in its constructor; re-run here
        # so init_weights() stays idempotent and fully re-initializes the module.
        self.input_proj.reset_parameters()

        # mean projection: set to mean direction.
        # small noise for asymmetry breaking so projection isn't exactly static at init, but normalised to keep initial scale consistent.
        if self.projection == "mean":
            self.projection_dir.fill_(1.0 / math.sqrt(self.n))
            with torch.no_grad():
                self.projection_dir.add_(torch.randn_like(self.projection_dir) * 0.01)
                self.projection_dir.div_(self.projection_dir.norm())
        # proj_v: base direction static; the proj_v segment of proj_fused is
        # zero-init (handled above) so it starts at base_projection_dir with
        # input-dependent variation.
        if self.projection == "v":
            self.base_projection_dir.fill_(1.0 / math.sqrt(self.n))

        # RMSNorm weights: must be ones for proper normalization
        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            nn.init.ones_(self.norm.weight)

    ### Static matrices whose init values are semantic anchors (see module note).
    ### 1-D anchors (alpha_*, log_dt_*, diss_diag) are caught by the ndim rule in
    ### split_decay_param_groups instead. Every HC class declares its own set;
    ### consumers collect them from submodules via hasattr (see the classmethod).
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

    def compute_generator(self, proj: CGHCProjections) -> torch.Tensor:
        """Return the effective generator A of shape [B, n, n].

        Args:
            proj: Fused input projection (from self.input_proj). Its per-map
                slices replace the old per-Linear GEMMs; the generator fields are
                present exactly when the corresponding component is enabled.

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

        Projections run in the ambient (autocast) dtype; assembly is float32:
        A entries are O(dt * weight), near bf16 resolution early in training,
        and the expm consumes fp32 anyway. Projection slices are upcast to fp32
        as they enter the assembly; everything downstream stays fp32.
        """
        B = proj.read_in.shape[0]
        device = proj.read_in.device
        A = torch.zeros(B, self.n, self.n, device=device, dtype=torch.float32)

        # --- Conservative branch ---
        if hasattr(self, "conserv_A"):
            M = self.conserv_A + proj.conserv.reshape(B, self.n, self.n).float()
            logit_conserv = self.log_dt_conserv + proj.dt_conserv.float()  # [B, n]
            dt_conserv = torch.exp(
                self.log_dt_min + self.log_dt_range * torch.sigmoid(logit_conserv)
            )
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
            logit_diss = self.log_dt_diss + proj.dt_diss.float()  # [B, n]
            dt_diss = torch.exp(
                self.log_dt_min + self.log_dt_range * torch.sigmoid(logit_diss)
            )
            sqrt_dt_diss = dt_diss.sqrt()  # [B, n]

        # --- PSD dissipative (Gram matrix) branch ---
        if hasattr(self, "diss_A"):
            R = self.diss_A + proj.diss.reshape(B, self.n, self.n).float()
            with torch.autocast(device_type=device.type, enabled=False):
                K = R @ R.transpose(-1, -2) / (self.n**0.5)  # [B, n, n], PSD (fp32 even under autocast)
            if not self.vec_dt:
                # Scalar dt: equivalent to the sandwich but avoids unnecessary sqrt
                diss_dt = dt_diss.unsqueeze(-1) * K
            else:
                # Per-stream sandwich: (D^{1/2} K D^{1/2})_{ij} = sqrt_dt_i * K_{ij} * sqrt_dt_j
                diss_dt = sqrt_dt_diss[:, :, None] * K * sqrt_dt_diss[:, None, :]
            A = A - diss_dt

        # --- Diagonal dissipative branch ---
        if hasattr(self, "diss_diag"):
            d = torch.exp(self.diss_log_scale) * F.softplus(
                self.diss_diag + proj.diss.float()
            )  # [B, n], positive
            # Sandwich of a diagonal reduces to elementwise product: diag(sqrt_dt * d * sqrt_dt)
            # = diag(dt_diss * d)
            A = A - torch.diag_embed(
                dt_diss * d
            )  # dt_diss [B,1] * d [B,n] broadcasts correctly

        # --- Laplacian dissipative branch (shares the single "diss" predictor) ---
        if hasattr(self, "laplacian_A"):
            scores = self.laplacian_A + proj.diss.reshape(B, self.n, self.n).float()
            scores = 0.5 * (scores + scores.transpose(-1, -2))  # symmetrize
            adjacency = torch.exp(self.laplacian_log_scale) * F.softplus(scores)
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

    def compute_transition(self, proj: CGHCProjections) -> torch.Tensor:
        """Return Phi = exp(A), shape [B, n, n] (dt is folded into A).

        Args:
            proj: Fused input projection (from self.input_proj).
        """
        A = self.compute_generator(proj)  # fp32
        return self._matrix_exp(A).to(proj.read_in.dtype)

    def _matrix_exp_eager(self, A: torch.Tensor) -> torch.Tensor:
        return expm_t18(A.float()).to(A.dtype)

    def compute_read_write_weights(self, proj: CGHCProjections):
        """Compute dynamic read/write weights from the fused input projection.

        Args:
            proj: Fused input projection (from self.input_proj).
        """
        B = proj.read_in.shape[0]

        h_read_in = proj.read_in.reshape(B, self.n, self.m)
        h_write_out = proj.write_out.reshape(B, self.n, self.m)

        read_in = torch.sigmoid(
            self.alpha_read_in * h_read_in + self.read_in
        ).transpose(1, 2)  # [B, m, n]
        write_out = 2 * torch.sigmoid(
            self.alpha_write_out * h_write_out + self.write_out
        )  # [B, n, m]

        return write_out, read_in

    def compute_projection(self, proj: CGHCProjections):
        """Compute projection direction.

        Args:
            proj: Fused input projection (from self.input_proj). Only proj.proj_v
                is used, and only in "v" mode.
        """
        if self.projection == "mean":
            return self.projection_dir.unsqueeze(0)  # [1, n]
        elif self.projection == "v":
            v = proj.proj_v + self.base_projection_dir  # [B, n]
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
        self, proj: CGHCProjections, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (transition_matrix, Y) for stream mixing.

        Subclasses override this to apply φ₁(A) to Y (forced/exact integration).
        """
        return self.compute_transition(proj), Y

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

        ### One fused GEMM for every x_norm-derived projection; sliced downstream.
        proj = self.input_proj(x_norm)

        write_out, read_in = self.compute_read_write_weights(proj)

        ### Source term Y = H^post F(H^pre X)  (read → compute → write)
        ### Read in from over-width space to backbone width
        x_read = einsum(read_in, src, "b m n, b n d -> b m d")  ### [B*, m, block_size]

        ### Process through the backbone module
        out = self.module(x_read.reshape(*leading, self.embed_dim), **kwargs)

        ### Write out from backbone width back to the over-width space
        out = out.reshape(B, self.m, self.block_size)  ### [B*, m, block_size]
        Y = einsum(write_out, out, "b n m, b m d -> b n d")  ### [B*, n, block_size]

        ### Compute transition matrix and (optionally) apply φ₁ to Y
        transition_matrix, Y = self._transition_and_source(proj, Y)

        projection_dir = self.compute_projection(proj)  ### [B, n] or None

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
