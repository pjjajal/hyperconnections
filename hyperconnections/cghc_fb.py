import torch
import torch.nn.functional as F
from einops import einsum

from hyperconnections.cghc import CGHCProjections, ContinuousGenHyperConnections


def clip_inflow(
    gamma: torch.Tensor,
    gain: torch.Tensor,
    decay: float,
    budget: float,
    transition_bound: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip the inflow weight so the running gain bound stays at or below ``budget``.

    With ||T|| <= tau (``transition_bound``), an inflow sublayer has stream-map norm at most
    max(tau, sqrt(decay^2 + gamma^2 tau^2)); tau = 1 gives max(1, sqrt(decay^2 + gamma^2)).
    With room r = budget / gain, allowing |gamma| <= sqrt(r^2 - decay^2) / tau keeps
    gain * max(tau, sqrt(decay^2 + gamma^2 tau^2)) <= budget. When tau < 1 the step can charge
    less than 1, which refills the budget.

    Args:
        gamma: requested inflow weights, [B, 1].
        gain: running gain bound before this sublayer, [B, 1], at most ``budget``.
        transition_bound: tau, an upper bound on ||T||, [B, 1] or a float in (0, 1].
    Returns:
        (clipped gamma, updated gain), both [B, 1].
    """
    room = budget / gain
    cap = (room.square() - decay**2).clamp_min(0.0).sqrt() / transition_bound
    gamma = torch.maximum(torch.minimum(gamma, cap), -cap)
    step = torch.maximum(
        torch.as_tensor(transition_bound, dtype=gain.dtype, device=gain.device),
        (decay**2 + gamma.square() * transition_bound**2).sqrt(),
    )
    return gamma, gain * step


class ContinuousGenHyperConnectionsFeedback(ContinuousGenHyperConnections):
    """CGHC with the inflow projection and a feedback budget on stream amplification.

    Every connector receives the running gain bound G of the sublayers before it and clips its
    inflow so G never exceeds ``inflow_budget``. Since the norm of a product of stream maps is at
    most the product of their norms, every stack of sublayers then amplifies the streams (and
    their gradients, which pass through the transposed maps) by at most ``inflow_budget``, at any
    depth and for any per-layer preserved directions v.

    ``forward`` takes and returns the gain alongside the streams; callers thread it from one
    connector to the next, starting from ones (or None) at the first connector.

    With ``refill=True`` (diagonal dissipation only), each step is charged the damping-aware bound
    max(tau, sqrt(decay^2 + gamma^2 tau^2)) with tau = exp(-min_i dt_diss_i d_i) >= ||T||, so
    damped steps charge less than 1 and give budget back. The guarantee is for exact arithmetic:
    T and the stream map are rounded to the training dtype (bf16) before mixing.
    """

    def __init__(self, *args, inflow_budget: float = 2.0, refill: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.projection_variant != "inflow" or not self.dynamic_inflow:
            raise ValueError("CGHC feedback requires projection_variant='inflow' and a projection")
        if inflow_budget < 1.0:
            raise ValueError("inflow_budget must be at least 1")
        if refill and self.dissipation_type != "diagonal":
            raise ValueError("CGHC feedback refill requires diagonal dissipation")
        self.inflow_budget = inflow_budget
        self.refill = refill

    def transition_bound(self, projections: CGHCProjections) -> torch.Tensor:
        """Per-row upper bound tau on ||T||, shape [B, 1], from the diagonal damping.

        With diagonal dissipation, A = (skew-symmetric rotation) - diag(dt_diss * d), so the
        symmetric part of A is exactly -diag(dt_diss * d) and
        ||exp(A)|| <= exp(-min_i dt_diss_i * d_i). This mirrors compute_generator.
        """
        logit_diss = self.log_dt_diss + projections.dt_diss.float()
        dt_diss = torch.exp(self.log_dt_min + self.log_dt_diss_range * torch.sigmoid(logit_diss))
        d = F.softplus(self.diss_diag + projections.diss.float())
        if self.sat_c is not None:
            d = d / (1 + d / self.sat_c)
        return torch.exp(-(dt_diss * d).amin(dim=-1, keepdim=True))

    def forward(
        self, x: torch.Tensor, gain: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, *, input_dim]
            gain: running gain bound, [B, *] (None means ones, i.e. the first connector).
        Returns:
            (updated streams [B, *, input_dim], updated gain [B, *]).
        """
        leading_shape = x.shape[:-1]
        streams = x.reshape(-1, self.n, self.block_size)
        batch_size = streams.shape[0]
        gain_dtype = torch.promote_types(x.dtype, torch.float32)
        if gain is None:
            gain = torch.ones(leading_shape, device=x.device, dtype=gain_dtype)
        # Controller state: the cap holds either way, so keep it out of the autograd graph.
        gain = gain.detach().reshape(batch_size, 1).to(gain_dtype)

        projections: CGHCProjections = self.input_proj.forward_rms_normalized(
            streams.flatten(1), self.norm
        )
        write_weights, read_weights = self.compute_read_write_weights(projections)

        module_input = einsum(read_weights, streams, "b m n, b n d -> b m d")
        module_output = self.module(
            (module_input.reshape(*leading_shape, self.embed_dim)
             if self.num_readouts is None else
             module_input.reshape(*leading_shape, self.num_readouts, self.block_size)),
            **kwargs,
        )
        if self.writeback is None:
            write_weights = write_weights.unsqueeze(2)
            writeback_components = module_output.reshape(batch_size, 1, self.m, self.block_size)
        else:
            writeback_components = self.writeback(module_output).reshape(
                batch_size, self.writeback_components, self.m, self.block_size
            )
        source_update = einsum(write_weights, writeback_components, "b n k m, b k m d -> b n d")

        streams = streams.reshape(
            batch_size, self.num_mixing_streams, self.block_size
        )
        source_update = source_update.reshape_as(streams)
        transition, source_update = self._transition_and_source(projections, source_update)

        tau = self.transition_bound(projections).detach().to(gain_dtype) if self.refill else 1.0
        gamma, gain = clip_inflow(
            self.compute_inflow(projections).to(gain_dtype),
            gain,
            self.projection_decay,
            self.inflow_budget,
            tau,
        )
        updated_streams = self._stream_mix(
            x=streams,
            transition_matrix=transition,
            Y=source_update,
            projection_dir=self.compute_projection(projections),
            inflow=gamma,
        )
        return updated_streams.reshape(*leading_shape, self.input_dim), gain.detach().reshape(leading_shape)
