import math

import pytest
import torch
import torch.nn as nn

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.cghc_fb import ContinuousGenHyperConnectionsFeedback, clip_inflow

N, D = 4, 32
DECAY = 0.99
KWARGS = dict(
    generator_type="conservative_diag_diss",
    projection="v",
    projection_variant="inflow",
    projection_decay=DECAY,
    dt=0.015,
    dt_min=1e-4,
    dt_max=0.35,
    learn_dt=True,
    vec_dt=True,
    sat_c=10.0,
    init_transition_contraction=0.95,
    use_triton=False,
)


def make(cls=ContinuousGenHyperConnectionsFeedback, **kw):
    torch.manual_seed(0)
    return cls(n=N, m=1, input_dim=N * D, embed_dim=D, module=nn.Linear(D, D, bias=False), **{**KWARGS, **kw}).double()


def random_transitions(batch, rotation=1.5, seed=1):
    g = torch.Generator().manual_seed(seed)
    skew = torch.randn(batch, N, N, dtype=torch.float64, generator=g) * rotation
    damp = torch.diag_embed(torch.rand(batch, N, dtype=torch.float64, generator=g) * 0.1)
    return torch.linalg.matrix_exp(skew - skew.mT - damp)


@pytest.mark.parametrize("budget", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("depth", [24, 96, 400])
def test_budget_caps_products_for_per_layer_v(budget, depth):
    """Full requested inflow, a fresh random v at every sublayer: products stay <= budget."""
    module, batch = make(inflow_budget=budget), 64
    g = torch.Generator().manual_seed(2)
    gain = torch.ones(batch, 1, dtype=torch.float64)
    product = torch.eye(N, dtype=torch.float64).expand(batch, N, N)
    worst = 0.0
    for k in range(depth):
        v = torch.nn.functional.normalize(torch.randn(batch, N, dtype=torch.float64, generator=g), dim=-1)
        requested = torch.ones(batch, 1, dtype=torch.float64)
        gamma, gain = clip_inflow(requested, gain, DECAY, budget)
        product = module._inflow_transition(random_transitions(batch, seed=100 + k), v, gamma) @ product
        worst = max(worst, torch.linalg.matrix_norm(product, ord=2).max().item())
        assert gain.max() <= budget + 1e-9
    assert worst <= budget + 1e-9


def test_worst_case_rotation_construction_is_capped():
    """The -arctan(1/alpha) rotation sequence grows to ~3,600 ungated at depth 24."""
    module, budget = make(inflow_budget=2.0), 2.0
    th = -math.atan(1 / DECAY)
    rot = lambda t: torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=torch.float64)
    embed = lambda m: torch.block_diag(m, torch.eye(N - 2, dtype=torch.float64))
    gain, product = torch.ones(1, 1, dtype=torch.float64), torch.eye(N, dtype=torch.float64)
    for k in range(400):
        R = embed(rot(k * th))
        T = R @ embed(torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)) @ R.T
        gamma, gain = clip_inflow(torch.ones(1, 1, dtype=torch.float64), gain, DECAY, budget)
        product = module._inflow_transition(T[None], R[:, 0][None], gamma)[0] @ product
    assert torch.linalg.matrix_norm(product, ord=2) <= budget + 1e-9


def test_large_budget_matches_plain_inflow():
    x = torch.randn(3, 5, N * D, dtype=torch.float64)
    fb_out, gain = make(inflow_budget=1e6)(x)
    plain_out = make(cls=ContinuousGenHyperConnections)(x)
    torch.testing.assert_close(fb_out, plain_out)
    assert gain.shape == (3, 5) and (gain >= 1).all()


def test_gain_threads_and_clips():
    module = make(inflow_budget=1.5)
    with torch.no_grad():  # request large inflow everywhere
        s, e = module.input_proj._slices["inflow"]
        module.input_proj.biases["inflow"].fill_(5.0)
    x = torch.randn(4, 3, N * D, dtype=torch.float64, requires_grad=True)
    gain = None
    for _ in range(30):
        x, gain = module(x, gain)
        assert gain.max() <= 1.5 + 1e-9
    assert torch.isclose(gain.max(), torch.tensor(1.5, dtype=gain.dtype), atol=1e-6)
    x.square().sum().backward()


def test_requires_inflow_and_valid_budget():
    with pytest.raises(ValueError):
        make(projection_variant="two_sided")
    with pytest.raises(ValueError):
        make(inflow_budget=0.5)


def test_transition_bound_dominates_actual_transition_norm():
    module = make(refill=True)
    with torch.no_grad():  # move the damping and dt projections off their init
        for key in ("diss", "dt_diss", "conserv", "dt_conserv"):
            s, e = module.input_proj._slices[key]
            module.input_proj.fused.weight[s:e].normal_(std=0.05)
    projections = module.input_proj(torch.randn(256, N * D, dtype=torch.float64))
    T = module.compute_transition(projections).double()
    tau = module.transition_bound(projections).double()
    assert (torch.linalg.matrix_norm(T, ord=2)[:, None] <= tau + 1e-9).all()
    assert (tau < 1).all()


@pytest.mark.parametrize("depth", [24, 96, 400])
def test_refill_caps_products_and_recovers_budget(depth):
    """Damped random transitions, fresh random v each sublayer, full requested inflow."""
    module, batch, budget = make(refill=True, inflow_budget=2.0), 64, 2.0
    g = torch.Generator().manual_seed(3)
    gain = torch.ones(batch, 1, dtype=torch.float64)
    product = torch.eye(N, dtype=torch.float64).expand(batch, N, N)
    worst, refilled = 0.0, False
    for k in range(depth):
        skew = torch.randn(batch, N, N, dtype=torch.float64, generator=g) * 1.5
        rates = torch.rand(batch, N, dtype=torch.float64, generator=g) * 0.2
        T = torch.linalg.matrix_exp(skew - skew.mT - torch.diag_embed(rates))
        tau = torch.exp(-rates.amin(dim=-1, keepdim=True))
        v = torch.nn.functional.normalize(torch.randn(batch, N, dtype=torch.float64, generator=g), dim=-1)
        requested = torch.where(torch.rand(batch, 1, generator=g, dtype=torch.float64) < 0.5, 1.0, 0.0)
        prev = gain
        gamma, gain = clip_inflow(requested, gain, DECAY, budget, tau)
        refilled |= bool((gain < prev).any())
        product = module._inflow_transition(T, v, gamma) @ product
        worst = max(worst, torch.linalg.matrix_norm(product, ord=2).max().item())
        assert gain.max() <= budget + 1e-9
    assert worst <= budget + 1e-9
    assert refilled


def test_refill_requires_diagonal_dissipation():
    with pytest.raises(ValueError):
        make(generator_type="conservative_psd_diss", refill=True)
