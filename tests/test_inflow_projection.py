import math

import pytest
import torch
import torch.nn as nn

from hyperconnections.cghc import ContinuousGenHyperConnections

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


def make(**kw):
    torch.manual_seed(0)
    return ContinuousGenHyperConnections(
        n=N, m=1, input_dim=N * D, embed_dim=D, module=nn.Linear(D, D, bias=False), **{**KWARGS, **kw}
    ).double()


def random_case(batch, rotation=1.5, seed=1):
    g = torch.Generator().manual_seed(seed)
    skew = torch.randn(batch, N, N, dtype=torch.float64, generator=g) * rotation
    damp = torch.diag_embed(torch.rand(batch, N, dtype=torch.float64, generator=g) * 0.1)
    T = torch.linalg.matrix_exp(skew - skew.mT - damp)
    v = torch.nn.functional.normalize(torch.randn(batch, N, dtype=torch.float64, generator=g) + 0.5, dim=-1)
    return T, v


def gain(M):
    return torch.linalg.matrix_norm(M, ord=2)


def test_inflow_endpoints_match_two_sided_and_one_sided():
    T, v = random_case(64)
    P = v[:, :, None] * v[:, None, :]
    Q = torch.eye(N, dtype=torch.float64) - P
    module = make()
    torch.testing.assert_close(module._inflow_transition(T, v, 0.0), DECAY * P + Q @ T @ Q)
    torch.testing.assert_close(module._inflow_transition(T, v, 1.0), DECAY * P + T @ Q)


@pytest.mark.parametrize("gamma", [-1.0, -0.3, 0.05, 0.3, 1.0])
def test_per_sublayer_gain_bound_either_sign(gamma):
    T, v = random_case(4096)
    M = make()._inflow_transition(T, v, gamma)
    # The v-perp block alone can carry gain 1, so the bound is max(1, sqrt(d^2 + gamma^2)).
    assert gain(M).max() <= max(1.0, math.sqrt(DECAY**2 + gamma**2)) + 1e-9


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_small_inflow_never_amplifies(sign):
    T, v = random_case(4096)
    M = make()._inflow_transition(T, v, sign * math.sqrt(1 - DECAY**2))
    assert gain(M).max() <= 1.0 + 1e-9


@pytest.mark.parametrize("depth", [24, 96, 400])
def test_fixed_v_bounds_any_depth(depth):
    """With the same v at every sublayer, a stack stays below 1 + max|gamma| / (1 - d)."""
    T, _ = random_case(depth * 8)
    v = torch.full((depth * 8, N), 0.5, dtype=torch.float64)
    gammas = torch.rand(depth * 8, 1, dtype=torch.float64) * 2 - 1
    maps = make()._inflow_transition(T, v, gammas).reshape(8, depth, N, N)
    product = torch.eye(N, dtype=torch.float64).expand(8, N, N)
    worst = 0.0
    for k in range(depth):
        product = maps[:, k] @ product
        worst = max(worst, gain(product).max().item())
    assert worst <= 1.0 + 1.0 / (1.0 - DECAY) + 1e-6


def test_inflow_starts_at_zero_and_trains():
    module = make()
    projections = module.input_proj(torch.randn(16, N * D, dtype=torch.float64))
    torch.testing.assert_close(module.compute_inflow(projections), torch.zeros(16, 1, dtype=torch.float64))
    x = torch.randn(3, 5, N * D, dtype=torch.float64, requires_grad=True)
    module(x).square().sum().backward()
    start, end = module.input_proj._slices["inflow"]
    assert module.input_proj.fused.weight.grad[start:end].abs().sum() > 0
    assert module.input_proj.biases["inflow"].grad is not None


def test_inflow_capped_symmetrically():
    module = make(inflow_max=0.2)
    with torch.no_grad():
        s, e = module.input_proj._slices["inflow"]
        module.input_proj.fused.weight[s:e].normal_(std=1.0)
    gamma = module.compute_inflow(module.input_proj(torch.randn(256, N * D, dtype=torch.float64)))
    assert gamma.abs().max() <= 0.2 and gamma.min() < 0 < gamma.max()


@pytest.mark.parametrize("kw", [dict(projection_inflow=1.0), dict(projection_inflow=-0.6, inflow_max=0.5), dict(inflow_max=1.5)])
def test_inflow_validated(kw):
    with pytest.raises(ValueError):
        make(**kw)
