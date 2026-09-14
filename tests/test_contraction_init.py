import pytest
import torch
from hyperconnections.cghc import ContinuousGenHyperConnections


def make(**kwargs):
    return ContinuousGenHyperConnections(
        n=4, m=1, input_dim=32, embed_dim=8, module=torch.nn.Identity(),
        dt=.015, dt_min=.0001, dt_max=.4, use_triton=False,
        generator_type=kwargs.pop('generator_type', 'conservative_diag_diss'),
        sat_c=kwargs.pop('sat_c', None), **kwargs)


@pytest.mark.parametrize('vec_dt', [False, True])
@pytest.mark.parametrize('dt_max_diss', [None, .7, 1.])
@pytest.mark.parametrize('sat_c,target', [(None, .95), (2., .999)])
def test_nominal_transition(vec_dt, dt_max_diss, sat_c, target):
    hc = make(vec_dt=vec_dt, dt_max_diss=dt_max_diss, sat_c=sat_c, init_transition_contraction=target)
    projections = hc.input_proj(torch.zeros(2, 32))
    projections = projections._replace(**{k: torch.zeros_like(v) for k, v in projections._asdict().items() if v is not None})
    t = hc.compute_transition(projections)
    torch.testing.assert_close(torch.linalg.svdvals(t), torch.full((2, 4), target), atol=2e-6, rtol=2e-6)
    hc.init_weights()
    assert torch.all(hc.diss_diag == hc._diss_diag_init)


def test_default_and_invalid():
    assert torch.count_nonzero(make().diss_diag) == 0
    for target in [0., 1., -1., float('nan')]:
        with pytest.raises(ValueError):
            make(init_transition_contraction=target)
    with pytest.raises(ValueError, match='unreachable'):
        make(sat_c=2., init_transition_contraction=.95)
    with pytest.raises(ValueError, match='diagonal'):
        make(generator_type='psd_diss', init_transition_contraction=.95)


def test_saturated_initialization_without_gate():
    hc = make(sat_c=10., init_transition_contraction=.95)
    d = torch.nn.functional.softplus(hc.diss_diag)
    effective = .015 * d / (1+d/10.)
    torch.testing.assert_close(effective.neg().exp(), torch.full((4,), .95))


@pytest.mark.parametrize('vec_dt', [False, True])
def test_independent_dissipative_ceiling(vec_dt):
    hc = make(vec_dt=vec_dt, dt_max_diss=.7)
    p = hc.input_proj(torch.zeros(2, 32))
    p = p._replace(**{k: torch.zeros_like(v) for k, v in p._asdict().items() if v is not None})
    # Isolate each branch at its ceiling using identical fixed predictor values.
    p = p._replace(conserv=torch.zeros_like(p.conserv),
                   dt_conserv=torch.full_like(p.dt_conserv, 100.),
                   dt_diss=torch.full_like(p.dt_diss, 100.))
    with torch.no_grad():
        hc.conserv_A.zero_()
        hc.conserv_A[0, 1] = 1.
        hc.conserv_A[1, 0] = -1.
    a = hc.compute_generator(p)
    torch.testing.assert_close(a[:, 0, 1], torch.full((2,), .4))
    torch.testing.assert_close(a.diagonal(dim1=-2, dim2=-1),
                               torch.full((2, 4), -.7 * torch.log(torch.tensor(2.)).item()))


@pytest.mark.parametrize('ceiling', [0., .015, -.1, float('nan'), float('inf')])
def test_invalid_dissipative_ceiling(ceiling):
    with pytest.raises(ValueError, match='dt_max_diss'):
        make(dt_max_diss=ceiling)
