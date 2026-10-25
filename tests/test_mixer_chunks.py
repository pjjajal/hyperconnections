"""mixer_chunks: per-chunk CGHC mixing (each D/C feature chunk has its own transition)."""
import copy

import pytest
import torch
from torch import nn

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.cghcf import ContinuousGenHyperConnectionsForced

CHUNKED = {"conserv", "dt_conserv", "diss", "dt_diss", "proj_v", "inflow"}


def make(chunks=1, **kwargs):
    torch.manual_seed(0)
    defaults = dict(n=4, m=1, input_dim=64, embed_dim=16, module=nn.Linear(16, 16), use_triton=False)
    return ContinuousGenHyperConnections(**{**defaults, **kwargs}, mixer_chunks=chunks)


def replicate(single, chunked, chunks):
    """Give every chunk of `chunked` a copy of `single`'s mixing parameters."""
    with torch.no_grad():
        src, dst = single.input_proj, chunked.input_proj
        for key, (s0, s1) in src._slices.items():
            d0, d1 = dst._slices[key]
            rows = src.fused.weight[s0:s1]
            dst.fused.weight[d0:d1] = rows.repeat(chunks, 1) if key in CHUNKED else rows
            if key in src.biases:
                dst.biases[key].copy_(src.biases[key].repeat(chunks) if key in CHUNKED else src.biases[key])
        for name, param in single.named_parameters():
            if name.startswith("input_proj"):
                continue
            target = dict(chunked.named_parameters())[name]
            target.copy_(param.expand_as(target) if target.dim() > param.dim() else param)
        for name, buf in single.named_buffers():
            dict(chunked.named_buffers())[name].copy_(buf)


def perturbed(model):
    with torch.no_grad():
        model.input_proj.fused.weight.normal_(std=0.2)
        for b in model.input_proj.biases.values():
            b.normal_(std=0.2)
    return model


GENERATORS = ["conservative", "psd_diss", "diagonal_diss", "laplacian", "conservative_psd_diss"]
VARIANTS = [("none", "one_sided"), ("mean", "one_sided"), ("v", "two_sided"), ("v", "inflow"),
            ("mean", "generator")]


def test_one_chunk_is_the_default_model():
    a, b = make(chunks=1), make()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 64)
    assert torch.equal(a(x), b(x))
    assert {k: v.shape for k, v in a.state_dict().items()} == {k: v.shape for k, v in b.state_dict().items()}


@pytest.mark.parametrize("generator_type", GENERATORS)
@pytest.mark.parametrize("projection,variant", VARIANTS)
@pytest.mark.parametrize("vec_dt", [False, True])
def test_identical_chunks_reproduce_unchunked(generator_type, projection, variant, vec_dt):
    kwargs = dict(generator_type=generator_type, projection=projection, projection_variant=variant,
                  vec_dt=vec_dt)
    if variant == "inflow":
        kwargs["projection_inflow"] = 0.3
    single = perturbed(make(chunks=1, **kwargs))
    chunked = make(chunks=4, **kwargs)
    chunked.module = copy.deepcopy(single.module)
    replicate(single, chunked, 4)
    x = torch.randn(2, 3, 64)
    torch.testing.assert_close(chunked(x), single(x), rtol=1e-5, atol=1e-5)


def test_chunks_are_independent():
    model = perturbed(make(chunks=4, generator_type="conservative_diag_diss", projection="v"))
    x = torch.randn(2, 3, 64)
    before = model(x).reshape(2, 3, 4, 4, 4)  # [B, T, stream, chunk, D/C]
    with torch.no_grad():
        model.conserv_A[1].add_(0.5 * torch.randn(4, 4))
    after = model(x).reshape(2, 3, 4, 4, 4)
    changed = (after - before).abs().amax(dim=(0, 1, 2, 4))
    assert changed[1] > 1e-4
    assert torch.equal(changed[[0, 2, 3]], torch.zeros(3))


def test_parameter_shapes_and_gradients():
    model = make(chunks=8, generator_type="conservative_psd_diss", projection="v", vec_dt=True)
    assert model.conserv_A.shape == (8, 4, 4)
    assert model.diss_A.shape == (8, 4, 4)
    assert model.log_dt_conserv.shape == (8, 4)
    model(torch.randn(2, 3, 64)).square().sum().backward()
    assert model.conserv_A.grad.abs().sum() > 0


def test_invalid_configurations():
    with pytest.raises(ValueError):
        make(chunks=3)  # does not divide block_size 16
    with pytest.raises(ValueError):
        make(chunks=2, stream_subdivisions=2)
    forced = ContinuousGenHyperConnectionsForced(n=4, m=1, input_dim=64, embed_dim=16,
                                                 module=nn.Identity(), use_triton=False, mixer_chunks=2)
    with pytest.raises(NotImplementedError):
        forced(torch.randn(1, 2, 64))
