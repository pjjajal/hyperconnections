"""mixer_chunks: per-chunk CGHC / mHC mixing (each D/C feature chunk has its own mixer)."""
import copy

import pytest
import torch
from torch import nn

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.cghcf import ContinuousGenHyperConnectionsForced
from hyperconnections.mhc import ManifoldHyperConnections

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
    forced = ContinuousGenHyperConnectionsForced(n=4, m=1, input_dim=64, embed_dim=16,
                                                 module=nn.Identity(), use_triton=False, mixer_chunks=2)
    with pytest.raises(NotImplementedError):
        forced(torch.randn(1, 2, 64))


def make_mhc(chunks=1, **kwargs):
    torch.manual_seed(0)
    defaults = dict(n=4, m=1, input_dim=64, embed_dim=16, module=nn.Linear(16, 16))
    return ManifoldHyperConnections(**{**defaults, **kwargs}, mixer_chunks=chunks)


def perturbed_mhc(model):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return model


def test_mhc_one_chunk_is_the_default_model():
    a = make_mhc(chunks=1)
    b = make_mhc()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 64)
    assert torch.equal(a(x), b(x))
    assert a.stream_mixing.shape == (4, 4)


@pytest.mark.parametrize("kernels", [(), (2, 4)])
def test_mhc_identical_chunks_reproduce_unchunked(kernels):
    single = perturbed_mhc(make_mhc(chunks=1, writeback_kernel_sizes=kernels))
    chunked = make_mhc(chunks=4, writeback_kernel_sizes=kernels)
    params = dict(chunked.named_parameters())
    with torch.no_grad():
        for name, param in single.named_parameters():
            if name.startswith("proj_stream_mixing"):
                params[name].copy_(param.repeat(4, *[1] * (param.dim() - 1)))
            else:
                params[name].copy_(param.expand_as(params[name]))
    x = torch.randn(2, 3, 64)
    torch.testing.assert_close(chunked(x), single(x), rtol=1e-5, atol=1e-5)


def test_mhc_chunks_are_independent():
    model = perturbed_mhc(make_mhc(chunks=4))
    assert model.stream_mixing.shape == (4, 4, 4)
    assert model.proj_stream_mixing.out_features == 64
    x = torch.randn(2, 3, 64)
    before = model(x).reshape(2, 3, 4, 4, 4)
    with torch.no_grad():
        model.stream_mixing[2].add_(torch.randn(4, 4))
    after = model(x).reshape(2, 3, 4, 4, 4)
    changed = (after - before).abs().amax(dim=(0, 1, 2, 4))
    assert changed[2] > 1e-4
    assert torch.equal(changed[[0, 1, 3]], torch.zeros(3))
    model(x).square().sum().backward()
    assert model.stream_mixing.grad[1].abs().sum() > 0


def test_mhc_invalid_chunks():
    with pytest.raises(ValueError):
        make_mhc(chunks=3)
