"""Smoke tests for ContinuousGenHyperConnectionsForced (cghcf.py).

Structural checks only: correct output shape, finite values, and backward runs.
These serve as a baseline before and after refactoring.
"""

import pytest
import torch
import torch.nn as nn

from hyperconnections.cghcf import ContinuousGenHyperConnectionsForced
from tests.conftest import IdentityModule, should_use_triton

CONFIGS = [
    (2, 1, 8),
    (4, 2, 8),
    (6, 3, 12),
]

ALL_GENERATOR_TYPES = [
    "conservative",
    "psd_diss",
    "diagonal_diss",
    "laplacian",
    "conservative_diag_diss",
    "conservative_psd_diss",
    "conservative_laplacian",
]


def make_cghcf(n, m, embed_dim, device="cpu", use_triton=None, **kwargs) -> ContinuousGenHyperConnectionsForced:
    input_dim = (n * embed_dim) // m
    # Toggle the custom-kernel flag to match the device unless explicitly set:
    # Triton kernels need CUDA tensors, so a CPU model must run the eager path.
    if use_triton is None:
        use_triton = should_use_triton(device)
    return ContinuousGenHyperConnectionsForced(
        n=n,
        m=m,
        input_dim=input_dim,
        embed_dim=embed_dim,
        module=IdentityModule(),
        use_triton=use_triton,
        **kwargs,
    ).to(device)


@pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
@pytest.mark.parametrize("generator_type", ALL_GENERATOR_TYPES)
def test_forward_shape_and_finite(n, m, embed_dim, generator_type):
    torch.manual_seed(0)
    model = make_cghcf(n, m, embed_dim, generator_type=generator_type)
    input_dim = (n * embed_dim) // m
    x = torch.randn(2, 4, input_dim)
    out = model(x)
    assert out.shape == x.shape
    assert out.isfinite().all()


@pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
def test_backward_runs(n, m, embed_dim):
    torch.manual_seed(0)
    model = make_cghcf(n, m, embed_dim)
    input_dim = (n * embed_dim) // m
    x = torch.randn(2, 4, input_dim, requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()


@pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
def test_shortconv_forward(n, m, embed_dim):
    torch.manual_seed(0)
    model = make_cghcf(n, m, embed_dim, shortconv_kernel_size=4)
    input_dim = (n * embed_dim) // m
    x = torch.randn(2, 4, input_dim)
    out = model(x)
    assert out.shape == x.shape
    assert out.isfinite().all()
