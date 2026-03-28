"""
Tests for hyperconnections/ops/stream_mix.py

Run on a CUDA machine with:
    pytest tests/test_ops.py -v

All tests skip automatically if no CUDA device is available.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

from hyperconnections.ops.stream_mix import stream_mix_add

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------

def ref_no_proj(Phi: torch.Tensor, x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """out = Phi @ x + Y"""
    return torch.bmm(Phi, x) + Y


def ref_proj(
    Phi: torch.Tensor,
    x: torch.Tensor,
    Y: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """out = Phi @ x + (v - Phi@v)(v^T x) + Y"""
    alpha = torch.einsum("bn, bnd -> bd", v, x)           # [B, D]
    Phi_v = torch.bmm(Phi, v.unsqueeze(-1)).squeeze(-1)   # [B, N]
    correction = (v - Phi_v).unsqueeze(2) * alpha.unsqueeze(1)  # [B, N, D]
    return torch.bmm(Phi, x) + correction + Y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(B, N, D, dtype=torch.float32, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    Phi = torch.randn(B, N, N, device=DEVICE, dtype=dtype)
    x   = torch.randn(B, N, D, device=DEVICE, dtype=dtype)
    Y   = torch.randn(B, N, D, device=DEVICE, dtype=dtype)
    if requires_grad:
        Phi.requires_grad_(True)
        x.requires_grad_(True)
        Y.requires_grad_(True)
    return Phi, x, Y


def make_v(B, N, dtype=torch.float32, seed=7):
    torch.manual_seed(seed)
    v = torch.randn(B, N, device=DEVICE, dtype=dtype)
    return torch.nn.functional.normalize(v, dim=-1)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestShapes:
    @pytest.mark.parametrize("B,N,D", [(2, 4, 64), (1, 8, 128), (4, 2, 32)])
    def test_no_proj_shape(self, B, N, D):
        Phi, x, Y = make_inputs(B, N, D)
        assert stream_mix_add(Phi, x, Y).shape == (B, N, D)

    @pytest.mark.parametrize("B,N,D", [(2, 4, 64), (1, 8, 128)])
    def test_proj_shape(self, B, N, D):
        Phi, x, Y = make_inputs(B, N, D)
        v = make_v(B, N)
        assert stream_mix_add(Phi, x, Y, v=v).shape == (B, N, D)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_output_dtype_preserved(self, dtype):
        Phi, x, Y = make_inputs(2, 4, 64, dtype=dtype)
        assert stream_mix_add(Phi, x, Y).dtype == dtype


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

class TestForwardCorrectness:
    @pytest.mark.parametrize("B,N,D", [
        (1, 4, 64),
        (2, 8, 128),
        (4, 4, 256),
        (3, 6, 48),   # D not a power of 2
    ])
    def test_no_proj_fp32(self, B, N, D):
        Phi, x, Y = make_inputs(B, N, D)
        got      = stream_mix_add(Phi, x, Y)
        expected = ref_no_proj(Phi, x, Y)
        assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4), \
            f"max diff = {(got - expected).abs().max().item():.2e}"

    @pytest.mark.parametrize("B,N,D", [(2, 4, 128), (1, 8, 64)])
    def test_no_proj_fp16(self, B, N, D):
        Phi, x, Y = make_inputs(B, N, D, dtype=torch.float16)
        got      = stream_mix_add(Phi, x, Y).float()
        expected = ref_no_proj(Phi.float(), x.float(), Y.float())
        assert torch.allclose(got, expected, atol=1e-2, rtol=1e-2), \
            f"max diff = {(got - expected).abs().max().item():.2e}"

    @pytest.mark.parametrize("B,N,D", [
        (1, 4, 64),
        (2, 8, 128),
        (3, 4, 48),
    ])
    def test_proj_fp32(self, B, N, D):
        Phi, x, Y = make_inputs(B, N, D)
        v = make_v(B, N)
        got      = stream_mix_add(Phi, x, Y, v=v)
        expected = ref_proj(Phi, x, Y, v)
        assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4), \
            f"max diff = {(got - expected).abs().max().item():.2e}"

    def test_identity_phi_proj_equals_no_proj(self):
        """With identity Phi: (v - Phi@v) = 0, so proj must equal no-proj."""
        B, N, D = 2, 4, 64
        _, x, Y = make_inputs(B, N, D)
        Phi_id = torch.eye(N, device=DEVICE).unsqueeze(0).expand(B, -1, -1).contiguous()
        v = make_v(B, N)
        got_proj   = stream_mix_add(Phi_id, x, Y, v=v)
        got_noproj = stream_mix_add(Phi_id, x, Y)
        assert torch.allclose(got_proj, got_noproj, atol=1e-5)


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------

class TestBackwardCorrectness:
    """Compare Triton gradients against PyTorch autograd on the reference."""

    def _grads(self, B, N, D, use_proj):
        Phi_base, x_base, Y_base = make_inputs(B, N, D)
        v = make_v(B, N) if use_proj else None

        # Triton path
        Phi_t = Phi_base.detach().requires_grad_(True)
        x_t   = x_base.detach().requires_grad_(True)
        Y_t   = Y_base.detach().requires_grad_(True)
        stream_mix_add(Phi_t, x_t, Y_t, v=v).sum().backward()

        # Reference path
        Phi_r = Phi_base.detach().requires_grad_(True)
        x_r   = x_base.detach().requires_grad_(True)
        Y_r   = Y_base.detach().requires_grad_(True)
        fn = ref_proj if use_proj else ref_no_proj
        args = (Phi_r, x_r, Y_r, v) if use_proj else (Phi_r, x_r, Y_r)
        fn(*args).sum().backward()

        return (Phi_t.grad, x_t.grad, Y_t.grad), (Phi_r.grad, x_r.grad, Y_r.grad)

    @pytest.mark.parametrize("B,N,D", [(1, 4, 64), (2, 8, 128), (3, 4, 48)])
    def test_grad_x_no_proj(self, B, N, D):
        (_, gx_t, _), (_, gx_r, _) = self._grads(B, N, D, use_proj=False)
        assert torch.allclose(gx_t, gx_r, atol=1e-4, rtol=1e-4), \
            f"grad_x max diff = {(gx_t - gx_r).abs().max().item():.2e}"

    @pytest.mark.parametrize("B,N,D", [(1, 4, 64), (2, 8, 128), (3, 4, 48)])
    def test_grad_Phi_no_proj(self, B, N, D):
        (gP_t, _, _), (gP_r, _, _) = self._grads(B, N, D, use_proj=False)
        assert torch.allclose(gP_t, gP_r, atol=1e-4, rtol=1e-4), \
            f"grad_Phi max diff = {(gP_t - gP_r).abs().max().item():.2e}"

    @pytest.mark.parametrize("B,N,D", [(1, 4, 64), (2, 8, 128)])
    def test_grad_Y_is_identity(self, B, N, D):
        (_, _, gY_t), (_, _, gY_r) = self._grads(B, N, D, use_proj=False)
        assert torch.allclose(gY_t, gY_r, atol=1e-6), "grad_Y must be identity"

    @pytest.mark.parametrize("B,N,D", [(1, 4, 64), (2, 8, 128)])
    def test_grad_x_proj(self, B, N, D):
        (_, gx_t, _), (_, gx_r, _) = self._grads(B, N, D, use_proj=True)
        assert torch.allclose(gx_t, gx_r, atol=1e-4, rtol=1e-4), \
            f"grad_x (proj) max diff = {(gx_t - gx_r).abs().max().item():.2e}"

    @pytest.mark.parametrize("B,N,D", [(1, 4, 64), (2, 8, 128)])
    def test_grad_Phi_proj(self, B, N, D):
        (gP_t, _, _), (gP_r, _, _) = self._grads(B, N, D, use_proj=True)
        assert torch.allclose(gP_t, gP_r, atol=1e-4, rtol=1e-4), \
            f"grad_Phi (proj) max diff = {(gP_t - gP_r).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.parametrize("D", [1, 7, 33, 100, 500])
    def test_odd_D(self, D):
        """Mask logic must handle D values that are not multiples of BLOCK_D."""
        Phi, x, Y = make_inputs(1, 4, D)
        got = stream_mix_add(Phi, x, Y)
        assert torch.allclose(got, ref_no_proj(Phi, x, Y), atol=1e-4), f"D={D} failed"

    def test_single_stream(self):
        """N=1: Phi is a 1×1 matrix."""
        Phi, x, Y = make_inputs(3, 1, 64)
        got = stream_mix_add(Phi, x, Y)
        assert torch.allclose(got, ref_no_proj(Phi, x, Y), atol=1e-5)

    def test_cpu_raises(self):
        Phi = torch.randn(2, 4, 4)
        x   = torch.randn(2, 4, 64)
        Y   = torch.randn(2, 4, 64)
        with pytest.raises(RuntimeError, match="CUDA"):
            stream_mix_add(Phi, x, Y)
