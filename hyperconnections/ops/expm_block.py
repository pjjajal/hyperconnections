import torch

from .numbers import (
    _a01, _a11, _a21, _a31,
    _b01, _b11, _b21, _b31, _b61,
    _b02, _b12, _b22, _b32, _b62,
    _b03, _b13, _b23, _b33, _b63,
    _b04, _b14, _b24, _b34, _b64,
    _THETA_18_F32,
)


def _blk_mul(X, Y):
    """Multiply structured block matrices.

    X = (A, B, c) represents [[A, B], [0, cI]]
    Y = (D, E, f) represents [[D, E], [0, fI]]

    Product:
        [[A, B], [0, cI]] [[D, E], [0, fI]]
      = [[AD, AE + fB], [0, cf I]]
    """
    A, B, c = X
    D, E, f = Y
    return (
        A @ D,
        A @ E + f * B,
        c * f,
    )


def _blk_add(X, Y):
    """Add structured block matrices."""
    A, B, c = X
    D, E, f = Y
    return (
        A + D,
        B + E,
        c + f,
    )


def _blk_scale(alpha, X):
    """Scale a structured block matrix."""
    A, B, c = X
    return (
        alpha * A,
        alpha * B,
        alpha * c,
    )


def _blk_lincomb4(c0, X0, c1, X1, c2, X2, c3, X3):
    """Linear combination of four structured block matrices."""
    A0, B0, d0 = X0
    A1, B1, d1 = X1
    A2, B2, d2 = X2
    A3, B3, d3 = X3

    return (
        c0 * A0 + c1 * A1 + c2 * A2 + c3 * A3,
        c0 * B0 + c1 * B1 + c2 * B2 + c3 * B3,
        c0 * d0 + c1 * d1 + c2 * d2 + c3 * d3,
    )


def _blk_lincomb5(c0, X0, c1, X1, c2, X2, c3, X3, c4, X4):
    """Linear combination of five structured block matrices."""
    A0, B0, d0 = X0
    A1, B1, d1 = X1
    A2, B2, d2 = X2
    A3, B3, d3 = X3
    A4, B4, d4 = X4

    return (
        c0 * A0 + c1 * A1 + c2 * A2 + c3 * A3 + c4 * A4,
        c0 * B0 + c1 * B1 + c2 * B2 + c3 * B3 + c4 * B4,
        c0 * d0 + c1 * d1 + c2 * d2 + c3 * d3 + c4 * d4,
    )


@torch.compile(
    fullgraph=True,
    mode="max-autotune",
)
def expm_t18_augmented_sparse(A: torch.Tensor):
    """Structured T_18 exponential of [[A, I], [0, 0]].

    This computes the same thing as:

        Z = [[A, I],
             [0, 0]]

        EZ = expm_t18(Z)

        E   = EZ[..., :n, :n]
        psi = EZ[..., :n, n:]

    but without materializing the 2n x 2n augmented matrix.

    Args:
        A: (..., n, n) float32 tensor.

    Returns:
        E:   (..., n, n), approximation to exp(A)
        psi: (..., n, n), approximation to phi_1(A)
             where phi_1(A) = integral_0^1 exp(theta A) d theta.
    """

    # Same scaling logic as your expm_t18.
    #
    # For the augmented matrix Z = [[A, I], [0, 0]],
    # ||Z||_1 = max(||A||_1, 1).
    #
    # Since theta_18 = 3.01 > 1, your original scaling based on ||A||_1
    # is enough for the identity-forcing augmented block.
    with torch.no_grad():
        A_norm = torch.linalg.matrix_norm(A, ord=1).max().clamp_min(_THETA_18_F32)
        s = torch.ceil(torch.log2(A_norm / _THETA_18_F32)).clamp(min=0)
        scale = 2.0 ** s

    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    zero_mat = torch.zeros_like(eye)

    zero = torch.zeros((), dtype=A.dtype, device=A.device)
    one = torch.ones((), dtype=A.dtype, device=A.device)

    # Structured representation:
    #
    #   (X, Y, c) <=> [[X, Y], [0, cI]]
    #
    # The actual scaled augmented matrix is
    #
    #   Z_s = Z / scale = [[A / scale, I / scale], [0, 0]]
    #
    # Important: the top-right block must also be divided by scale.
    Z = (
        A / scale,
        eye / scale,
        zero,
    )

    Id = (
        eye,
        zero_mat,
        one,
    )

    # Structured powers of Z.
    Z_2 = _blk_mul(Z, Z)
    Z_3 = _blk_mul(Z_2, Z)
    Z_6 = _blk_mul(Z_3, Z_3)

    # Same T_18 factorization as your dense version, but every object is now
    # a structured block tuple.
    B_1 = _blk_lincomb4(
        _a01, Id,
        _a11, Z,
        _a21, Z_2,
        _a31, Z_3,
    )

    B_2 = _blk_lincomb5(
        _b01, Id,
        _b11, Z,
        _b21, Z_2,
        _b31, Z_3,
        _b61, Z_6,
    )

    B_3 = _blk_lincomb5(
        _b02, Id,
        _b12, Z,
        _b22, Z_2,
        _b32, Z_3,
        _b62, Z_6,
    )

    B_4 = _blk_lincomb5(
        _b03, Id,
        _b13, Z,
        _b23, Z_2,
        _b33, Z_3,
        _b63, Z_6,
    )

    B_5 = _blk_lincomb5(
        _b04, Id,
        _b14, Z,
        _b24, Z_2,
        _b34, Z_3,
        _b64, Z_6,
    )

    Z_9 = _blk_add(_blk_mul(B_1, B_5), B_4)
    T_18 = _blk_add(B_2, _blk_mul(_blk_add(B_3, Z_9), Z_9))

    E, psi, c = T_18

    # Repeated squaring of the structured block exponential.
    #
    # If X = [[E, psi], [0, cI]], then
    #
    #   X^2 = [[E^2, E psi + c psi], [0, c^2 I]].
    #
    # This is handled by _blk_mul.
    for i in range(8):
        E_sq, psi_sq, c_sq = _blk_mul((E, psi, c), (E, psi, c))

        cond = s > i
        E = torch.where(cond, E_sq, E)
        psi = torch.where(cond, psi_sq, psi)
        c = torch.where(cond, c_sq, c)

    return E, psi