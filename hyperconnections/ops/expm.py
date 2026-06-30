import torch

# Matrix exponential via optimized Taylor polynomial T_18.
# Algorithm: Bader, Blanes, Casas 2019 — doi.org/10.3390/math7121174
# Polynomial structure and constants live in numbers.py (shared with
# expm_block.py and expm_triton.py).

from .numbers import (
    _a01, _a11, _a21, _a31,
    _b01, _b11, _b21, _b31, _b61,
    _b02, _b12, _b22, _b32, _b62,
    _b03, _b13, _b23, _b33, _b63,
    _b04, _b14, _b24, _b34, _b64,
    _THETA_18_F32,
)


def expm_t18(A: torch.Tensor, S: int = 8) -> torch.Tensor:
    """Compute the matrix exponential of A using the T_18 Taylor approximation.

    Args:
        A: (..., n, n) float32 tensor.

    Returns:
        exp(A) with the same shape and dtype as A.
    """
    original_dtype = A.dtype
    A = A.to(torch.float32)

    ### Scaling: s = ceil(log2(||A||_1 / theta_18)), clamped to 0 if no scaling needed.
    ### Kept as float32 so the comparison (s > i) and division stay on-device.
    with torch.no_grad():
        A_norm = torch.linalg.matrix_norm(A, ord=1).max().clamp_min(_THETA_18_F32)
        s = torch.ceil(torch.log2(A_norm / _THETA_18_F32)).clamp(min=0)
        scale = 2.0**s
    A = A / scale

    eye = torch.eye(A.shape[-1], dtype=torch.float32, device=A.device)
    A_2 = A @ A
    A_3 = A_2 @ A
    A_6 = A_3 @ A_3

    B_1 = _a01 * eye + _a11 * A + _a21 * A_2 + _a31 * A_3
    B_2 = _b01 * eye + _b11 * A + _b21 * A_2 + _b31 * A_3 + _b61 * A_6
    B_3 = _b02 * eye + _b12 * A + _b22 * A_2 + _b32 * A_3 + _b62 * A_6
    B_4 = _b03 * eye + _b13 * A + _b23 * A_2 + _b33 * A_3 + _b63 * A_6
    B_5 = _b04 * eye + _b14 * A + _b24 * A_2 + _b34 * A_3 + _b64 * A_6

    A_9 = B_1 @ B_5 + B_4
    T_18 = B_2 + (B_3 + A_9) @ A_9

    ### Unrolled repeated squaring (max 8 doublings covers ||A||_1 up to ~386).
    ### Gated by torch.where to keep a static graph for torch.compile.
    for i in range(S):
        T_18 = torch.where(s > i, T_18 @ T_18, T_18)

    return T_18.to(original_dtype)
