import torch

# Matrix exponential via optimized Taylor polynomial T_18.
#
# Algorithm from:
#   Bader, Blanes, Casas, "Computing the Matrix Exponential with an
#   Optimized Taylor Polynomial Approximation", Mathematics 2019, 7, 1174.
#   https://doi.org/10.3390/math7121174
#
# Uses the k=5 product decomposition (equation 15):
#
#   A2 = A²
#   A3 = A2 · A
#   A6 = A3²
#   B1 = a01·I + a11·A + a21·A2 + a31·A3
#   B2 = b01·I + b11·A + b21·A2 + b31·A3 + b61·A6
#   B3 = b02·I + b12·A + b22·A2 + b32·A3 + b62·A6
#   B4 = b03·I + b13·A + b23·A2 + b33·A3 + b63·A6
#   B5 = b04·I + b14·A + b24·A2 + b34·A3 + b64·A6
#   A9 = B1·B5 + B4
#   T18(A) = B2 + (B3 + A9)·A9
#
# For float32 (u <= 2^-24), the norm threshold is theta_18 = 3.01 (Table 2).
# If ||A||_1 > theta_18, scale by s = ceil(log2(||A||_1 / theta_18)) and
# recover via repeated squaring: exp(A) = exp(A / 2^s)^(2^s).

# B1 = a01·I + a11·A + a21·A2 + a31·A3
_a01 = 0.0
_a11 = -0.10036558103014462
_a21 = -0.008029246482411570
_a31 = -0.000892138498045730

# B2 = b01·I + b11·A + b21·A2 + b31·A3 + b61·A6
_b01 = 0.0
_b11 = 0.39784974949645076
_b21 = 1.36783778460411719
_b31 = 0.49828962252538268
_b61 = -0.00063789819459247233

# B3 = b02·I + b12·A + b22·A2 + b32·A3 + b62·A6
_b02 = -10.96763960529620626
_b12 = 1.68015813878906197
_b22 = 0.05717798464788655
_b32 = -0.00698210122488052
_b62 = 0.00003349750170860705

# B4 = b03·I + b13·A + b23·A2 + b33·A3 + b63·A6
_b03 = -0.09043168323908106
_b13 = -0.06764045190713819
_b23 = 0.06759613017740597
_b33 = 0.02955525704293155
_b63 = -0.00001391802575160607

# B5 = b04·I + b14·A + b24·A2 + b34·A3 + b64·A6
_b04 = 0.0
_b14 = 0.0
_b24 = -0.09233646193671186
_b34 = -0.01693649390020817
_b64 = -0.00001400867981820361

# Scaling threshold for float32 (Table 2, u <= 2^-24, m=18)
_THETA_18_F32 = 3.01


@torch.compile
def expm_t18(A: torch.Tensor) -> torch.Tensor:
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
        scale = 2.0 ** s
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
    for i in range(8):
        T_18 = torch.where(s > i, T_18 @ T_18, T_18)

    return T_18.to(original_dtype)
