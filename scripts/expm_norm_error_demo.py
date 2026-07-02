"""
Why the matrix-exponential error grows with ||A|| — a standalone demo.

Pure math, no kernel / hyperconnections code.  Needs only numpy (scipy used as the
reference if available, else a numpy eigendecomposition).  Reproduces the behaviour
seen on the 'lrg' (large-norm) expm test matrices: a scaling-and-squaring exp(A)
computed in low precision has error that scales with ||A||, via two compounding
mechanisms.

  (1) OUTPUT MAGNITUDE.  For non-normal A, ||exp(A)|| grows like e^{alpha(A)}
      (alpha = largest real part of an eigenvalue), up to ~e^{||A||}.  Floating point
      keeps a fixed *relative* precision u (fp32 ~ 6e-8, bf16 ~ 4e-3), so the absolute
      error is ~ u * ||exp(A)|| — it scales with the size of the answer itself.  A flat
      absolute tolerance is meaningless; score with atol*(1 + ||ref||).

  (2) SQUARING AMPLIFICATION.  exp(A) = ( exp(A / 2^s) )^(2^s),
      s = ceil(log2(||A||_1 / theta)).  Each squaring Y -> Y^2 doubles the relative
      error ( d(Y^2)/Y^2 = 2 dY/Y ), so the polynomial-stage error eps0 is blown up by
      ~ 2^s ~ ||A|| / theta.  Relative error grows ~ linearly with the norm.

A third, subtler effect is non-normality / the "hump": intermediate quantities can
dwarf exp(A), so cancellation loses digits.  The skew-symmetric (normal) sweep below
isolates (2): there ||exp(A)|| = 1, yet the error still climbs with s.
"""
import numpy as np

try:
    from scipy.linalg import expm as expm_reference            # fp64 ground truth
except ImportError:                                            # numpy-only fallback
    def expm_reference(A):
        """exp(A) via eigendecomposition in fp64 — independent of scaling/squaring."""
        w, V = np.linalg.eig(np.asarray(A, np.float64))
        return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real

THETA = 3.0   # scaling threshold (T18 uses ~3.01 for fp32)


def taylor_core(A, terms=18):
    """Truncated degree-`terms` Taylor exp(A); accurate only for ||A|| <~ THETA."""
    S = term = np.eye(A.shape[0], dtype=A.dtype)
    for k in range(1, terms + 1):
        term = term @ A / k          # term_k = A^k / k!, stays in A.dtype
        S = S + term
    return S


def expm_scaling_squaring(A, dtype=np.float32, terms=18):
    """exp(A): scale -> Taylor core -> square s times, all in `dtype`. T18-shaped."""
    A = A.astype(dtype)
    norm1 = float(np.max(np.abs(A).sum(axis=0)))         # ||A||_1
    s = int(np.ceil(np.log2(max(norm1 / THETA, 1.0))))
    X = taylor_core(A / (2.0 ** s), terms)
    for _ in range(s):
        X = X @ X
    return X, s


def make_matrix(n, norm1, kind="general", seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    if kind == "skew":                # normal => exp(A) orthogonal, ||exp||_2 = 1
        A = 0.5 * (A - A.T)
    return A * (norm1 / np.max(np.abs(A).sum(axis=0)))    # rescale to ||A||_1 = norm1


def sweep(kind, dtype=np.float32, n=8):
    print(f"\n=== {kind} matrices, working dtype = {np.dtype(dtype).name} ===")
    print(f"{'||A||_1':>8} {'s':>3} {'||expA||_inf':>13} {'abs_err':>11} {'rel_err':>11}")
    for norm1 in (0.5, 1, 2, 4, 8, 16, 32):
        A = make_matrix(n, norm1, kind=kind)
        ref = np.asarray(expm_reference(A.astype(np.float64)))
        got, s = expm_scaling_squaring(A, dtype=dtype)
        abs_err = float(np.max(np.abs(got.astype(np.float64) - ref)))
        mag = float(np.max(np.abs(ref)))
        print(f"{norm1:8.1f} {s:3d} {mag:13.3e} {abs_err:11.3e} {abs_err / mag:11.3e}")


if __name__ == "__main__":
    sweep("general")    # (1)+(2): magnitude blow-up AND squaring amplification
    sweep("skew")       # ||exp||=1, so abs_err == rel_err: pure squaring amplification
    # To watch the bf16 floor (~1e-2 relative), redo the math in torch.bfloat16.
