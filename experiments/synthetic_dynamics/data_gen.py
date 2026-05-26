"""Utilities for generating synthetic signals and transformation matrices."""

import numpy as np
import torch
from scipy.stats import ortho_group


def generate_memory_state(
    n_samples: int,
    n_streams: int,
    d: int,
    n_memories: int | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate matrix memory states using outer products H = sum_i k_i @ v_i^T.

    Args:
        n_samples: Number of samples to generate
        n_streams: Number of streams (n)
        d: Dimension of each value vector
        n_memories: Number of key-value pairs to store (default: n_streams)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (H, keys, values) where:
            H: Memory matrices of shape [n_samples, n_streams, d]
            keys: Key vectors of shape [n_samples, n_memories, n_streams]
            values: Value vectors of shape [n_samples, n_memories, d]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_memories = n_memories or n_streams

    # Generate random keys and values
    keys = torch.randn(n_samples, n_memories, n_streams)
    keys = keys / (torch.norm(keys, dim=-1, keepdim=True) + 1e-8)  # Normalize keys
    values = torch.randn(n_samples, n_memories, d)

    # Construct memory matrices: H = sum_i k_i @ v_i^T
    # keys: [B, n_memories, n]
    # values: [B, n_memories, d]
    # H: [B, n, d]
    H = torch.einsum('bkn,bkd->bnd', keys, values)

    return H, keys, values


def generate_orthogonal_matrix(n: int, seed: int | None = None) -> torch.Tensor:
    """Generate a random orthogonal matrix.

    Args:
        n: Matrix dimension
        seed: Random seed for reproducibility

    Returns:
        Orthogonal matrix of shape [n, n]
    """
    if seed is not None:
        np.random.seed(seed)

    Q = ortho_group.rvs(n)
    return torch.from_numpy(Q).float()


def generate_permutation_matrix(
    n: int,
    mode: str = "cyclic",
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a permutation matrix.

    Args:
        n: Matrix dimension
        mode: Type of permutation ('cyclic', 'random', 'reverse')
        seed: Random seed for reproducibility

    Returns:
        Permutation matrix of shape [n, n]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if mode == "cyclic":
        perm = torch.roll(torch.arange(n), -1)
    elif mode == "reverse":
        perm = torch.arange(n - 1, -1, -1)
    elif mode == "random":
        perm = torch.randperm(n)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    P = torch.zeros(n, n)
    P[torch.arange(n), perm] = 1
    return P


def generate_subspace_basis(
    n_samples: int,
    n_basis_vectors: int,
    n_streams: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate random basis vectors for a subspace.

    Args:
        n_samples: Number of samples
        n_basis_vectors: Number of basis vectors
        n_streams: Dimension of the stream space
        seed: Random seed

    Returns:
        Basis vectors of shape [n_samples, n_basis_vectors, n_streams]
    """
    if seed is not None:
        torch.manual_seed(seed)

    basis = torch.randn(n_samples, n_basis_vectors, n_streams)
    Q, _ = torch.linalg.qr(basis.mT)
    return Q.mT


def sample_keys_from_basis(
    basis: torch.Tensor,
    n_keys: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample keys as linear combinations of basis vectors.

    Args:
        basis: Basis vectors [n_samples, n_basis_vectors, n_streams]
        n_keys: Number of keys to sample
        seed: Random seed

    Returns:
        Keys of shape [n_samples, n_keys, n_streams]
    """
    n_samples, n_basis_vectors, n_streams = basis.shape

    if seed is not None:
        torch.manual_seed(seed)

    # Sample random coefficients
    coeffs = torch.randn(n_samples, n_keys, n_basis_vectors)

    # Keys = coefficients @ basis
    keys = torch.einsum('bmk,bkn->bmn', coeffs, basis)
    keys = keys / (torch.norm(keys, dim=-1, keepdim=True) + 1e-8)  # Normalize keys

    return keys
