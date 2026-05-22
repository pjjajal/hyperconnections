"""Dataset classes for synthetic tasks."""

import torch
from torch.utils.data import Dataset
from typing import Literal

from .data_gen import (
    generate_memory_state,
    generate_orthogonal_matrix,
    generate_permutation_matrix,
    generate_subspace_basis,
    sample_keys_from_basis,
)


class SignalPreservationDataset(Dataset):
    """Signal Preservation Task.

    Generate memory states H_0 = sum_i k_i @ v_i^T and test preservation.
    Target: H_L = H_0
    """

    def __init__(
        self,
        n_samples: int,
        n_streams: int,
        d: int,
        n_memories: int | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            n_samples: Number of samples in the dataset
            n_streams: Number of streams (n)
            d: Dimension of each value vector
            n_memories: Number of key-value pairs (default: n_streams)
            seed: Random seed
        """
        self.n_samples = n_samples
        self.n_streams = n_streams
        self.d = d

        # Generate memory states using outer products
        H_0, keys, values = generate_memory_state(
            n_samples, n_streams, d, n_memories, seed
        )

        self.H_0 = H_0
        self.keys = keys
        self.values = values
        self.H_target = H_0.clone()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "input": self.H_0[idx],  # [n_streams, d]
            "target": self.H_target[idx],  # [n_streams, d]
        }


class SignalRotationDataset(Dataset):
    """Signal Permutation and Rotation Task.

    Generate memory states and apply transformation T.
    Target: H_L = T @ H_0

    Supports:
    - Permutation: T = P
    - Orthogonal rotation: T = Q
    """

    def __init__(
        self,
        n_samples: int,
        n_streams: int,
        d: int,
        n_memories: int | None = None,
        transform_type: Literal["permutation", "rotation"] = "permutation",
        permutation_mode: str = "cyclic",
        seed: int | None = None,
    ):
        """
        Args:
            n_samples: Number of samples
            n_streams: Number of streams (n)
            d: Dimension of each value vector
            n_memories: Number of key-value pairs (default: n_streams)
            transform_type: Type of transformation ('permutation' or 'rotation')
            permutation_mode: For permutation type ('cyclic', 'random', 'reverse')
            seed: Random seed
        """
        self.n_samples = n_samples
        self.n_streams = n_streams
        self.d = d
        self.transform_type = transform_type

        # Generate input memory states
        H_0, keys, values = generate_memory_state(
            n_samples, n_streams, d, n_memories, seed
        )
        self.H_0 = H_0
        self.keys = keys
        self.values = values

        # Generate transformation matrix
        if transform_type == "permutation":
            T = generate_permutation_matrix(n_streams, permutation_mode, seed)
        elif transform_type == "rotation":
            T = generate_orthogonal_matrix(n_streams, seed)
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")

        self.T = T

        # Compute target: H_L = T @ H_0
        self.H_target = torch.einsum("ij,bjd->bid", T, self.H_0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "input": self.H_0[idx],  # [n_streams, d]
            "target": self.H_target[idx],  # [n_streams, d]
        }


class SignalFilteringDataset(Dataset):
    """Signal Filtering Task.

    Signal: H_0 = sum_i k_signal_i @ v_signal_i^T
    Noise: noise[layer] = sum_j k_noise_j @ v_noise_j^T (per layer)

    Signal keys sampled from signal_basis, noise keys sampled from noise_basis.
    Goal: recover H_0 after noise injection at each layer.
    """

    def __init__(
        self,
        n_samples: int,
        n_streams: int,
        d: int,
        n_layers: int,
        n_signal_basis: int,
        n_signal_memories: int,
        n_noise_basis: int,
        n_noise_memories: int,
        noise_scale: float = 1.0,
        seed: int | None = None,
    ):
        """
        Args:
            n_samples: Number of samples
            n_streams: Number of streams (n)
            d: Dimension of values
            n_layers: Number of layers
            n_signal_basis: Number of signal basis vectors
            n_signal_memories: Number of signal key-value pairs
            n_noise_basis: Number of noise basis vectors
            n_noise_memories: Number of noise key-value pairs per layer
            noise_scale: Noise magnitude
            seed: Random seed
        """
        self.n_samples = n_samples
        self.n_streams = n_streams
        self.d = d
        self.n_layers = n_layers
        self.noise_scale = noise_scale

        # Generate signal basis
        signal_basis = generate_subspace_basis(
            n_samples, n_signal_basis, n_streams, seed
        )

        # Generate noise basis
        noise_basis = generate_subspace_basis(
            n_samples, n_noise_basis, n_streams, seed + 1000 if seed else None
        )

        # Sample signal keys from signal basis
        signal_keys = sample_keys_from_basis(signal_basis, n_signal_memories, seed + 10 if seed else None)

        # Generate signal values
        if seed is not None:
            torch.manual_seed(seed + 20)
        signal_values = torch.randn(n_samples, n_signal_memories, d)

        # Construct H_0 = sum_i signal_keys[i] @ signal_values[i]^T
        H_0 = torch.einsum('bkn,bkd->bnd', signal_keys, signal_values)

        self.H_0 = H_0
        self.signal_basis = signal_basis
        self.noise_basis = noise_basis

        # Generate noise per layer
        # noise[layer] = sum_j noise_keys[layer, j] @ noise_values[layer, j]^T
        # Scale by 1/sqrt(n_layers) so total noise energy is independent of depth
        noise = torch.zeros(n_samples, n_layers, n_streams, d)
        layer_noise_scale = noise_scale / (n_layers ** 0.5)

        for layer in range(n_layers):
            # Sample noise keys from noise basis (different per layer)
            noise_keys = sample_keys_from_basis(
                noise_basis, n_noise_memories, seed + 100 + layer if seed else None
            )

            # Generate noise values
            if seed is not None:
                torch.manual_seed(seed + 200 + layer)
            noise_values = torch.randn(n_samples, n_noise_memories, d)

            # Construct noise for this layer
            noise[:, layer] = torch.einsum('bkn,bkd->bnd', noise_keys, noise_values) * layer_noise_scale

        self.noise = noise
        self.H_target = H_0.clone()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "input": self.H_0[idx],  # [n_streams, d]
            "target": self.H_target[idx],  # [n_streams, d]
            "noise": self.noise[idx],  # [n_layers, n_streams, d]
            "signal_basis": self.signal_basis[idx],  # [n_signal_basis, n_streams]
            "noise_basis": self.noise_basis[idx],  # [n_noise_basis, n_streams]
        }
