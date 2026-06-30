"""Synthetic tasks for testing hyperconnections."""

from .data_gen import (
    generate_memory_state,
    generate_orthogonal_matrix,
    generate_permutation_matrix,
)
from .datasets import (
    SignalPreservationDataset,
    SignalRotationDataset,
    SignalFilteringDataset,
)
from .models import (
    ZeroModule,
    IdentityModule,
    SimpleMLPModule,
    StreamDynamicsModel,
)
from .metrics import mse_loss

__all__ = [
    "generate_memory_state",
    "generate_orthogonal_matrix",
    "generate_permutation_matrix",
    "SignalPreservationDataset",
    "SignalRotationDataset",
    "SignalFilteringDataset",
    "ZeroModule",
    "IdentityModule",
    "SimpleMLPModule",
    "StreamDynamicsModel",
    "mse_loss",
]
