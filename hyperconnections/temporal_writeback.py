from collections.abc import Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalWriteback(nn.Module):
    """Add causal, multi-scale temporal components to a module output.

    The original output is the first component. Each remaining component is
    produced by a causal depthwise convolution and, by default, orthogonalized
    against the preceding components along the feature dimension.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: Sequence[int],
        orthogonalize: bool = True,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("TemporalWriteback requires at least one kernel size")

        self.dim = dim
        self.num_components = len(kernel_sizes) + 1
        self.orthogonalize = orthogonalize
        self.kernel_sizes = tuple(kernel_sizes)
        self.weights = nn.ParameterList(
            nn.Parameter(torch.empty(dim, kernel_size))
            for kernel_size in self.kernel_sizes
        )
        for weight, kernel_size in zip(self.weights, self.kernel_sizes):
            bound = 1.0 / math.sqrt(kernel_size)
            nn.init.uniform_(weight, -bound, bound)

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim < 3:
            raise ValueError(
                "Temporal write-back requires a sequence dimension; "
                f"expected [..., sequence, {self.dim}], got {tuple(output.shape)}"
            )
        if output.shape[-1] != self.dim:
            raise ValueError(
                f"Expected output dimension {self.dim}, got {output.shape[-1]}"
            )

        leading_shape = output.shape[:-2]
        sequence_length = output.shape[-2]
        output_3d = output.reshape(-1, sequence_length, self.dim)
        channels_first = output_3d.transpose(1, 2)
        components = [output_3d]
        for weight, kernel_size in zip(self.weights, self.kernel_sizes):
            padded = F.pad(channels_first, (kernel_size - 1, 0))
            component = F.conv1d(
                padded,
                weight.unsqueeze(1),
                groups=self.dim,
            ).transpose(1, 2)
            components.append(component)

        if self.orthogonalize:
            components = self._orthogonalize(components)

        stacked = torch.stack(components, dim=-2).to(output.dtype)
        return stacked.reshape(
            *leading_shape,
            sequence_length,
            self.num_components,
            self.dim,
        )

    @staticmethod
    def _orthogonalize(components: list[torch.Tensor]) -> list[torch.Tensor]:
        work_dtype = (
            torch.float32
            if components[0].dtype in {torch.float16, torch.bfloat16}
            else components[0].dtype
        )
        basis: list[torch.Tensor] = []

        for component in components:
            vector = component.to(work_dtype)
            for previous in basis:
                denominator = previous.square().sum(dim=-1, keepdim=True)
                denominator = denominator.clamp_min(torch.finfo(work_dtype).eps)
                coefficient = (vector * previous).sum(dim=-1, keepdim=True)
                vector = vector - coefficient / denominator * previous
            basis.append(vector)

        return basis
