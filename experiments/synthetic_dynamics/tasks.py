"""Online-sampled synthetic stream-dynamics tasks."""

from dataclasses import dataclass

import torch


def relative_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-example squared error divided by target energy."""
    error = (prediction - target).square().sum(dim=(1, 2))
    energy = target.square().sum(dim=(1, 2)).clamp_min(1e-8)
    return (error / energy).mean()


def _rotation(n: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(n, n, generator=generator))
    q = q * torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    if torch.linalg.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


@dataclass
class SyntheticTask:
    """One fixed task with fresh Gaussian states sampled for every batch."""

    name: str
    n: int
    d: int
    depth: int
    transform: torch.Tensor | None = None
    projector: torch.Tensor | None = None
    noise_projector: torch.Tensor | None = None

    @classmethod
    def create(cls, name: str, n: int, d: int, depth: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        if name == "rotation":
            transform = _rotation(n, generator)
            return cls(name, n, d, depth, transform=transform)
        if name == "permutation":
            indices = torch.roll(torch.arange(n), -1)
            transform = torch.zeros(n, n)
            transform[torch.arange(n), indices] = 1.0
            return cls(name, n, d, depth, transform=transform)
        if name == "filtering":
            if n < 3:
                raise ValueError("filtering requires at least three streams")
            address = torch.zeros(n)
            address[0] = 1.0
            other_mean = torch.zeros(n)
            other_mean[1:] = 1.0 / (n - 1) ** 0.5
            signal_projector = torch.outer(address, address)
            noise_projector = (
                torch.eye(n)
                - signal_projector
                - torch.outer(other_mean, other_mean)
            )
            return cls(
                name,
                n,
                d,
                depth,
                projector=signal_projector,
                noise_projector=noise_projector,
            )
        if name == "preservation":
            return cls(name, n, d, depth)
        raise ValueError(f"unknown task: {name}")

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        state = torch.randn(batch_size, self.n, self.d, generator=generator)

        if self.name == "preservation":
            return state, state, None
        if self.transform is not None:
            target = torch.einsum("ij,bjd->bid", self.transform, state)
            return state, target, None

        signal = torch.einsum("ij,bjd->bid", self.projector, state)
        noise = torch.randn(
            batch_size, self.depth, self.n, self.d, generator=generator
        )
        noise = torch.einsum("ij,bljd->blid", self.noise_projector, noise)
        if self.depth > 1:
            noise[:, :-1] /= (self.depth - 1) ** 0.5
        noise[:, -1] = 0.0
        return signal, signal, noise

    def metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        noise: torch.Tensor | None,
    ) -> dict[str, float]:
        metrics = {"error": relative_error(prediction, target).item()}
        if self.projector is None:
            return metrics

        projector = self.projector.to(prediction)
        signal = torch.einsum("ij,bjd->bid", projector, prediction)
        nuisance = prediction - signal
        signal_error = relative_error(signal, target).item()
        noise_energy = noise.square().sum(dim=(1, 2, 3)).clamp_min(1e-8)
        nuisance_error = (
            nuisance.square().sum(dim=(1, 2)) / noise_energy
        ).mean()
        metrics.update(
            signal_error=signal_error,
            nuisance_error=nuisance_error.item(),
        )
        return metrics
