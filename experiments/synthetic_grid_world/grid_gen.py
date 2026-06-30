from pathlib import Path

import torch
from torch.utils.data import Dataset
from typing import NamedTuple

from tqdm import tqdm

NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3


class Trajectory(NamedTuple):
    actions: torch.Tensor  # [T]
    observations: torch.Tensor  # [T + 1]
    positions: torch.Tensor  # [T + 1]
    final_position: torch.Tensor  # scalar


class Hardness(NamedTuple):
    candidate_counts: torch.Tensor
    max_candidates: int
    min_candidates: int
    min_candidates_before_final: int
    initial_candidates: int
    final_candidates: int
    final_unique: bool
    first_unique_step: int
    frac_ambiguous_steps: float
    num_ambiguous_steps: int


class Sample(NamedTuple):
    grid: torch.Tensor      # [n_rows, n_cols]
    trajectory: Trajectory
    hardness: Hardness
    belief_masks: torch.Tensor


def generate_grid(
    n_grids: int,
    n_rows: int,
    n_cols: int,
    n_colours: int,
    generator: torch.Generator | None = None,
):
    """Generate a grid world dataset.

    Args:
        n_grids: Number of grids to generate
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        n_colours: Number of possible colours for each cell
        trajectory_length: Length of the trajectory to generate for each grid
        generator: Torch random number generator

    Returns:
        A tensor of shape [n_grids, n_rows, n_cols] with colour indices
    """
    positions = torch.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    grid = torch.randint(0, n_colours, (n_grids, n_rows, n_cols), generator=generator)
    return grid, positions


def move_position(
    row: torch.Tensor, col: torch.Tensor, action: torch.Tensor, n_rows: int, n_cols: int
):
    """Move a position using circular boundary conditions."""
    if action.item() == NORTH:
        row = (row - 1) % n_rows
    elif action.item() == EAST:
        col = (col + 1) % n_cols
    elif action.item() == SOUTH:
        row = (row + 1) % n_rows
    elif action.item() == WEST:
        col = (col - 1) % n_cols
    else:
        raise ValueError(f"Unknown action: {action.item()}")

    return row, col


def move_belief_mask(mask: torch.Tensor, action: torch.Tensor):
    """Move a belief mask using circular boundary conditions.

    Args:
        mask: [n_rows, n_cols] boolean tensor.
        action: scalar tensor.

    Returns:
        shifted_mask: [n_rows, n_cols] boolean tensor.
    """
    a = action.item()

    if a == NORTH:
        return torch.roll(mask, shifts=-1, dims=0)
    elif a == EAST:
        return torch.roll(mask, shifts=1, dims=1)
    elif a == SOUTH:
        return torch.roll(mask, shifts=1, dims=0)
    elif a == WEST:
        return torch.roll(mask, shifts=-1, dims=1)
    else:
        raise ValueError(f"Unknown action: {a}")


def generate_trajectory(
    grid: torch.Tensor,
    trajectory_length: int,
    generator: torch.Generator | None = None,
):
    """Generate one trajectory on a fixed grid.

    Args:
        grid: [n_rows, n_cols] colour grid.
        trajectory_length: number of actions.
        generator: torch random generator.

    Returns:
        A dict containing true states, actions, and observations.
    """
    n_rows, n_cols = grid.shape

    actions = torch.randint(
        low=0,
        high=4,
        size=(trajectory_length,),
        generator=generator,
    )

    rows = torch.empty(trajectory_length + 1, dtype=torch.long)
    cols = torch.empty(trajectory_length + 1, dtype=torch.long)

    rows[0] = torch.randint(0, n_rows, size=(), generator=generator)
    cols[0] = torch.randint(0, n_cols, size=(), generator=generator)

    for t in range(trajectory_length):
        rows[t + 1], cols[t + 1] = move_position(
            rows[t],
            cols[t],
            actions[t],
            n_rows,
            n_cols,
        )

    observations = grid[rows, cols]
    flat_positions = rows * n_cols + cols

    return Trajectory(
        actions=actions,
        observations=observations,
        positions=flat_positions,
        final_position=flat_positions[-1],
    )


def compute_belief_trajectory(
    grid: torch.Tensor,
    actions: torch.Tensor,
    observations: torch.Tensor,
):
    """Compute exact belief masks for a trajectory.

    Belief B_t is the set of possible current locations after observing o_t.

    Args:
        grid: [n_rows, n_cols]
        actions: [T]
        observations: [T + 1]

    Returns:
        belief_masks: [T + 1, n_rows, n_cols] boolean tensor
        candidate_counts: [T + 1] long tensor
    """
    trajectory_length = actions.shape[0]
    n_rows, n_cols = grid.shape

    belief_masks = torch.empty(
        trajectory_length + 1,
        n_rows,
        n_cols,
        dtype=torch.bool,
    )

    # Initial belief: all cells matching the initial observation.
    belief = grid == observations[0]
    belief_masks[0] = belief

    for t in range(trajectory_length):
        # Predict step: move every candidate according to the known action.
        belief = move_belief_mask(belief, actions[t])

        # Correction step: keep only cells whose colour matches the new observation.
        belief = belief & (grid == observations[t + 1])

        belief_masks[t + 1] = belief

    candidate_counts = belief_masks.flatten(1).sum(dim=1)

    return belief_masks, candidate_counts


def compute_hardness(candidate_counts: torch.Tensor):

    counts = candidate_counts
    trajectory_length = counts.shape[0] - 1

    unique_steps = torch.nonzero(counts == 1, as_tuple=False).flatten()
    if len(unique_steps) > 0:
        t_unique = int(unique_steps[0].item())
    else:
        t_unique = -1

    if trajectory_length > 0:
        min_cadidates_before_final = int(counts[:-1].min().item())
    else:
        min_cadidates_before_final = int(counts[0].item())

    return Hardness(
        candidate_counts=counts,
        max_candidates=int(counts.max().item()),
        min_candidates=int(counts.min().item()),
        min_candidates_before_final=min_cadidates_before_final,
        initial_candidates=int(counts[0].item()),
        final_candidates=int(counts[-1].item()),
        final_unique=bool(counts[-1].item() == 1),
        first_unique_step=t_unique,
        frac_ambiguous_steps=float((counts > 1).float().mean().item()),
        num_ambiguous_steps=int((counts > 1).sum().item()),
    )


def hardness_check(
    candidate_counts: torch.Tensor,
    ambiguous_threshold: float = 0.75,
    final_unique: bool = True,
):
    frac_ambiguous_steps = (candidate_counts > 1).float().mean().item()
    long_enough = False
    if frac_ambiguous_steps >= ambiguous_threshold:
        long_enough = True

    if final_unique:
        final_unique_ok = bool(candidate_counts[-1].item() == 1)
    else:
        final_unique_ok = True

    return long_enough and final_unique_ok


class GridWorldDataset(torch.utils.data.Dataset):
    """
    Dataset of synthetic grid worlds. Each sample has a unique randomly-generated grid.
    """

    def __init__(
        self,
        n_samples,
        n_rows,
        n_cols,
        n_colours,
        trajectory_length,
        ambiguous_threshold: float = 0.75,
        final_unique: bool = True,
        seed: int = 42,
        cache_path: str | Path | None = None,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_colours = n_colours
        self.trajectory_length = trajectory_length
        self.ambiguous_threshold = ambiguous_threshold
        self.final_unique = final_unique

        if cache_path is not None and Path(cache_path).exists():
            self.samples = torch.load(cache_path, weights_only=False)
            print(f"Loaded {len(self.samples)} samples from {cache_path}")
            return

        self.rng = torch.Generator().manual_seed(seed)
        self.samples = []
        self._generate_samples(n_samples)

        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.samples, cache_path)
            print(f"Saved {len(self.samples)} samples to {cache_path}")

    def _generate_samples(self, n_samples: int):
        pbar = tqdm(total=n_samples, desc="Generating samples")
        while len(self.samples) < n_samples:
            grid = torch.randint(
                0, self.n_colours, (self.n_rows, self.n_cols), generator=self.rng
            )
            trajectory = generate_trajectory(grid, self.trajectory_length, self.rng)
            belief_masks, candidate_counts = compute_belief_trajectory(
                grid, trajectory.actions, trajectory.observations
            )
            if hardness_check(candidate_counts, self.ambiguous_threshold, self.final_unique):
                hardness = compute_hardness(candidate_counts)
                self.samples.append(
                    Sample(
                        grid=grid,
                        trajectory=trajectory,
                        hardness=hardness,
                        belief_masks=belief_masks,
                    )
                )
                pbar.update(1)
        pbar.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    grids, positions = generate_grid(5, 4, 4, 2)
    print(grids.shape)  # [5, 3, 3]
    print(positions.shape)  # [5, 3, 3]
    trajectory_generator = torch.Generator().manual_seed(42)
    trajectory = generate_trajectory(grids[0], 12, trajectory_generator)
    print(trajectory.actions.shape)  # [4]
    print(trajectory.observations.shape)  # [5]
    belief_masks, candidate_counts = compute_belief_trajectory(
        grids[0], trajectory.actions, trajectory.observations
    )
    print(belief_masks)  # [5, 3, 3]
    print(hardness := compute_hardness(candidate_counts))
