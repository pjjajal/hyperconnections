"""Generate and cache grid-world datasets at four hardness levels.

Each level is defined by (n_colours, trajectory_length, ambiguous_threshold).
Lower n_colours + longer trajectory + higher threshold = harder.

Usage:
    uv run --extra experiments python -m experiments.synthetic_grid_world.make_datasets
    uv run --extra experiments python -m experiments.synthetic_grid_world.make_datasets --stats-only
"""

import argparse
import json
import torch
from pathlib import Path

from experiments.synthetic_grid_world.grid_gen import GridWorldDataset

DATASET_DIR = Path("experiments/synthetic_grid_world/datasets")

# Grid geometry fixed across all levels — only difficulty knobs change.
GRID = dict(n_rows=6, n_cols=6, final_unique=True)

N_TRAIN = 100_000
N_VAL   = 10_000

HARDNESS_LEVELS = [
    dict(
        name="h1_easy",
        n_colours=6,
        trajectory_length=6,
        ambiguous_threshold=0.25,
    ),
    dict(
        name="h2_medium",
        n_colours=4,
        trajectory_length=8,
        ambiguous_threshold=0.50,
    ),
    dict(
        name="h3_hard",
        n_colours=3,
        trajectory_length=10,
        ambiguous_threshold=0.65,
    ),
    dict(
        name="h4_veryhard",
        n_colours=2,
        trajectory_length=12,
        ambiguous_threshold=0.80,
    ),
]


def dataset_stats(samples) -> dict:
    frac_amb   = [s.hardness.frac_ambiguous_steps for s in samples]
    first_uniq = [s.hardness.first_unique_step     for s in samples]
    init_cands = [s.hardness.initial_candidates    for s in samples]
    n = len(samples)
    return dict(
        mean_frac_ambiguous   = sum(frac_amb)   / n,
        mean_first_unique_step= sum(first_uniq) / n,
        mean_initial_candidates=sum(init_cands) / n,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Load existing cached datasets and recompute stats without regenerating.",
    )
    args = parser.parse_args()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for level in HARDNESS_LEVELS:
        name = level["name"]
        kwargs = {k: v for k, v in level.items() if k != "name"}

        print(f"\n{'─'*60}")
        print(f"  {name}  |  n_colours={kwargs['n_colours']}  "
              f"traj_len={kwargs['trajectory_length']}  "
              f"amb_thresh={kwargs['ambiguous_threshold']}")
        print(f"{'─'*60}")

        train_path = DATASET_DIR / f"{name}_train.pt"

        if args.stats_only:
            samples = torch.load(train_path, weights_only=False)
        else:
            val_path = DATASET_DIR / f"{name}_val.pt"
            train_ds = GridWorldDataset(
                n_samples=N_TRAIN, seed=42,
                cache_path=train_path, **GRID, **kwargs,
            )
            GridWorldDataset(
                n_samples=N_VAL, seed=43,
                cache_path=val_path, **GRID, **kwargs,
            )
            samples = train_ds.samples

        stats = dataset_stats(samples)
        print(f"  mean frac_ambiguous_steps : {stats['mean_frac_ambiguous']:.3f}")
        print(f"  mean first_unique_step    : {stats['mean_first_unique_step']:.2f} / {kwargs['trajectory_length']}")
        print(f"  mean initial_candidates   : {stats['mean_initial_candidates']:.2f}")

        all_stats[name] = {**kwargs, **GRID, **stats}

    stats_path = DATASET_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    if not args.stats_only:
        print(f"\nAll datasets saved to {DATASET_DIR}/")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
