"""Plot single-depth train and validation curves across seeds."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ["preservation", "rotation", "permutation", "filtering"]
MODELS = ["cghc", "mhc", "ghc"]
COLORS = {"cghc": "tab:blue", "mhc": "tab:orange", "ghc": "tab:green"}
LABELS = {"cghc": "cGHC", "mhc": "mHC", "ghc": "GHC"}


def load_results(root: Path) -> list[dict]:
    results = []
    for path in root.glob("*.json"):
        result = json.loads(path.read_text())
        if result.get("history"):
            results.append(result)
    return results


def select_depth(results: list[dict], requested: int | None) -> int:
    depths = sorted({result["depth"] for result in results})
    if requested is not None:
        if requested not in depths:
            raise ValueError(f"depth {requested} not found; available depths: {depths}")
        return requested
    if len(depths) != 1:
        raise ValueError(f"results contain depths {depths}; select one with --depth")
    return depths[0]


def curve(runs: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_steps = sorted(
        set.intersection(
            *({entry["step"] for entry in run["history"]} for run in runs)
        )
    )
    values = np.asarray(
        [
            [
                {entry["step"]: entry[metric] for entry in run["history"]}[step]
                for step in common_steps
            ]
            for run in runs
        ],
        dtype=np.float64,
    )
    mean = values.mean(axis=0)
    if len(runs) == 1:
        standard_error = np.zeros_like(mean)
    else:
        standard_error = values.std(axis=0, ddof=1) / len(runs) ** 0.5
    return np.asarray(common_steps), mean, standard_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("experiments/synthetic_dynamics/results")
    )
    parser.add_argument("--depth", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = load_results(args.results)
    if not results:
        raise ValueError(f"no results with training history found in {args.results}")
    depth = select_depth(results, args.depth)
    results = [result for result in results if result["depth"] == depth]

    grouped = defaultdict(list)
    for result in results:
        grouped[(result["task"], result["model"])].append(result)

    figure, axes = plt.subplots(
        len(TASKS), 2, figsize=(10, 2.6 * len(TASKS)), sharex=True, squeeze=False
    )
    for row, task in enumerate(TASKS):
        for column, metric in enumerate(("train_loss", "val_loss")):
            axis = axes[row, column]
            for model in MODELS:
                runs = grouped.get((task, model), [])
                if not runs:
                    continue
                steps, mean, standard_error = curve(runs, metric)
                color = COLORS[model]
                axis.plot(steps, mean, color=color, label=LABELS[model])
                axis.fill_between(
                    steps,
                    mean - standard_error,
                    mean + standard_error,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )

            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title("Train" if column == 0 else "Validation")
            if column == 0:
                axis.set_ylabel(task.title())
            if row == len(TASKS) - 1:
                axis.set_xlabel("Optimization step")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=len(handles))
    figure.suptitle(f"Synthetic dynamics, depth {depth}", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.97))

    output = args.output or args.results / f"learning_curves_L{depth}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
