import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
    }
)


MODEL_LABELS = {
    "cghc": "CGHC",
    "mhc": "MHC",
    "ghc": "GHC",
    "identity_hc": "Identity HC",
}

MODEL_COLORS = {
    "cghc": "tab:blue",
    "mhc": "tab:orange",
    "ghc": "tab:green",
    "identity_hc": "tab:red",
}


def parse_args():
    parser = argparse.ArgumentParser("Plot results for synthetic tasks.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/synthetic/runs/preservation",
        help="Directory containing training run subdirectories",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Depth (n_layers) of runs to plot",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window for the loss curve (1 = no smoothing)",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="experiments/synthetic/plots",
        help="Root directory for output plots",
    )
    return parser.parse_args()


def load_run(run_dir: Path):
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return None
    with open(config_path) as f:
        config = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    return config, metrics


def global_steps_and_loss(metrics):
    if not metrics:
        return np.array([]), np.array([])
    max_step_per_epoch = max(e["step"] for e in metrics) + 1
    steps = np.array(
        [e["epoch"] * max_step_per_epoch + e["step"] for e in metrics], dtype=np.int64
    )
    loss = np.array([e["loss"] for e in metrics], dtype=np.float64)
    return steps, loss


def moving_average(x, window):
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        loaded = load_run(run_dir)
        if loaded is None:
            continue
        config, metrics = loaded
        if config.get("n_layers") != args.depth:
            continue
        runs.append((run_dir.name, config, metrics))

    if not runs:
        print(f"No runs with depth={args.depth} found in {results_dir}")
        return

    results_name = results_dir.name
    out_dir = Path(args.out_root) / results_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    for run_name, config, metrics in runs:
        steps, loss = global_steps_and_loss(metrics)
        if loss.size == 0:
            continue
        if args.smooth > 1:
            loss_s = moving_average(loss, args.smooth)
            steps_s = steps[args.smooth - 1 :]
        else:
            steps_s = steps
            loss_s = loss
        model = config.get("model", run_name)
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, None)
        ax.plot(steps_s, loss_s, label=label, color=color, alpha=0.9)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{results_name} — depth {args.depth}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = out_dir / f"{args.depth}.png"
    out_pdf = out_dir / f"{args.depth}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
