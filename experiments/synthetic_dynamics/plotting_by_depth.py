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
        "savefig.dpi": 200,
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

MODEL_ORDER = ["cghc", "mhc", "ghc", "identity_hc"]

TASK_LABELS = {
    "preservation": "Preservation",
    "filtering": "Filtering",
    "filtering_ret": "Filtering (retrieval loss)",
    "rotation_orthogonal": "Rotation (orthogonal)",
    "rotation_permutation": "Rotation (permutation)",
}

DEFAULT_TASK_ORDER = [
    "preservation",
    "rotation_orthogonal",
    "rotation_permutation",
    "filtering",
]

# Tasks for which we plot queried SNR (in dB) instead of loss.
SNR_TASKS = {"filtering"}
SNR_KEY = "snr_queried"
SNR_YLABEL = "Queried SNR (dB)"


def parse_args():
    parser = argparse.ArgumentParser(
        "Plot synthetic results as one figure per depth, with tasks as subplots."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="experiments/synthetic/runs",
        help="Root directory containing one subdirectory per task.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated task subdir names (in plot order), or 'all'.",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="all",
        help="Comma-separated depths to plot, or 'all'.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=200,
        help="Moving-average window for curves (1 = no smoothing).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="experiments/synthetic/plots_by_depth",
        help="Root directory for output plots.",
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


def global_steps(metrics):
    if not metrics:
        return np.array([], dtype=np.int64)
    max_step_per_epoch = max(e["step"] for e in metrics) + 1
    return np.array(
        [e["epoch"] * max_step_per_epoch + e["step"] for e in metrics], dtype=np.int64
    )


def metric_series(metrics, key):
    vals = [e[key] for e in metrics if key in e]
    if len(vals) != len(metrics):
        return None
    return np.array(vals, dtype=np.float64)


def to_db(x):
    return 10.0 * np.log10(np.clip(x, 1e-30, None))


def moving_average(x, window):
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def smooth_pair(steps, values, window):
    if window <= 1 or len(values) < window:
        return steps, values
    return steps[window - 1 :], moving_average(values, window)


def collect_runs_by_task_depth(runs_root: Path, tasks: list[str]):
    """Returns {task: {depth: [(model, config, metrics), ...]}}."""
    out: dict[str, dict[int, list]] = {}
    for task in tasks:
        task_dir = runs_root / task
        if not task_dir.is_dir():
            continue
        per_depth: dict[int, list] = {}
        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            loaded = load_run(run_dir)
            if loaded is None:
                continue
            config, metrics = loaded
            if not metrics:
                continue
            depth = int(config.get("n_layers"))
            model = config.get("model", run_dir.name)
            per_depth.setdefault(depth, []).append((model, config, metrics))
        for runs in per_depth.values():
            runs.sort(
                key=lambda r: MODEL_ORDER.index(r[0]) if r[0] in MODEL_ORDER else 99
            )
        if per_depth:
            out[task] = per_depth
    return out


def plot_loss(runs, ax, smooth):
    plotted = False
    for model, _config, metrics in runs:
        steps = global_steps(metrics)
        loss = metric_series(metrics, "loss")
        if loss is None:
            continue
        steps_s, loss_s = smooth_pair(steps, loss, smooth)
        ax.plot(
            steps_s,
            loss_s,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model),
            alpha=0.9,
        )
        plotted = True
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    return plotted


def plot_snr(runs, key, ax, smooth):
    plotted = False
    for model, _config, metrics in runs:
        steps = global_steps(metrics)
        vals = metric_series(metrics, key)
        if vals is None:
            continue
        vals_db = to_db(vals)
        steps_s, vals_s = smooth_pair(steps, vals_db, smooth)
        ax.plot(
            steps_s,
            vals_s,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model),
            alpha=0.9,
        )
        plotted = True
    ax.grid(True, which="both", alpha=0.3)
    return plotted


def save_fig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def figure_per_depth(data, tasks, depth, out_dir, smooth):
    cols = len(tasks)
    fig, axes = plt.subplots(
        1,
        cols,
        figsize=(4.5 * cols, 3.4),
        squeeze=False,
    )
    axes_flat = axes[0]
    legend_handles = None
    for idx, task in enumerate(tasks):
        ax = axes_flat[idx]
        runs = data.get(task, {}).get(depth, [])
        use_snr = task in SNR_TASKS
        if use_snr:
            plotted = plot_snr(runs, SNR_KEY, ax, smooth)
            ax.set_ylabel(SNR_YLABEL)
        else:
            plotted = plot_loss(runs, ax, smooth)
            ax.set_ylabel("Loss")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Step")
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "no runs",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )
        if plotted and legend_handles is None:
            legend_handles = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(legend_handles[1]),
            frameon=False,
        )
    save_fig(fig, out_dir, f"depth{depth}")


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    if args.tasks == "all":
        present = {p.name for p in runs_root.iterdir() if p.is_dir()}
        tasks = [t for t in DEFAULT_TASK_ORDER if t in present]
    else:
        tasks = [t for t in args.tasks.split(",") if t]

    data = collect_runs_by_task_depth(runs_root, tasks)
    if not data:
        print(f"No runs found under {runs_root} for tasks {tasks}")
        return

    all_depths = sorted({d for per_depth in data.values() for d in per_depth.keys()})
    if args.depths == "all":
        depths = all_depths
    else:
        depths = [int(d) for d in args.depths.split(",") if d]

    out_dir = Path(args.out_root)
    for depth in depths:
        figure_per_depth(data, tasks, depth, out_dir, args.smooth)


if __name__ == "__main__":
    main()
