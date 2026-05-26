import argparse
import json
import math
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

SNR_KEYS = [
    ("input_snr", "Input SNR (dB)"),
    ("output_snr", "Output SNR (dB)"),
    ("snr_queried", "Queried SNR (dB)"),
]


def parse_args():
    parser = argparse.ArgumentParser("Plot results for synthetic tasks.")
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
        help="Comma-separated task subdir names, or 'all'.",
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
        default="experiments/synthetic/plots",
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


def collect_runs(task_dir: Path):
    """Group runs by depth. Returns {depth: [(model, config, metrics), ...]}."""
    runs_by_depth: dict[int, list] = {}
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
        runs_by_depth.setdefault(depth, []).append((model, config, metrics))
    for depth, runs in runs_by_depth.items():
        runs.sort(key=lambda r: MODEL_ORDER.index(r[0]) if r[0] in MODEL_ORDER else 99)
    return runs_by_depth


def grid_shape(n):
    if n <= 0:
        return 1, 1
    cols = min(3, n)
    rows = math.ceil(n / cols)
    return rows, cols


def plot_loss_per_depth(runs, smooth, ax):
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
            color=MODEL_COLORS.get(model, None),
            alpha=0.9,
        )
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


def plot_snr_per_depth(runs, key, smooth, ax):
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
            color=MODEL_COLORS.get(model, None),
            alpha=0.9,
        )
    ax.grid(True, which="both", alpha=0.3)


def has_snr(runs_by_depth):
    for runs in runs_by_depth.values():
        for _model, _config, metrics in runs:
            if any(metric_series(metrics, k) is None for k, _ in SNR_KEYS):
                return False
    return bool(runs_by_depth)


def save_fig(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def plot_task(task_dir: Path, out_dir: Path, smooth: int):
    runs_by_depth = collect_runs(task_dir)
    if not runs_by_depth:
        print(f"No runs found in {task_dir}")
        return

    task_name = task_dir.name
    depths = sorted(runs_by_depth.keys())
    snr_available = has_snr(runs_by_depth)

    # Per-depth individual loss figures.
    for depth in depths:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_loss_per_depth(runs_by_depth[depth], smooth, ax)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{task_name} — depth {depth}")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, out_dir, f"loss_depth{depth}")

        if snr_available:
            fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
            for ax, (key, ylabel) in zip(axes, SNR_KEYS):
                plot_snr_per_depth(runs_by_depth[depth], key, smooth, ax)
                ax.set_ylabel(ylabel)
            axes[0].set_title(f"{task_name} — depth {depth}")
            axes[-1].set_xlabel("Step")
            axes[0].legend()
            fig.tight_layout()
            save_fig(fig, out_dir, f"snr_depth{depth}")

    # Task-level grid: one subplot per depth (loss).
    rows, cols = grid_shape(len(depths))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharex=False, sharey=True
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for idx, depth in enumerate(depths):
        ax = axes_flat[idx]
        plot_loss_per_depth(runs_by_depth[depth], smooth, ax)
        ax.set_title(f"depth {depth}")
        if idx % cols == 0:
            ax.set_ylabel("Loss")
        if idx >= (rows - 1) * cols:
            ax.set_xlabel("Step")
    for j in range(len(depths), len(axes_flat)):
        axes_flat[j].set_visible(False)
    axes_flat[0].legend()
    fig.suptitle(f"{task_name} — loss across depths")
    fig.tight_layout()
    save_fig(fig, out_dir, "loss_grid")

    # Task-level grid for each SNR metric.
    if snr_available:
        for key, ylabel in SNR_KEYS:
            fig, axes = plt.subplots(
                rows, cols, figsize=(4.2 * cols, 3.2 * rows), sharex=False, sharey=True
            )
            axes_flat = np.atleast_1d(axes).ravel()
            for idx, depth in enumerate(depths):
                ax = axes_flat[idx]
                plot_snr_per_depth(runs_by_depth[depth], key, smooth, ax)
                ax.set_title(f"depth {depth}")
                if idx % cols == 0:
                    ax.set_ylabel(ylabel)
                if idx >= (rows - 1) * cols:
                    ax.set_xlabel("Step")
            for j in range(len(depths), len(axes_flat)):
                axes_flat[j].set_visible(False)
            axes_flat[0].legend()
            fig.suptitle(f"{task_name} — {ylabel} across depths")
            fig.tight_layout()
            save_fig(fig, out_dir, f"snr_{key}_grid")

    # Depth-scaling summary: best metric across the run vs depth, one line per model.
    summary = depth_scaling_summary(runs_by_depth)
    if summary:
        plot_depth_scaling(summary, task_name, out_dir, snr_available)


def depth_scaling_summary(runs_by_depth):
    """For each (model, depth), compute min loss and max SNR over the run."""
    depths = sorted(runs_by_depth.keys())
    summary: dict[str, dict[int, dict[str, float]]] = {}
    for depth in depths:
        for model, _config, metrics in runs_by_depth[depth]:
            entry = summary.setdefault(model, {}).setdefault(depth, {})
            loss = metric_series(metrics, "loss")
            if loss is not None and len(loss):
                entry["loss"] = float(np.min(loss))
            for key, _ in SNR_KEYS:
                vals = metric_series(metrics, key)
                if vals is not None and len(vals):
                    entry[key] = float(np.max(vals))
    return summary


def plot_depth_scaling(summary, task_name, out_dir, snr_available):
    keys = ["loss"]
    if snr_available:
        keys += [k for k, _ in SNR_KEYS]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    label_for = {"loss": "Min loss"}
    for k, name in SNR_KEYS:
        label_for[k] = f"Max {name}"

    for ax, key in zip(axes, keys):
        for model in MODEL_ORDER:
            if model not in summary:
                continue
            depth_map = summary[model]
            xs = sorted(d for d in depth_map if key in depth_map[d])
            if not xs:
                continue
            ys = [depth_map[d][key] for d in xs]
            if key != "loss":
                ys = [to_db(np.array([y]))[0] for y in ys]
            ax.plot(
                xs,
                ys,
                marker="o",
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Depth")
        ax.set_ylabel(label_for[key])
        ax.grid(True, which="both", alpha=0.3)
        if key == "loss":
            ax.set_yscale("log")
    axes[0].legend()
    fig.suptitle(f"{task_name} — best metric vs depth")
    fig.tight_layout()
    save_fig(fig, out_dir, "depth_scaling")


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    if args.tasks == "all":
        task_dirs = [p for p in sorted(runs_root.iterdir()) if p.is_dir()]
    else:
        task_dirs = [runs_root / t for t in args.tasks.split(",") if t]

    out_root = Path(args.out_root)
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            print(f"Skipping (not a directory): {task_dir}")
            continue
        out_dir = out_root / task_dir.name
        plot_task(task_dir, out_dir, args.smooth)


if __name__ == "__main__":
    main()
