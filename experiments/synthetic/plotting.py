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
    if window <= 1:
        return steps, values
    return steps[window - 1 :], moving_average(values, window)


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

    def save(fig, stem):
        out_png = out_dir / f"{stem}.png"
        out_pdf = out_dir / f"{stem}.pdf"
        fig.savefig(out_png)
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")

    # Loss plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for run_name, config, metrics in runs:
        if not metrics:
            continue
        steps = global_steps(metrics)
        loss = metric_series(metrics, "loss")
        if loss is None:
            continue
        steps_s, loss_s = smooth_pair(steps, loss, args.smooth)
        model = config.get("model", run_name)
        ax.plot(
            steps_s,
            loss_s,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, None),
            alpha=0.9,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{results_name} — depth {args.depth}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, f"{args.depth}")

    # SNR plot (if all SNR metrics are present)
    snr_keys = [
        ("input_snr", "Input SNR (dB)"),
        ("output_snr", "Output SNR (dB)"),
        ("snr_queried", "Queried SNR (dB)"),
    ]
    has_snr = all(
        metric_series(m, k) is not None
        for _, _, m in runs
        for k, _ in snr_keys
        if m
    )
    if has_snr:
        fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
        for ax, (key, ylabel) in zip(axes, snr_keys):
            for run_name, config, metrics in runs:
                if not metrics:
                    continue
                steps = global_steps(metrics)
                vals = metric_series(metrics, key)
                if vals is None:
                    continue
                vals_db = to_db(vals)
                steps_s, vals_s = smooth_pair(steps, vals_db, args.smooth)
                model = config.get("model", run_name)
                ax.plot(
                    steps_s,
                    vals_s,
                    label=MODEL_LABELS.get(model, model),
                    color=MODEL_COLORS.get(model, None),
                    alpha=0.9,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)
        axes[0].set_title(f"{results_name} — depth {args.depth}")
        axes[-1].set_xlabel("Step")
        axes[0].legend()
        fig.tight_layout()
        save(fig, f"{args.depth}_snr")


if __name__ == "__main__":
    main()
