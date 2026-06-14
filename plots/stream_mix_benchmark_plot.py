import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import scienceplots

plt.style.use(["science", "no-latex", "ieee"])
matplotlib.rcParams["font.family"] = "monospace"

METHODS = {
    "triton_median_ms":   "Triton stream_mix",
    "eager_median_ms":    "PyTorch eager",
    "compiled_median_ms": "torch.compile",
}

MARKERS = {
    "triton_median_ms":   "^",
    "eager_median_ms":    "o",
    "compiled_median_ms": "s",
}


def parse_config(cfg: str) -> tuple[int, int, int, int]:
    b = int(re.search(r"B=(\d+)", cfg).group(1))
    n = int(re.search(r"N=(\d+)", cfg).group(1))
    m = int(re.search(r"m=(\d+)", cfg).group(1))
    d = int(re.search(r"D=(\d+)", cfg).group(1))
    return b, n, m, d


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df[["batch", "n", "m", "d"]] = df["config"].apply(
        lambda c: pd.Series(parse_config(c))
    )
    return df


def plot_for_variant_phi(
    df_sub: pd.DataFrame,
    phi_type: str,
    variant: str,
    out_dir: Path,
) -> None:
    ns = sorted(df_sub["n"].unique())
    ncols = len(ns)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        sub = df_sub[df_sub["n"] == n].sort_values("batch")
        for col, label in METHODS.items():
            ax.scatter(
                sub["batch"],
                sub[col],
                label=label,
                marker=MARKERS[col],
                s=18,
                linewidths=0.5,
            )
            ax.plot(sub["batch"], sub[col], linewidth=0.8, alpha=0.6)

        ax.set_title(f"N={n}", fontsize=7)
        ax.set_xlabel("Batch size", fontsize=7)
        ax.set_ylabel("Time (ms)", fontsize=7)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(axis="both", labelsize=6)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=5, frameon=True, loc="upper left")

    safe_phi = phi_type.strip().replace(" ", "_")
    safe_var = variant.strip().replace(" ", "_")
    fig.suptitle(f"stream_mix benchmark — {phi_type} / {variant}", fontsize=9)
    fig.tight_layout()

    out_path = out_dir / f"stream_mix_benchmark_{safe_phi}_{safe_var}.PNG"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stream_mix benchmark CSVs from benchmark_reports/."
    )
    parser.add_argument(
        "csvs",
        nargs="*",
        type=Path,
        help="One or more benchmark report CSV files.",
    )
    parser.add_argument(
        "--singles-dir",
        type=Path,
        default=None,
        dest="singles_dir",
        help="Path to benchmark_reports/singles/; auto-discovers all stream_mix CSVs.",
    )
    parser.add_argument(
        "--phi-type",
        default=None,
        dest="phi_type",
        help="Filter to a single phi_type (e.g. 'random', 'skew_sym'). Default: plot all.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Filter to a single variant (e.g. 'fwd', 'no-proj fwd'). Default: plot all.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to write output PNGs (default: current directory).",
    )
    args = parser.parse_args()

    if args.singles_dir:
        subdir = args.singles_dir / "stream_mix"
        csvs = sorted(subdir.glob("**/*.csv")) + sorted(subdir.glob("**/*.CSV"))
        if not csvs:
            print(f"No CSVs found under {subdir}", file=sys.stderr)
            sys.exit(1)
    elif args.csvs:
        csvs = args.csvs
        missing = [p for p in csvs if not p.exists()]
        if missing:
            for p in missing:
                print(f"File not found: {p}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Provide CSV files or --singles-dir.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(csvs)

    phi_types = sorted(df["phi_type"].unique())
    variants  = sorted(df["variant"].unique())

    if args.phi_type:
        phi_types = [p for p in phi_types if p == args.phi_type]
    if args.variant:
        variants = [v for v in variants if v == args.variant]

    for phi_type in phi_types:
        for variant in variants:
            sub = df[(df["phi_type"] == phi_type) & (df["variant"] == variant)]
            if sub.empty:
                continue
            plot_for_variant_phi(sub.copy(), phi_type, variant, args.out_dir)


if __name__ == "__main__":
    main()
