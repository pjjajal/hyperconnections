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
    "expm": {
        "triton_median_ms":     "triton expm_t18",
        "matrix_exp_median_ms": "torch.linalg.matrix_exp",
        "t18_median_ms":        "compiled expm_t18",
    },
    "expm_force": {
        "blk_triton_median_ms": "triton expm_force",
        "matrix_exp_median_ms": "torch.linalg.matrix_exp",
        "blk_t18_median_ms":    "compiled expm_force",
    },
}

MARKERS = {
    "triton_median_ms":     "^",
    "matrix_exp_median_ms": "o",
    "t18_median_ms":        "s",
    "blk_triton_median_ms": "^",
    "blk_t18_median_ms":    "s",
}


def parse_config(cfg: str) -> tuple[int, int]:
    b = int(re.search(r"B=(\d+)", cfg).group(1))
    n = int(re.search(r"N=(\d+)", cfg).group(1))
    return b, n


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df[["batch", "n"]] = df["config"].apply(
        lambda c: pd.Series(parse_config(c))
    )
    return df


def plot_for_atype(df_atype: pd.DataFrame, atype: str, methods: dict, out_dir: Path) -> None:
    ns = sorted(df_atype["n"].unique())
    ncols = len(ns)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        sub = df_atype[df_atype["n"] == n].sort_values("batch")
        for col, label in methods.items():
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

    safe_atype = atype.strip().replace(" ", "_")
    fig.suptitle(f"expm benchmark — {atype}", fontsize=9)
    fig.tight_layout()

    out_path = out_dir / f"expm_benchmark_{safe_atype}.PNG"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot expm benchmark CSVs from benchmark_reports/."
    )
    parser.add_argument(
        "csvs",
        nargs="*",
        type=Path,
        help="One or more benchmark report CSV files.",
    )
    parser.add_argument(
        "--bench",
        choices=list(METHODS),
        default="expm",
        help="Which benchmark variant to plot: expm or expm_force (default: expm).",
    )
    parser.add_argument(
        "--singles-dir",
        type=Path,
        default=None,
        dest="singles_dir",
        help="Path to benchmark_reports/singles/; auto-discovers all CSVs for --bench.",
    )
    parser.add_argument(
        "--matrix-type",
        default=None,
        dest="matrix_type",
        help="Filter to a single matrix type (e.g. 'rand fwd'). Default: plot all.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to write output PNGs (default: current directory).",
    )
    args = parser.parse_args()

    if args.singles_dir:
        subdir = args.singles_dir / args.bench
        csvs = sorted(subdir.glob("**/*.CSV")) + sorted(subdir.glob("**/*.csv"))
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
    methods = METHODS[args.bench]

    atypes = sorted(df["atype"].unique())
    if args.matrix_type:
        atypes = [a for a in atypes if a == args.matrix_type]

    for atype in atypes:
        plot_for_atype(df[df["atype"] == atype].copy(), atype, methods, args.out_dir)


if __name__ == "__main__":
    main()
