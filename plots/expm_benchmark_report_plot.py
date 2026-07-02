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
    "matrix_exp_ms": "torch.linalg.matrix_exp",
    "t18_ms":        "compiled expm_t18",
    "triton_ms":     "triton expm_t18",
}

MARKERS = {
    "matrix_exp_ms": "o",
    "t18_ms":        "s",
    "triton_ms":     "^",
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


def plot_for_atype(df_atype: pd.DataFrame, atype: str, out_dir: Path) -> None:
    ns = sorted(df_atype["n"].unique())
    ncols = len(ns)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        sub = df_atype[df_atype["n"] == n].sort_values("batch")
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
        nargs="+",
        type=Path,
        help="One or more benchmark report CSV files.",
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
        help="Directory to write output PDFs (default: current directory).",
    )
    args = parser.parse_args()

    missing = [p for p in args.csvs if not p.exists()]
    if missing:
        for p in missing:
            print(f"File not found: {p}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(args.csvs)

    atypes = sorted(df["atype"].unique())
    if args.matrix_type:
        atypes = [a for a in atypes if a == args.matrix_type]

    for atype in atypes:
        plot_for_atype(df[df["atype"] == atype].copy(), atype, args.out_dir)


if __name__ == "__main__":
    main()
