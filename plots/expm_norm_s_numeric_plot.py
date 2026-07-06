"""Plot the T18 squaring-count (S) numerical-error ablation from a
``benchmark_expm_norm_correctness_*`` CSV produced by
``benchmarks/expm_norm_bench.py``.

For each config the relative error at a given constexpr ``S`` is normalised by
the error at the norm-matched baseline ``S == s_req`` and plotted against the
offset ``delta_S = S - s_req``; one panel per correctness check
(``exp``/``psi``/``gradient``). Values below zero mean the offset ``S`` is *more*
accurate than the norm-matched baseline.

Standalone entry point; borrows headless setup + ``save_fig`` from
``bench_plot_common`` so styling/saving matches the other plot scripts. The
science-paper look is kept by default (``scienceplots``); ``--plain`` falls back
to the shared rcParams when scienceplots/LaTeX are unavailable.

Examples
--------
# Auto-discover the newest fp32 correctness CSV under benchmark_reports/:
    python plots/expm_norm_s_numeric_plot.py

# Plot the bf16 run that was moved into a subfolder:
    python plots/expm_norm_s_numeric_plot.py --report-dir benchmark_reports/expm_norm_bf16

# Explicit CSV + output path, only N in {4, 8}, only two checks:
    python plots/expm_norm_s_numeric_plot.py path/to/report.CSV -o fig.png \
        --n 4 8 --checks "vs linalg.matrix_exp" "vs autograd"
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# bench_plot_common selects the Agg backend and installs the shared rcParams on
# import; we reuse its save_fig and (with --plain) its styling.
import bench_plot_common as bpc
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCE = True
except ImportError:
    _HAS_SCIENCE = False

_MERGE_KEYS = ["config", "variant", "check"]
_CORR_GLOB = "benchmark_expm_norm_correctness_*"
_DTYPE_RE = re.compile(r"\b(fp32|bf16|fp16)\b")


def _cfg_dtype(cfg: str):
    m = _DTYPE_RE.search(cfg)
    return m.group(1) if m else None


def _cfg_n(cfg: str):
    m = re.search(r"\bN=(\d+)", cfg)
    return int(m.group(1)) if m else None


def discover_csv(report_dir: Path) -> Path:
    """Newest ``benchmark_expm_norm_correctness_*`` CSV under ``report_dir``."""
    cands = sorted(report_dir.glob(_CORR_GLOB), key=lambda p: p.stat().st_mtime)
    if not cands:
        print(f"No {_CORR_GLOB} CSV found under {report_dir}", file=sys.stderr)
        sys.exit(1)
    return cands[-1]


def build_plot_df(
    df: pd.DataFrame,
    dtypes=None,
    ns=None,
    checks=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the requested filters, then attach ``delta_S`` and the
    baseline-normalised ``log2_rel_err_ratio``.

    Returns the long dataframe and the ordered list of checks to plot.
    """
    df = df.copy()
    df["config"] = df["config"].astype(str)

    if dtypes:
        keep = set(dtypes)
        df = df[df["config"].map(_cfg_dtype).isin(keep)]
    if ns:
        keep_n = set(ns)
        df = df[df["config"].map(_cfg_n).isin(keep_n)]

    # Check order: CSV order of appearance, restricted to --checks if given.
    present = list(dict.fromkeys(df["check"].tolist()))
    if checks:
        want = {c.strip() for c in checks}
        present = [c for c in present if c.strip() in want]
        df = df[df["check"].map(lambda c: c.strip() in want)]

    if df.empty:
        print("No rows left after filtering; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    df["delta_S"] = df["S"] - df["s_req"]

    baseline = (
        df[df["S"] == df["s_req"]]
        .loc[:, _MERGE_KEYS + ["rel_err"]]
        .rename(columns={"rel_err": "rel_err_baseline"})
    )
    plot_df = df.merge(baseline, on=_MERGE_KEYS, how="inner")

    eps = np.finfo(float).tiny
    plot_df["log2_rel_err_ratio"] = np.log2(
        np.maximum(plot_df["rel_err"], eps)
        / np.maximum(plot_df["rel_err_baseline"], eps)
    )
    return plot_df, present


def dtype_tag(plot_df: pd.DataFrame) -> str:
    """Filename tag for the dtype(s) present (e.g. 'fp32', 'bf16', 'fp32-bf16')."""
    present = sorted({d for d in plot_df["config"].map(_cfg_dtype) if d})
    return "-".join(present) if present else "all"


def make_figure(plot_df: pd.DataFrame, checks: list[str], title, seed: int):
    fig, axes = plt.subplots(
        1,
        len(checks),
        figsize=(3.2 * len(checks), 2.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(checks) == 1:
        axes = [axes]

    rng = np.random.default_rng(seed)
    for ax, check in zip(axes, checks):
        sub = plot_df[plot_df["check"] == check]
        offsets = sorted(sub["delta_S"].unique())

        data = [
            sub.loc[sub["delta_S"] == d, "log2_rel_err_ratio"].to_numpy()
            for d in offsets
        ]
        ax.boxplot(data, positions=offsets, widths=0.55,
                   showfliers=False, patch_artist=False)

        for d in offsets:
            y = sub.loc[sub["delta_S"] == d, "log2_rel_err_ratio"].to_numpy()
            x = d + rng.normal(0.0, 0.045, size=len(y))
            ax.scatter(x, y, s=8, alpha=0.35, linewidths=0)

        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.axvline(0.0, linestyle=":", linewidth=0.8)
        ax.set_title(check.strip())
        ax.set_xlabel(r"$\Delta S = S - s_{\mathrm{req}}$")
        ax.set_xticks(offsets)

    axes[0].set_ylabel(
        r"$\log_2\left(\mathrm{relerr}(S) / "
        r"\mathrm{relerr}(S=s_{\mathrm{req}})\right)$"
    )
    if title:
        fig.suptitle(title)
    return fig


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the T18 squaring-count (S) numerical-error ablation "
                    "from a benchmark_expm_norm_correctness CSV.",
    )
    parser.add_argument(
        "csv", nargs="?", type=Path, default=None,
        help="Correctness CSV to plot. If omitted, the newest "
             f"{_CORR_GLOB} under --report-dir is used.",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("benchmark_reports"),
        help="Directory to auto-discover the correctness CSV in and to write the "
             "default output into (default: benchmark_reports). Point at "
             "benchmark_reports/expm_norm_bf16 for the bf16 run.",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="Output figure path (default: "
             "<report-dir>/expm_squaring_offset_error_<dtype>_<today>.png).",
    )
    parser.add_argument(
        "--dtype", nargs="+", choices=["fp32", "bf16", "fp16"], default=None,
        help="Only plot configs of these dtype(s) (default: all present).",
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=None, metavar="N",
        help="Only plot these matrix sizes N (default: all present).",
    )
    parser.add_argument(
        "--checks", nargs="+", default=None, metavar="CHECK",
        help="Subset of correctness checks to plot, matched on name "
             "(e.g. 'vs linalg.matrix_exp' 'vs autograd'). Default: all.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Figure suptitle (default: none).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output DPI (default: 300).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for point jitter (default: 0).",
    )
    parser.add_argument(
        "--plain", action="store_true",
        help="Use bench_plot_common's default styling instead of scienceplots.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if args.plain or not _HAS_SCIENCE:
        if not args.plain and not _HAS_SCIENCE:
            print("scienceplots not installed; using default (--plain) style.",
                  file=sys.stderr)
        # bench_plot_common's rcParams are already active from import.
    else:
        plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update({"savefig.dpi": args.dpi, "figure.dpi": args.dpi})

    csv_path = args.csv if args.csv is not None else discover_csv(args.report_dir)
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    plot_df, checks = build_plot_df(df, args.dtype, args.n, args.checks)

    out_path = args.out or (
        args.report_dir
        / f"expm_squaring_offset_error_{dtype_tag(plot_df)}_{date.today():%Y_%m_%d}.png"
    )

    fig = make_figure(plot_df, checks, args.title, args.seed)
    bpc.save_fig(fig, out_path)


if __name__ == "__main__":
    main()
