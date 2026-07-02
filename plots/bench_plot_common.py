"""Shared plotting helpers for the benchmark-report plot scripts.

Both the per-kernel scripts (``expm_benchmark_report_plot.py``,
``stream_mix_benchmark_plot.py``) and the suite driver (``generate_all.py``)
import from here so the styling, CSV parsing, and plot routines live in one
place.

Style: FlashAttention-style grouped bar charts (value-labelled bars vs.
workload size). Each figure is a single dtype, so methods are distinguished by
distinct colours rather than line styles, and dtype is a separate figure /
folder level.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi":      130,
    "savefig.dpi":     130,
})

# Distinct colour per logical method (green / red / purple, FA-style).
METHOD_COLOR = {
    "triton_median_ms":     "#2ca02c",  # green  — Triton (the kernel under test)
    "blk_triton_median_ms": "#2ca02c",
    "matrix_exp_median_ms": "#d62728",  # red    — torch baseline
    "eager_median_ms":      "#d62728",
    "t18_median_ms":        "#9467bd",  # purple — compiled reference
    "blk_t18_median_ms":    "#9467bd",
    "compiled_median_ms":   "#9467bd",
}

# Distinct colour per speedup series (one per baseline compared against).
SPEEDUP_COLOR = {
    "speedup_vs_torch":    "#d62728",  # red
    "speedup_vs_eager":    "#d62728",
    "speedup_vs_t18":      "#9467bd",  # purple
    "speedup_vs_compiled": "#9467bd",
}

# Per-kernel descriptors: report subdir, perf-CSV glob, grouping columns, and
# the method/speedup column -> legend-label maps.  Single source of truth for
# the driver and the standalone scripts.
KERNELS = {
    "expm": {
        "subdir": "expm",
        "perf_glob": "**/benchmark_expm_perf_*",
        "group_cols": ["atype"],
        "split_atype": True,       # atype = "<matrix type> <direction>"
        "avg_label": "matrix types",
        "methods": {
            "triton_median_ms":     "Triton expm_t18",
            "matrix_exp_median_ms": "torch.matrix_exp",
            "t18_median_ms":        "compiled t18",
        },
        "speedups": {
            "speedup_vs_torch": "vs torch.matrix_exp",
            "speedup_vs_t18":   "vs compiled t18",
        },
    },
    "expm_force": {
        "subdir": "expm_force",
        "perf_glob": "**/benchmark_expm_force_perf_*",
        "group_cols": ["atype"],
        "split_atype": True,
        "avg_label": "matrix types",
        "methods": {
            "blk_triton_median_ms": "Triton expm_force",
            "matrix_exp_median_ms": "torch.matrix_exp",
            "blk_t18_median_ms":    "compiled expm_force",
        },
        "speedups": {
            "speedup_vs_torch": "vs torch.matrix_exp",
            "speedup_vs_t18":   "vs compiled t18",
        },
    },
    "stream_mix": {
        "subdir": "stream_mix",
        "perf_glob": "**/stream_mix_perf.csv",
        "group_cols": ["phi_type", "variant"],
        "summary_group_col": "variant",   # average over phi_type, keep variant
        "drop_variants": ["fwd", "bwd"],  # bare aggregates duplicate "no-proj"
        "avg_label": "phi types",
        "methods": {
            "triton_median_ms":   "Triton stream_mix",
            "eager_median_ms":    "PyTorch eager",
            "compiled_median_ms": "torch.compile",
        },
        "speedups": {
            "speedup_vs_eager":    "vs eager",
            "speedup_vs_compiled": "vs torch.compile",
        },
    },
}


def parse_config(cfg: str) -> dict:
    """Parse a ``config`` cell like ``"B=1024 N=16 m=1 D=1024"``.

    ``m``/``D`` are stream_mix-only and simply omitted when absent.
    """
    out = {
        "batch": int(re.search(r"B=(\d+)", cfg).group(1)),
        "n":     int(re.search(r"N=(\d+)", cfg).group(1)),
    }
    m = re.search(r"m=(\d+)", cfg)
    d = re.search(r"D=(\d+)", cfg)
    if m:
        out["m"] = int(m.group(1))
    if d:
        out["d"] = int(d.group(1))
    return out


def load_csvs(paths) -> pd.DataFrame:
    """Read + concat perf CSVs, adding parsed ``batch``/``n`` (+ ``m``/``d``)."""
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    parsed = df["config"].apply(lambda c: pd.Series(parse_config(c)))
    return pd.concat([df, parsed], axis=1)


def summarise(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Add a ``summary_group`` column for the averaged-over-input-type plots.

    For expm/expm_force, ``atype`` (e.g. ``"rand fwd"``) is split into matrix
    type + direction and we group by direction (averaging over matrix type).
    For stream_mix we group by ``variant`` (averaging over ``phi_type``).
    """
    df = df.copy()
    if spec.get("split_atype"):
        parts = df["atype"].str.rsplit(" ", n=1, expand=True)
        df["matrix_type"] = parts[0].str.strip()
        df["summary_group"] = parts[1].str.strip()  # direction: fwd / bwd
    else:
        if spec.get("drop_variants"):
            df = df[~df[spec["summary_group_col"]].isin(spec["drop_variants"])]
        df["summary_group"] = df[spec["summary_group_col"]]
    return df


def _fmt_batch(b) -> str:
    """Humanise a batch size for the x tick (1024 -> '1k', 120064 -> '120k')."""
    b = int(b)
    if b >= 1024 and b % 1024 == 0:
        return f"{b // 1024}k"
    if b >= 1000:
        return f"{round(b / 1000)}k"
    return str(b)


def _fmt_latency(v: float) -> str:
    return f"{v:.3g}"


def _fmt_speedup(v: float) -> str:
    return f"{v:.2f}"


def _grouped_bars(ax, df_one, value_cols, labels, colors, fmt, agg="median") -> list[float]:
    """Draw value-labelled grouped bars: x = batch size, one bar per series.

    ``df_one`` is restricted to a single (summary group, N, dtype). The per-batch
    bar height is the ``agg`` (``"median"`` to dedup repeated runs of one config,
    ``"mean"`` to average across input matrix / phi types). Returns all values.
    """
    batches = sorted(df_one["batch"].unique())
    x = np.arange(len(batches))
    cols = [c for c in value_cols if c in df_one.columns]
    n = max(len(cols), 1)
    width = 0.8 / n

    all_vals: list[float] = []
    for i, col in enumerate(cols):
        vals = [getattr(df_one.loc[df_one["batch"] == b, col], agg)() for b in batches]
        all_vals += [v for v in vals if v == v]  # drop NaN
        offset = (i - (len(cols) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=labels[col], color=colors.get(col))
        ax.bar_label(bars, labels=[fmt(v) for v in vals], fontsize=6, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([_fmt_batch(b) for b in batches])
    ax.set_xlabel("Batch size (workload)")
    ax.legend(loc="upper left", frameon=True)
    return all_vals


def make_latency_figure(df_one, methods: dict, title: str, agg="median"):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    vals = _grouped_bars(ax, df_one, list(methods), methods, METHOD_COLOR, _fmt_latency, agg)
    if vals:
        # Methods span a wide dynamic range (Triton can be ~100x faster than the
        # torch baseline at large batch), so a log y-axis keeps every bar legible.
        ax.set_yscale("log")
        lo = min(v for v in vals if v > 0)
        ax.set_ylim(lo * 0.5, max(vals) * 3)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def make_speedup_figure(df_one, speedups: dict, title: str, agg="median"):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    vals = _grouped_bars(ax, df_one, list(speedups), speedups, SPEEDUP_COLOR, _fmt_speedup, agg)
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0, zorder=0)  # parity
    if vals:
        ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("Triton speedup (×)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_fig(fig, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
