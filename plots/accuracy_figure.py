"""Compact main-paper numerical-accuracy figure for the matrix-exponential
kernels (plain ``expm`` = impulse variant, ``expm_force`` = constant-forcing).

Two panels, curated from ``matrix_exponential_accuracy_figures.md``:

  A  Empirical CDF of relative error, one curve per Triton kernel output
     (exp(A), block ``E``, ``Psi``, grad), with guide lines at 1e-3/1e-4/1e-5.
     "Curves further left = lower error"; pooled over N, dtype and matrix type.

  B  Accuracy-runtime Pareto (the money plot): x = speedup over
     ``torch.linalg.matrix_exp``, y = relative error, one point per
     (kernel output, N, dtype). Lower-right = faster AND more accurate.

Reads the correctness CSVs (``rel_err`` column, added by expm_common._check) and
the perf CSVs (``speedup_vs_torch``) produced by ``scripts/jobs/full_eval.sh``
under ``--reports-dir``. Standalone; borrows the Agg backend + rcParams +
``save_fig`` from ``bench_plot_common`` so styling matches the rest of the suite,
and the field-wise config parser from ``expm_correctness_table``.

    python plots/accuracy_figure.py --reports-dir benchmark_reports/full_eval_2026_07_08 \
        --out plots/accuracy_figure.png
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import bench_plot_common as bpc            # sets Agg backend + shared rcParams
import expm_correctness_table as ect        # field-wise parse_corr_config + DROP_MATRIX_TYPES
import matplotlib.pyplot as plt

# One entry per error series: label, correctness glob, (variant, check) selecting
# the Triton-vs-ground-truth row, the matching perf glob, a colour, whether it is
# a forward output (→ appears in the Pareto, which compares against matrix_exp).
_SERIES = [
    ("expm exp(A)",   "**/benchmark_expm_correctness_*.CSV",
        "triton", "fwd",                  "**/benchmark_expm_perf_*.CSV",
        "#2ca02c", True),
    ("expm_force E",  "**/benchmark_expm_force_correctness_*.CSV",
        "triton", "vs linalg.matrix_exp", "**/benchmark_expm_force_perf_*.CSV",
        "#1f77b4", True),
    ("expm_force Ψ", "**/benchmark_expm_force_correctness_*.CSV",
        "triton", "vs Gauss-Legendre",    "**/benchmark_expm_force_perf_*.CSV",
        "#9467bd", False),
    ("expm_force grad", "**/benchmark_expm_force_correctness_*.CSV",
        "triton", "vs autograd",          "**/benchmark_expm_force_perf_*.CSV",
        "#d62728", False),
]

_N_MARKER = {4: "o", 8: "s", 16: "^", 32: "D"}
_EPS = np.finfo(float).tiny


def _parse_cfg(cfg):
    try:
        return ect.parse_corr_config(cfg)
    except Exception:
        return None


def load_corr(reports_dir: Path, glob: str, variant: str, check: str,
              dtypes=None) -> pd.DataFrame:
    """Correctness rows for one (variant, check), with matrix_type/n/dtype parsed
    and the large-norm stress case dropped. Uses rel_err (falls back to max_err)."""
    paths = sorted(reports_dir.glob(glob))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[(df["variant"] == variant) & (df["check"] == check)]
    if df.empty:
        return pd.DataFrame()

    recs = df["config"].map(_parse_cfg)
    keep = recs.notna()
    df, recs = df[keep].reset_index(drop=True), recs[keep].reset_index(drop=True)
    df["matrix_type"] = recs.map(lambda p: p["matrix_type"])
    df["n"] = recs.map(lambda p: p["n"])
    df["dtype"] = recs.map(lambda p: p["dtype"])

    df = df[~df["matrix_type"].isin(ect.DROP_MATRIX_TYPES)]
    if dtypes:
        df = df[df["dtype"].isin(dtypes)]
    err_col = "rel_err" if "rel_err" in df.columns else "max_err"
    df = df[df[err_col].notna()].copy()
    df["err"] = df[err_col].astype(float).clip(lower=_EPS)
    return df


def load_perf(reports_dir: Path, glob: str) -> pd.DataFrame:
    """Perf rows with n/batch/direction parsed (direction from the atype suffix)."""
    paths = sorted(reports_dir.glob(glob))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["n"] = df["config"].str.extract(r"N=(\d+)").astype(int)
    df["batch"] = df["config"].str.extract(r"B=(\d+)").astype(int)
    df["direction"] = df["atype"].str.rsplit(" ", n=1).str[-1].str.strip()
    return df


def fwd_speedup(perf: pd.DataFrame, n: int, dtype: str):
    """Median forward speedup over matrix_exp at the largest available batch."""
    if perf.empty:
        return None
    sub = perf[(perf["n"] == n) & (perf["dtype"] == dtype) & (perf["direction"] == "fwd")]
    sub = sub[sub["speedup_vs_torch"].notna()]
    if sub.empty:
        return None
    sub = sub[sub["batch"] == sub["batch"].max()]
    return float(sub["speedup_vs_torch"].median())


def _ecdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    return v, np.arange(1, len(v) + 1) / len(v)


def panel_cdf(ax, corr_by_series):
    for label, df, colour in corr_by_series:
        if df.empty:
            continue
        x, y = _ecdf(df["err"].to_numpy())
        ax.step(x, y, where="post", color=colour, label=label, linewidth=1.6)
    for guide in (1e-3, 1e-4, 1e-5):
        ax.axvline(guide, color="0.7", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Relative error")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.02)
    ax.set_title("(A) Error distribution")
    ax.legend(loc="lower right", frameon=True, fontsize=9)


def panel_pareto(ax, reports_dir, dtypes):
    perf_cache = {}
    plotted_n, plotted_series, plotted_dtype = set(), [], set()
    for label, cglob, variant, check, pglob, colour, is_fwd in _SERIES:
        if not is_fwd:
            continue
        corr = load_corr(reports_dir, cglob, variant, check, dtypes)
        if corr.empty:
            continue
        if pglob not in perf_cache:
            perf_cache[pglob] = load_perf(reports_dir, pglob)
        perf = perf_cache[pglob]
        plotted_series.append((label, colour))
        # error representative: median rel-err over matrix types at (n, dtype)
        grp = corr.groupby(["n", "dtype"])["err"].median().reset_index()
        for _, r in grp.iterrows():
            n, dtype = int(r["n"]), r["dtype"]
            sp = fwd_speedup(perf, n, dtype)
            if sp is None:
                continue
            filled = dtype == "bf16"
            ax.scatter(sp, r["err"], marker=_N_MARKER.get(n, "o"), s=60,
                       facecolors=colour if filled else "none",
                       edgecolors=colour, linewidths=1.4, zorder=3)
            plotted_n.add(n); plotted_dtype.add(dtype)

    ax.axvline(1.0, color="0.4", linestyle="--", linewidth=1.0, zorder=0)  # parity w/ matrix_exp
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Speedup over torch.matrix_exp ($\times$)")
    ax.set_ylabel("Relative error")
    ax.set_title("(B) Accuracy vs. runtime")

    # Two compact legends: colour = kernel output, marker = N, fill = dtype.
    from matplotlib.lines import Line2D
    series_handles = [Line2D([], [], marker="o", linestyle="none", color=c, label=l)
                      for l, c in plotted_series]
    n_handles = [Line2D([], [], marker=_N_MARKER.get(n, "o"), linestyle="none",
                        color="0.3", label=f"N={n}") for n in sorted(plotted_n)]
    dtype_handles = []
    if "fp32" in plotted_dtype:
        dtype_handles.append(Line2D([], [], marker="o", linestyle="none",
                                    markerfacecolor="none", markeredgecolor="0.3", label="fp32"))
    if "bf16" in plotted_dtype:
        dtype_handles.append(Line2D([], [], marker="o", linestyle="none",
                                    color="0.3", label="bf16"))
    leg1 = ax.legend(handles=series_handles, loc="upper right", frameon=True, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=n_handles + dtype_handles, loc="lower left", frameon=True, fontsize=8)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compact accuracy figure (CDF + Pareto).")
    p.add_argument("--reports-dir", type=Path, default=Path("benchmark_reports"),
                   help="Root to glob correctness + perf CSVs from.")
    p.add_argument("--dtype", nargs="+", choices=["fp32", "bf16", "fp16"], default=None,
                   help="Restrict to these dtypes (default: all present, pooled).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output figure path (default: <reports-dir>/accuracy_figure_<today>.png).")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args(argv)

    if not args.reports_dir.exists():
        print(f"reports dir not found: {args.reports_dir}", file=sys.stderr)
        sys.exit(1)
    plt.rcParams.update({"savefig.dpi": args.dpi, "figure.dpi": args.dpi})

    corr_by_series = [
        (label, load_corr(args.reports_dir, cglob, variant, check, args.dtype), colour)
        for (label, cglob, variant, check, _pglob, colour, _fwd) in _SERIES
    ]
    if all(df.empty for _, df, _ in corr_by_series):
        print(f"[error] no correctness rows under {args.reports_dir}; nothing to plot.",
              file=sys.stderr)
        sys.exit(1)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    panel_cdf(axA, corr_by_series)
    panel_pareto(axB, args.reports_dir, args.dtype)

    out = args.out or (args.reports_dir / f"accuracy_figure_{date.today():%Y_%m_%d}.png")
    bpc.save_fig(fig, out)


if __name__ == "__main__":
    main()
