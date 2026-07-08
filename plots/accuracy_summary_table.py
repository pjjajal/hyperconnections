"""Small aggregate accuracy/runtime summary table (Approach 5 of
``matrix_exponential_accuracy_figures.md``): one LaTeX table whose rows are the
Triton kernel outputs and whose columns are median / p95 / max relative error
plus the median speedup over ``torch.linalg.matrix_exp``.

This is the *main-text* companion to the per-configuration correctness table
(``expm_correctness_table.py``, which belongs in the appendix). Relative error
comes from the ``rel_err`` column of the correctness CSVs (pooled over N and
matrix type); speedup is the median of ``speedup_vs_torch`` over the swept batch
sizes for the matching pass direction.

    python plots/accuracy_summary_table.py --reports-dir benchmark_reports/full_eval_2026_07_08 \
        --out plots/accuracy_summary.tex
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import expm_correctness_table as ect        # parse_corr_config, DROP_MATRIX_TYPES, fmt_err

# (method label, corr glob, variant, check, perf glob, pass direction for speedup)
_ROWS = [
    (r"Triton \texttt{expm} $\exp(A)$",     "**/benchmark_expm_correctness_*.CSV",
        "triton", "fwd",                  "**/benchmark_expm_perf_*.CSV",       "fwd"),
    (r"Triton \texttt{expm\_force} $E$",     "**/benchmark_expm_force_correctness_*.CSV",
        "triton", "vs linalg.matrix_exp", "**/benchmark_expm_force_perf_*.CSV", "fwd"),
    (r"Triton \texttt{expm\_force} $\Psi$",  "**/benchmark_expm_force_correctness_*.CSV",
        "triton", "vs Gauss-Legendre",    "**/benchmark_expm_force_perf_*.CSV", "fwd"),
    (r"Triton \texttt{expm\_force} grad",    "**/benchmark_expm_force_correctness_*.CSV",
        "triton", "vs autograd",          "**/benchmark_expm_force_perf_*.CSV", "bwd"),
]

_DEFAULT_CAPTION = (
    r"Aggregate numerical accuracy and runtime of the Triton matrix-exponential "
    r"kernels. Relative error is pooled over $N$ and matrix type; speedup is the "
    r"median over the swept batch sizes vs.\ \texttt{torch.linalg.matrix\_exp}."
)


def _parse_cfg(cfg):
    try:
        return ect.parse_corr_config(cfg)
    except Exception:
        return None


def load_corr(reports_dir, glob, variant, check):
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
    df["dtype"] = recs.map(lambda p: p["dtype"])
    df = df[~df["matrix_type"].isin(ect.DROP_MATRIX_TYPES)]
    err_col = "rel_err" if "rel_err" in df.columns else "max_err"
    df = df.rename(columns={err_col: "err"})
    df = df[df["err"].notna()]
    return df


def load_perf(reports_dir, glob):
    paths = sorted(reports_dir.glob(glob))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["direction"] = df["atype"].str.rsplit(" ", n=1).str[-1].str.strip()
    return df


def _fmt_speedup(v):
    return "---" if v is None or v != v else rf"${v:.1f}{{\times}}$"


def build_rows(reports_dir, dtypes):
    """Return (dtype, method_tex, med, p95, mx, speedup) tuples for the table."""
    perf_cache = {}
    out = []
    for method, cglob, variant, check, pglob, direction in _ROWS:
        corr = load_corr(reports_dir, cglob, variant, check)
        if pglob not in perf_cache:
            perf_cache[pglob] = load_perf(reports_dir, pglob)
        perf = perf_cache[pglob]
        for dt in dtypes:
            ce = corr[corr["dtype"] == dt]["err"].astype(float) if not corr.empty else pd.Series(dtype=float)
            if len(ce):
                med, p95, mx = float(ce.median()), float(np.percentile(ce, 95)), float(ce.max())
            else:
                med = p95 = mx = None
            sp = None
            if not perf.empty:
                ps = perf[(perf["dtype"] == dt) & (perf["direction"] == direction)]["speedup_vs_torch"]
                ps = ps.dropna()
                if len(ps):
                    sp = float(ps.median())
            out.append((dt, method, med, p95, mx, sp))
    return out


def build_table(rows, dtypes, caption, label) -> str:
    multi = len(dtypes) > 1
    if multi:
        header = r"Method & Dtype & Median & p95 & Max & Speedup \\"
        col_spec = "llcccc"
    else:
        header = r"Method & Median & p95 & Max & Speedup \\"
        col_spec = "lcccc"
    lines = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule", header, r"\midrule",
    ]
    for dt, method, med, p95, mx, sp in rows:
        cells = [ect.fmt_err(med), ect.fmt_err(p95), ect.fmt_err(mx), _fmt_speedup(sp)]
        prefix = f"{method} & {dt} & " if multi else f"{method} & "
        lines.append(prefix + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Emit the small aggregate accuracy/runtime table.")
    p.add_argument("--reports-dir", type=Path, default=Path("benchmark_reports"),
                   help="Root to glob correctness + perf CSVs from.")
    p.add_argument("--dtype", choices=["fp32", "bf16", "all"], default="all",
                   help="One dtype, or 'all' for a dtype column (default: all).")
    p.add_argument("--caption", default=_DEFAULT_CAPTION)
    p.add_argument("--label", default="tab:accuracy_summary")
    p.add_argument("--out", type=Path, default=None, help="Write .tex here (default: stdout).")
    args = p.parse_args(argv)

    if not args.reports_dir.exists():
        print(f"reports dir not found: {args.reports_dir}", file=sys.stderr)
        sys.exit(1)

    dtypes = ["fp32", "bf16"] if args.dtype == "all" else [args.dtype]
    rows = build_rows(args.reports_dir, dtypes)
    if all(med is None and sp is None for _dt, _m, med, _p, _mx, sp in rows):
        print(f"[warn] no data under {args.reports_dir}; table will be all '---'.", file=sys.stderr)
    table = build_table(rows, dtypes, args.caption, args.label)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
        print(f"Wrote: {args.out}", file=sys.stderr)
    else:
        print(table)


if __name__ == "__main__":
    main()
