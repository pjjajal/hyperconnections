"""Emit a LaTeX numerical-correctness table from the expm / expm_force
correctness CSVs in ``benchmark_reports/``.

Companion to ``generate_all.py`` (which emits the perf *figures*): this script
reduces the per-config correctness checks to a single ``\\begin{table}`` block
ready to paste into the paper. Each table *row* is one ``(kernel output,
reference)`` pair; each *column* is a value of $N$; each *cell* is the
worst-case (max) recorded ``max_err`` over the selected batch sizes, matrix
types and dtypes. Cells with no matching data print ``---``.

    # the default 4-row table from the paper draft, to stdout
    python plots/expm_correctness_table.py

    # a subset of rows + N columns, bf16 only, written to a file
    python plots/expm_correctness_table.py --rows expm_fwd ef_E ef_psi \\
        --n 4 8 16 32 --dtype bf16 --out tab_corr.tex

``max_err`` is the *absolute* max element error from ``_check`` in
``benchmarks/expm_common.py`` ((got - ref).abs().max()), not a relative error;
the default caption is worded accordingly.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

# Correctness-CSV families: a glob (relative to --reports-dir) matching every
# report file for that kernel — current run, baselines, and sweeps.  The expm
# glob is written so it does NOT also match the expm_force files.
SOURCES = {
    "expm":       "**/benchmark_expm_correctness_*.CSV",
    "expm_force": "**/benchmark_expm_force_correctness_*.CSV",
}

# Catalogue of selectable table rows. ``--rows`` picks a subset (and order);
# DEFAULT_ROWS reproduces the paper-draft table. Each entry selects exactly one
# (variant, check) pair from one SOURCES family and carries its two LaTeX label
# columns. The variant/check strings are the literal values written by
# benchmarks/expm_bench.py and benchmarks/expm_force_bench.py.
ROWS = {
    # --- plain expm kernel -------------------------------------------------
    "expm_fwd": {
        "source": "expm", "variant": "triton", "check": "fwd",
        "kernel_tex": r"\texttt{expm\_t18\_triton}",
        "ref_tex":    r"\texttt{matrix\_exp}",
    },
    "expm_grad": {
        "source": "expm", "variant": "triton", "check": "grad_A",
        "kernel_tex": r"\texttt{expm\_t18\_triton} grad",
        "ref_tex":    r"autograd \texttt{matrix\_exp}",
    },
    # --- expm_force (block) kernel: forward E, forward Psi, backward grad ---
    "ef_E": {
        "source": "expm_force", "variant": "triton", "check": "E vs exp",
        "kernel_tex": r"\texttt{expm\_t18\_block\_triton} $E$",
        "ref_tex":    r"\texttt{matrix\_exp}",
    },
    "ef_E_t18": {
        "source": "expm_force", "variant": "triton", "check": "E vs T18",
        "kernel_tex": r"\texttt{expm\_t18\_block\_triton} $E$",
        "ref_tex":    r"compiled T18",
    },
    "ef_psi": {
        "source": "expm_force", "variant": "triton", "check": "psi vs quad",
        "kernel_tex": r"\texttt{expm\_t18\_block\_triton} $\Psi$",
        "ref_tex":    r"GL quadrature (fp64)",
    },
    "ef_psi_t18": {
        "source": "expm_force", "variant": "triton", "check": "psi vs T18",
        "kernel_tex": r"\texttt{expm\_t18\_block\_triton} $\Psi$",
        "ref_tex":    r"compiled T18",
    },
    "ef_grad": {
        "source": "expm_force", "variant": "triton", "check": "grad vs aug",
        "kernel_tex": r"backward grad",
        "ref_tex":    r"autograd through $2N$ exp",
    },
    "ef_grad_t18": {
        "source": "expm_force", "variant": "triton", "check": "grad vs T18",
        "kernel_tex": r"backward grad",
        "ref_tex":    r"compiled T18",
    },
}
DEFAULT_ROWS = ["expm_fwd", "ef_E", "ef_psi", "ef_grad"]

# Matrix types dropped unless --matrix-types is given explicitly. The large-norm
# ("lrg") stress case has |exp(A)| ~ 1e4, so its raw *absolute* max_err (~2e4)
# swamps every other type (all O(0.1)) and is meaningless without normalising by
# the reference magnitude — which the CSV does not store. Excluding it keeps the
# reported absolute error representative, mirroring `drop_variants` in
# bench_plot_common. Pass --matrix-types lrg ... to include it.
DROP_MATRIX_TYPES = ("lrg",)

DEFAULT_CAPTION = (
    r"Numerical correctness of Triton kernels vs.\ independent references. "
    r"Worst-case (max) absolute error over the listed batch sizes, matrix "
    r"types and dtypes, per $N$. Lower is better."
)

# config cells look like "[diag] B=1024 N=16 bf16" (matrix-type prefix optional
# -> "rand"; dtype suffix). Note: one stray report has a "B=32678" typo, which
# the max aggregation simply absorbs.
_CFG_RE = re.compile(
    r"(?:\[(?P<mt>[^\]]+)\]\s*)?B=(?P<b>\d+)\s+N=(?P<n>\d+)\s+(?P<dtype>\w+)"
)


def parse_corr_config(cfg: str) -> dict:
    m = _CFG_RE.search(cfg)
    if not m:
        raise ValueError(f"unparseable config: {cfg!r}")
    return {
        "matrix_type": (m.group("mt") or "rand").strip(),
        "batch":       int(m.group("b")),
        "n":           int(m.group("n")),
        "dtype":       m.group("dtype"),
    }


def load_source(reports_dir: Path, glob: str) -> pd.DataFrame:
    """Read + concat every correctness CSV for one family, adding parsed
    ``matrix_type`` / ``batch`` / ``n`` / ``dtype`` columns. Empty if none."""
    paths = sorted(reports_dir.glob(glob))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    parsed = df["config"].apply(lambda c: pd.Series(parse_corr_config(c)))
    return pd.concat([df, parsed], axis=1)


def cell_value(df, variant, check, n, batches, matrix_types, dtypes):
    """Worst (max) ``max_err`` for one (variant, check, N) under the filters,
    or ``None`` when no row matches."""
    if df.empty:
        return None
    sub = df[(df["variant"] == variant) & (df["check"] == check) & (df["n"] == n)]
    if batches:
        sub = sub[sub["batch"].isin(batches)]
    if matrix_types:
        sub = sub[sub["matrix_type"].isin(matrix_types)]
    if dtypes:
        sub = sub[sub["dtype"].isin(dtypes)]
    sub = sub[sub["max_err"].notna()]
    if sub.empty:
        return None
    return float(sub["max_err"].max())


def fmt_err(v) -> str:
    """Render an error value as LaTeX scientific notation ($m{\\times}10^{e}$)."""
    if v is None:
        return "---"
    if v == 0:
        return r"$0$"
    exp = math.floor(math.log10(abs(v)))
    mant = v / 10 ** exp
    return rf"${mant:.1f}{{\times}}10^{{{exp}}}$"


def build_table(caches, row_keys, ns, batches, matrix_types, dtypes,
                caption, label) -> str:
    specs = [ROWS[k] for k in row_keys]
    # Pad the two label columns so the emitted LaTeX source lines up like the
    # paper draft (purely cosmetic; ignored by LaTeX).
    kw = max((len(s["kernel_tex"]) for s in specs), default=0)
    rw = max((len(s["ref_tex"]) for s in specs), default=0)

    col_spec = "l" + "c" * (1 + len(ns))
    head_ns = " & ".join(rf"$N{{=}}{n}$" for n in ns)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Kernel & Reference & {head_ns} \\",
        r"\midrule",
    ]
    for spec, key in zip(specs, row_keys):
        df = caches[spec["source"]]
        cells = []
        for n in ns:
            v = cell_value(df, spec["variant"], spec["check"], n,
                           batches, matrix_types, dtypes)
            if v is None:
                print(f"[warn] no data for row '{key}' at N={n}", file=sys.stderr)
            cells.append(fmt_err(v))
        lines.append(
            f"{spec['kernel_tex'].ljust(kw)} & {spec['ref_tex'].ljust(rw)} & "
            + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a LaTeX numerical-correctness table from "
                    "benchmark_reports/ correctness CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("benchmark_reports"),
        help="Root to glob correctness CSVs from (default: benchmark_reports).",
    )
    parser.add_argument(
        "--rows", nargs="+", choices=list(ROWS), default=DEFAULT_ROWS,
        metavar="ROW",
        help="Subset/order of table rows (default: the paper-draft 4). "
             "Choices: " + ", ".join(ROWS),
    )
    parser.add_argument(
        "--n", nargs="+", type=int, default=[4, 8, 16, 32], metavar="N",
        help="N values, one column each (default: 4 8 16 32).",
    )
    parser.add_argument(
        "--batches", nargs="*", type=int, default=None, metavar="B",
        help="Restrict to these batch sizes (default: all present).",
    )
    parser.add_argument(
        "--matrix-types", nargs="*", default=None, dest="matrix_types",
        metavar="T",
        help="Restrict to these matrix types, e.g. rand skew npsd diag lrg "
             "(default: all present).",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp32", "all"], default="all",
        help="Restrict to one dtype, or 'all' for the worst case (default: all).",
    )
    parser.add_argument(
        "--caption", default=DEFAULT_CAPTION,
        help="Table caption (default: the absolute-error wording).",
    )
    parser.add_argument(
        "--label", default="tab:expm_corr",
        help="LaTeX \\label (default: tab:expm_corr).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write the .tex here (default: stdout).",
    )
    args = parser.parse_args()

    if not args.reports_dir.exists():
        print(f"reports dir not found: {args.reports_dir}", file=sys.stderr)
        sys.exit(1)

    dtypes = None if args.dtype == "all" else [args.dtype]
    needed = {ROWS[k]["source"] for k in args.rows}
    caches = {src: load_source(args.reports_dir, SOURCES[src]) for src in needed}
    for src, df in caches.items():
        if df.empty:
            print(f"[warn] no correctness CSVs for source '{src}' under "
                  f"{args.reports_dir} (matching {SOURCES[src]})", file=sys.stderr)

    if args.matrix_types is not None:
        matrix_types = args.matrix_types
    else:
        present = sorted(set().union(*(set(df["matrix_type"].unique())
                                       for df in caches.values() if not df.empty)))
        matrix_types = [t for t in present if t not in DROP_MATRIX_TYPES]
        dropped = [t for t in present if t in DROP_MATRIX_TYPES]
        if dropped:
            print(f"[info] excluding matrix types {dropped} by default "
                  f"(pass --matrix-types to override)", file=sys.stderr)

    table = build_table(
        caches, args.rows, args.n, args.batches, matrix_types, dtypes,
        args.caption, args.label,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
        print(f"Wrote: {args.out}", file=sys.stderr)
    else:
        print(table)


if __name__ == "__main__":
    main()
