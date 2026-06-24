"""Regenerate the full suite of benchmark plots in one command.

By default this emits compact *summary* plots that average over the input
matrix types (expm/expm_force) or phi types (stream_mix), so each kernel is
reduced to a handful of figures: one latency + one speedup bar chart per
(direction-or-variant, N, dtype). Pass ``--per-type`` to additionally emit the
detailed per-type charts under a ``by_type/`` subfolder.

    python plots/generate_all.py
    python plots/generate_all.py --per-type
    python plots/generate_all.py --kernels stream_mix --out-dir /tmp/plots
"""

import argparse
import sys
from pathlib import Path

import bench_plot_common as bpc


def _safe(value) -> str:
    return str(value).strip().replace(" ", "_")


def _emit(df, methods, speedups, base_dir, prefix, latency_title, speedup_title, agg):
    bpc.save_fig(
        bpc.make_latency_figure(df, methods, latency_title, agg=agg),
        base_dir / f"{prefix}_latency.PNG",
    )
    bpc.save_fig(
        bpc.make_speedup_figure(df, speedups, speedup_title, agg=agg),
        base_dir / f"{prefix}_speedup.PNG",
    )
    return 2


def generate_kernel(kernel, spec, singles_dir, out_root, per_type=False) -> int:
    subdir = singles_dir / spec["subdir"]
    csvs = sorted(subdir.glob(spec["perf_glob"]))
    if not csvs:
        print(f"[warn] no perf CSVs under {subdir}; skipping {kernel}", file=sys.stderr)
        return 0

    raw = bpc.load_csvs(csvs)
    written = 0

    # --- Summary: averaged over input matrix / phi types -------------------
    sdf = bpc.summarise(raw, spec)
    avg = spec["avg_label"]
    for sgroup, g in sdf.groupby("summary_group"):
        for n in sorted(g["n"].unique()):
            for dtype in sorted(g["dtype"].unique()):
                sub = g[(g["n"] == n) & (g["dtype"] == dtype)]
                base = out_root / kernel / dtype / f"n{n}"
                written += _emit(
                    sub, spec["methods"], spec["speedups"], base,
                    f"{kernel}_avg_{_safe(sgroup)}",
                    f"{kernel} latency — avg over {avg}, {sgroup}  (N={n}, {dtype})",
                    f"{kernel} speedup — avg over {avg}, {sgroup}  (N={n}, {dtype})",
                    agg="mean",
                )

    # --- Optional detailed per-type charts ---------------------------------
    if per_type:
        for key, g in raw.groupby(spec["group_cols"]):
            key = key if isinstance(key, tuple) else (key,)
            label = " / ".join(str(k) for k in key)
            tag = "_".join(_safe(k) for k in key)
            for n in sorted(g["n"].unique()):
                for dtype in sorted(g["dtype"].unique()):
                    sub = g[(g["n"] == n) & (g["dtype"] == dtype)]
                    base = out_root / kernel / dtype / f"n{n}" / "by_type"
                    written += _emit(
                        sub, spec["methods"], spec["speedups"], base,
                        f"{kernel}_{tag}",
                        f"{kernel} latency — {label}  (N={n}, {dtype})",
                        f"{kernel} speedup — {label}  (N={n}, {dtype})",
                        agg="median",
                    )

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the benchmark plot suite for all kernels."
    )
    parser.add_argument(
        "--singles-dir", type=Path, default=Path("benchmark_reports/singles"),
        help="Path to benchmark_reports/singles/ (default: benchmark_reports/singles).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("plots"),
        help="Root directory for the output tree (default: plots).",
    )
    parser.add_argument(
        "--kernels", nargs="*", choices=list(bpc.KERNELS), default=list(bpc.KERNELS),
        help="Subset of kernels to plot (default: all).",
    )
    parser.add_argument(
        "--per-type", action="store_true", dest="per_type",
        help="Also emit detailed per-matrix-type / per-phi-type charts under by_type/.",
    )
    args = parser.parse_args()

    if not args.singles_dir.exists():
        print(f"singles dir not found: {args.singles_dir}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for kernel in args.kernels:
        total += generate_kernel(
            kernel, bpc.KERNELS[kernel], args.singles_dir, args.out_dir, args.per_type
        )

    print(f"Done. Wrote {total} figures under {args.out_dir}/")


if __name__ == "__main__":
    main()
