"""Plot stream_mix benchmark CSVs from benchmark_reports/.

Standalone single-kernel entry point; shares all plotting logic with the suite
driver via ``bench_plot_common``.  Writes a latency and a speedup figure per
(phi_type, variant, N) into ``<out-dir>/stream_mix/n<N>/``, overlaying both
dtypes.
"""

import argparse
import sys
from pathlib import Path

import bench_plot_common as bpc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot stream_mix benchmark CSVs from benchmark_reports/."
    )
    parser.add_argument(
        "csvs", nargs="*", type=Path,
        help="One or more benchmark report CSV files.",
    )
    parser.add_argument(
        "--singles-dir", type=Path, default=None, dest="singles_dir",
        help="Path to benchmark_reports/singles/; auto-discovers all stream_mix CSVs.",
    )
    parser.add_argument(
        "--phi-type", default=None, dest="phi_type",
        help="Filter to a single phi_type (e.g. 'random', 'skew_sym'). Default: plot all.",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Filter to a single variant (e.g. 'fwd', 'no-proj fwd'). Default: plot all.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."),
        help="Root directory for the output tree (default: current directory).",
    )
    args = parser.parse_args()

    spec = bpc.KERNELS["stream_mix"]

    if args.singles_dir:
        subdir = args.singles_dir / spec["subdir"]
        csvs = sorted(subdir.glob(spec["perf_glob"]))
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

    df = bpc.load_csvs(csvs)

    phi_types = sorted(df["phi_type"].unique())
    variants = sorted(df["variant"].unique())
    if args.phi_type:
        phi_types = [p for p in phi_types if p == args.phi_type]
    if args.variant:
        variants = [v for v in variants if v == args.variant]

    for phi_type in phi_types:
        for variant in variants:
            g = df[(df["phi_type"] == phi_type) & (df["variant"] == variant)]
            if g.empty:
                continue
            safe = f"{phi_type.strip().replace(' ', '_')}_{variant.strip().replace(' ', '_')}"
            label = f"{phi_type} / {variant}"
            for n in sorted(g["n"].unique()):
                for dtype in sorted(g["dtype"].unique()):
                    sub = g[(g["n"] == n) & (g["dtype"] == dtype)]
                    out_dir = args.out_dir / "stream_mix" / dtype / f"n{n}"
                    bpc.save_fig(
                        bpc.make_latency_figure(sub, spec["methods"], f"stream_mix latency — {label}  (N={n}, {dtype})"),
                        out_dir / f"stream_mix_latency_{safe}.PNG",
                    )
                    bpc.save_fig(
                        bpc.make_speedup_figure(sub, spec["speedups"], f"stream_mix speedup — {label}  (N={n}, {dtype})"),
                        out_dir / f"stream_mix_speedup_{safe}.PNG",
                    )


if __name__ == "__main__":
    main()
