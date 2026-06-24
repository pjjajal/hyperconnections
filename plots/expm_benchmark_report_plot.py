"""Plot expm / expm_force benchmark CSVs from benchmark_reports/.

Standalone single-kernel entry point; shares all plotting logic with the suite
driver via ``bench_plot_common``.  Writes a latency and a speedup figure per
(matrix type, N) into ``<out-dir>/<bench>/n<N>/``, overlaying both dtypes.
"""

import argparse
import sys
from pathlib import Path

import bench_plot_common as bpc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot expm / expm_force benchmark CSVs from benchmark_reports/."
    )
    parser.add_argument(
        "csvs", nargs="*", type=Path,
        help="One or more benchmark report CSV files.",
    )
    parser.add_argument(
        "--bench", choices=["expm", "expm_force"], default="expm",
        help="Which benchmark variant to plot (default: expm).",
    )
    parser.add_argument(
        "--singles-dir", type=Path, default=None, dest="singles_dir",
        help="Path to benchmark_reports/singles/; auto-discovers CSVs for --bench.",
    )
    parser.add_argument(
        "--matrix-type", default=None, dest="matrix_type",
        help="Filter to a single matrix type (e.g. 'rand fwd'). Default: plot all.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."),
        help="Root directory for the output tree (default: current directory).",
    )
    args = parser.parse_args()

    spec = bpc.KERNELS[args.bench]

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

    atypes = sorted(df["atype"].unique())
    if args.matrix_type:
        atypes = [a for a in atypes if a == args.matrix_type]

    for atype in atypes:
        g = df[df["atype"] == atype]
        safe = atype.strip().replace(" ", "_")
        for n in sorted(g["n"].unique()):
            for dtype in sorted(g["dtype"].unique()):
                sub = g[(g["n"] == n) & (g["dtype"] == dtype)]
                out_dir = args.out_dir / args.bench / dtype / f"n{n}"
                bpc.save_fig(
                    bpc.make_latency_figure(sub, spec["methods"], f"{args.bench} latency — {atype}  (N={n}, {dtype})"),
                    out_dir / f"{args.bench}_latency_{safe}.PNG",
                )
                bpc.save_fig(
                    bpc.make_speedup_figure(sub, spec["speedups"], f"{args.bench} speedup — {atype}  (N={n}, {dtype})"),
                    out_dir / f"{args.bench}_speedup_{safe}.PNG",
                )


if __name__ == "__main__":
    main()
