"""
Driver for the sandboxed kernel-code research loop.

One invocation = evaluate the CURRENT sandbox candidate for a target and post the
result row to that target's forum.  This is Layer 1 (mechanical) plumbing; the
creative work — editing sandbox/<...>_candidate.py with a new kernel idea — is done
by the agent (Layer 2) before calling this.

    # fresh start: copy canonical kernels into the sandbox
    python -m hyperconnections.kernel_research.auto_research --target expm_force --reseed

    # evaluate whatever is in the sandbox now and post to the forum
    python -m hyperconnections.kernel_research.auto_research --target stream_mix --pass bwd --note "fused dPhi"

Targets are discovered from the registry in targets.py, so this driver never needs
editing when a new kernel target is added.  Run on a GPU node.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from hyperconnections.kernel_research.targets import TARGETS, reseed

_THIS         = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the current sandbox candidate and post to its forum.")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--reseed", action="store_true", help="recopy canonical kernel into the sandbox, then exit")
    ap.add_argument("--pass", dest="direction", choices=["fwd", "bwd"], default="bwd")
    ap.add_argument("--no-post", dest="post", action="store_false", default=True,
                    help="evaluate only; do not append to the forum")
    ap.add_argument("--note", default="", help="one-line idea, forwarded to the forum row")
    ### pass-through size flags (see evaluate.py)
    ap.add_argument("--n", type=int, nargs="+", default=None)
    ap.add_argument("--b", type=int, default=None)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--proj", action="store_true", default=False)
    args = ap.parse_args()

    if args.reseed:
        reseed(args.target)
        return

    cmd = [sys.executable, "-m", "hyperconnections.kernel_research.evaluate",
           "--target", args.target, "--pass", args.direction]
    if args.post:
        cmd.append("--post")
    if args.note:
        cmd += ["--note", args.note]
    if args.n is not None:
        cmd += ["--n", *map(str, args.n)]
    if args.b is not None:
        cmd += ["--b", str(args.b)]
    if args.d is not None:
        cmd += ["--d", str(args.d)]
    if args.proj:
        cmd.append("--proj")

    ### Subprocess (not import) so each evaluation is a clean process — same reason
    ### the existing bench_sweep.sh runs one subprocess per config.
    subprocess.run(cmd, cwd=_PROJECT_ROOT, check=False)


if __name__ == "__main__":
    main()
