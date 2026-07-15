"""
Forward/backward evaluator for sandboxed kernel candidates — the script the
research agents invoke.  Thin CLI over the framework in targets.py.

For any registered target it imports the sandbox candidate (reloaded each call),
checks it against the target's trusted reference, and times it against the
canonical baseline in the same process.  Forward vs backward is just `--pass`.

  - Report, never sys.exit, on a correctness FAIL or a broken candidate edit
    (a syntax/compile error in the sandbox becomes an ERR row, not a crash).
  - One row per (target, size); --post appends each row to the target's forum.

Usage (on a GPU node):
    python -m hyperconnections.kernel_research.evaluate --target expm_force --pass bwd
    python -m hyperconnections.kernel_research.evaluate --target stream_mix --pass fwd --post --note "fused dPhi"
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import uuid

import torch

from hyperconnections.kernel_research.targets import TARGETS, set_bench

### project root + forum CLI (targets.py already put benchmarks/ on sys.path)
_THIS         = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_SCRATCH_DIR  = os.path.dirname(_PROJECT_ROOT)
_FORUM_CLI    = [sys.executable, os.path.join(_SCRATCH_DIR, "pj-forum", "forum-skill", "scripts", "forum")]
_FORUM_AGENT  = "KernelResearcher"


def _cand_id() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _row(target, r, note) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    tail = f" | {note}" if note else ""
    return (
        f"CAND {_cand_id()} | {target} | {r['size']} | "
        f"cand={r['cand']:.4f}ms | base={r['base']:.4f}ms | "
        f"speedup={r['speedup']:.2f}x | err={r['err']:.2e} | {status}{tail}"
    )


def _err_row(target, size, note, exc) -> str:
    tail = f" | {note}" if note else ""
    return (
        f"CAND {_cand_id()} | {target} | {size} | cand=nan | base=nan | "
        f"speedup=nan | err=nan | ERR:{type(exc).__name__}: {exc}{tail}"
    )


def _forum_append(forum, msg) -> None:
    subprocess.run(
        _FORUM_CLI + ["append", "--agent", _FORUM_AGENT, "--forum", forum, "-m", msg],
        cwd=_PROJECT_ROOT, check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a sandboxed kernel candidate (bf16).")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--pass", dest="direction", choices=["fwd", "bwd"], default="bwd",
                    help="forward or backward (default bwd)")
    ap.add_argument("--n", type=int, nargs="+", default=None, help="problem size(s) (target-specific)")
    ap.add_argument("--b", type=int, default=None, help="batch size")
    ap.add_argument("--d", type=int, default=None, help="feature dim (targets that use it)")
    ap.add_argument("--proj", action="store_true", default=False, help="projection variant (targets that use it)")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--rep", type=int, default=128)
    ap.add_argument("--note", default="", help="one-line idea, appended to the forum row")
    ap.add_argument("--post", action="store_true", default=False, help="append rows to the target forum")
    ap.add_argument("--json", action="store_true", default=False, help="also print JSON results")
    args = ap.parse_args()

    set_bench(args.warmup, args.rep)
    if not torch.cuda.is_available():
        sys.exit("No CUDA device available — run evaluate.py on a GPU node.")

    target = TARGETS[args.target]
    rows, results = [], []
    for size in target.sizes_from_args(args):
        try:
            r = target.evaluate(args.direction, size)
            row = _row(target.name, r, args.note)
            results.append(r)
        except Exception as exc:  # broken candidate / OOM / compile error → report, don't crash
            torch.cuda.empty_cache()
            row = _err_row(target.name, target.size_label(size, args.direction), args.note, exc)
        print(row, flush=True)
        rows.append(row)

    if args.json:
        print(json.dumps(results, indent=2))

    if args.post:
        for row in rows:
            _forum_append(target.name, row)
        print(f"[evaluate] posted {len(rows)} row(s) to forum '{target.name}'")


if __name__ == "__main__":
    main()
