"""Shared formatting / dtype helpers for all benchmark scripts."""
from __future__ import annotations

import logging
import statistics
import sys
from dataclasses import dataclass

import numpy as np
import torch
import triton.testing

DEVICE = "cuda:0"

_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"


def _col(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text

def ok(s="PASS"): return _col(s, _GREEN)
def fail(s):      return _col(s, _RED)
def warn(s):      return _col(s, _YELLOW)
def bold(s):      return _col(s, _BOLD)


logger = logging.getLogger("bench")

def setup_logging(verbose: bool) -> None:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


###
### Per-iteration timing statistics
###
@dataclass(frozen=True)
class BenchStats:
    """Summary of one do_bench run's per-iteration times (all in ms)."""
    median: float
    p99: float
    var: float
    mean: float


def bench_stats(fn, warmup, rep, grad_to_none=None) -> BenchStats:
    """Benchmark `fn` once and return median / P99 / variance / mean of its
    per-iteration runtimes.

    Uses do_bench(return_mode="all"), which returns the raw list of timings from
    a single benchmark run, so all four stats come at no extra timing cost.
    (`quantiles=` must stay unset — it would override `return_mode`.)
    """
    times = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none, return_mode="all"
    )
    return BenchStats(
        median=statistics.median(times),
        p99=float(np.percentile(times, 99)),
        var=statistics.pvariance(times),
        mean=statistics.mean(times),
    )


def stat_fields(prefix: str, s: BenchStats) -> dict:
    """Expand a BenchStats into CSV columns prefixed by the variant name."""
    return {
        f"{prefix}_median_ms": s.median,
        f"{prefix}_p99_ms":    s.p99,
        f"{prefix}_var":       s.var,
        f"{prefix}_mean_ms":   s.mean,
    }


def _corr_row(
    config, variant, check, max_err, atol, passed,
    config_width: int = 34, variant_width: int = 10, check_width: int = 10,
) -> str:
    result = ok("PASS") if passed else fail("FAIL")
    return (
        f"{config:>{config_width}}  {variant:>{variant_width}}  {check:>{check_width}}"
        f"  {max_err:>10.2e}  {atol:>8.0e}  {result}"
    )
