"""Shared formatting / dtype helpers for all benchmark scripts."""
from __future__ import annotations

import sys

import torch

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


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _corr_row(
    config, variant, check, max_err, atol, passed,
    config_width: int = 34, variant_width: int = 10,
) -> str:
    result = ok("PASS") if passed else fail("FAIL")
    return (
        f"{config:>{config_width}}  {variant:>{variant_width}}  {check:>10}"
        f"  {max_err:>10.2e}  {atol:>8.0e}  {result}"
    )
