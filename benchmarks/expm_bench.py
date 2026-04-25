from __future__ import annotations

import argparse
import sys
from itertools import product
from typing import Sequence

import torch
import triton
import triton.testing

from hyperconnections.ops import stream_mix_add


###
### Helpers
###
DEVICE = "cuda:0"

_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"

def _col(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text
def ok(s="PASS"):
    return _col(s, _GREEN)
def fail(s):
    return _col(s, _RED)
def warn(s):
    return _col(s, _YELLOW)
def bold(s):
    return _col(s, _BOLD)
def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]
