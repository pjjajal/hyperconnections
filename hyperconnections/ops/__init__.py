from __future__ import annotations

from .expm import expm_t18
from .expm_block import expm_t18_augmented_sparse

### Single authoritative Triton-availability gate.
### Both stream_mix and expm_triton must load for HAS_TRITON = True.
### Callers (cghc.py, cghcf.py, cghc_strang.py) import HAS_TRITON from here;
### there is no other public HAS_TRITON in the package.
try:
    from .stream_mix import stream_mix_add
    from .expm_triton import expm_t18_triton
    HAS_TRITON = True
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    HAS_TRITON = False
    stream_mix_add  = None  # type: ignore[assignment]
    expm_t18_triton = None  # type: ignore[assignment]


__all__ = [
    "HAS_TRITON",
    "stream_mix_add",
    "expm_t18",
    "expm_t18_triton",
    "expm_t18_augmented_sparse",
]
