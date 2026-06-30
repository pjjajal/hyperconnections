from __future__ import annotations

from .expm import expm_t18
from .expm_block import expm_t18_augmented_sparse

### Single authoritative Triton-availability gate.
### Both stream_mix and expm_triton must load for HAS_TRITON = True.  Detection relies on
### expm_triton.py importing triton *unguarded* — stream_mix.py swallows a missing-triton
### ModuleNotFoundError internally (its own _has_triton), so its import never fails here.
### Keep expm_triton's triton import unguarded or this gate would silently report True.
### Callers cghc.py and cghc_strang.py import HAS_TRITON from here (cghcf.py inherits the
### dispatch from cghc.py); there is no other public HAS_TRITON in the package.
try:
    from .stream_mix import stream_mix_add
    from .expm_triton import expm_t18_triton, expm_t18_block_triton
    HAS_TRITON = True
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise
    HAS_TRITON = False
    stream_mix_add        = None  # type: ignore[assignment]
    expm_t18_triton       = None  # type: ignore[assignment]
    expm_t18_block_triton = None  # type: ignore[assignment]


__all__ = [
    "HAS_TRITON",
    "stream_mix_add",
    "expm_t18",
    "expm_t18_triton",
    "expm_t18_block_triton",
    "expm_t18_augmented_sparse",
]
