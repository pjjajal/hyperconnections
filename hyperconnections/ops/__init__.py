from .stream_mix import HAS_TRITON, stream_mix_add
from .expm import expm_t18
from .expm_block import expm_t18_augmented_sparse

# Conditionally import triton-based functions
if HAS_TRITON:
    from .expm_triton import expm_t18_triton
else:
    def expm_t18_triton(*args, **kwargs):
        raise RuntimeError("expm_t18_triton requires Triton to be installed")


__all__ = ["HAS_TRITON", "stream_mix_add", "expm_t18", "expm_t18_triton", "expm_t18_augmented_sparse"]
