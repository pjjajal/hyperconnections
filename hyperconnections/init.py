import torch
from timm.layers import trunc_normal_ as _timm_trunc_normal_


@torch.no_grad()
def trunc_normal_(tensor: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    """timm's ``trunc_normal_``, sampled in float32 and copied into ``tensor``.

    Sampling directly in bf16 rounds some uniform draws to +/-1, whose ``erfinv`` is
    infinite, so those entries land on timm's +/-2 clamp bounds instead of ~N(0, std).
    """
    sample = torch.empty(tensor.shape, dtype=torch.float32, device=tensor.device)
    _timm_trunc_normal_(sample, std=std)
    return tensor.copy_(sample)
