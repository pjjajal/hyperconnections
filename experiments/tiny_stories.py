import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Optional

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.ghc import GeneralizedHyperConnections
from hyperconnections.mhc import ManifoldHyperConnections


TINY_DATASETS_PATH = "karpathy/tinystories-gpt4-clean"


# RoPE Specific
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.

    Args:
        x: Tensor of shape (..., dim)

    Returns:
        Rotated tensor of shape (..., dim)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# TODO: Add frequency scaling for context extension. See:
# http://arxiv.org/abs/2402.13753
# https://huggingface.co/docs/transformers/internal/rope_utils
def precompute_freqs(
    dim: int,
    end: int,
    base_freq: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Precomputes position-dependent frequencies for RoPE.

    Args:
        dim: Embedding dimension (must be even)
        end: Maximum sequence length
        base_freq: Base for frequency bands
        device: Device for tensor

    Returns:
        Frequency tensor of shape (end, dim)
    """
    freqs = 1.0 / (base_freq ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs


def apply_rotary_pos_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Applies rotary position embeddings.

    Args:
        x: Input tensor [B, H, N, D]
        freqs: Precomputed frequencies [max_seq_len, D]

    Returns:
        Output tensor [B, H, N, D]
    """
    seq_len = x.shape[2]
    freqs = freqs[:seq_len].unsqueeze(0).unsqueeze(1)
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension must be divisible by number of heads, but got {embed_dim} and {num_heads}"
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, freqs: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv_proj(x).split(self.embed_dim, dim=-1)  # [B, N, 3*D]

        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, D]
        out = self.out_proj(out)
        return out
    

class Mlp(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = Attention(embed_dim, num_heads)
        self.mlp = Mlp(embed_dim, mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, freqs: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = x + self.norm
        attn_output = self.attn(x, freqs)
        x = x + attn_output
        x = self.norm1(x)

        # MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)

        return x

class HCBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = Attention(embed_dim, num_heads)
        self.mlp = Mlp(embed_dim, mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Hyperconnections
        self.hyperconnections = GeneralizedHyperConnections(
            n=4,
            m=2,
            input_dim=embed_dim * 2,
            embed_dim=embed_dim,
            module=self.attn,
            bias=True,
            elementwise_affine=True,
        )

    def forward(self, x, freqs: torch.Tensor) -> torch.Tensor:
        # Self-attention with hyperconnections
        attn_output = self.hyperconnections(x, freqs)
        x = x + attn_output
        x = self.norm1(x)

        # MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)

        return x



def parse_args():
    pass


def main(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
