import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Literal
import math


class MuPConfig:
    """Configuration for Maximal Update Parametrization (muP)

    muP enables hyperparameter transfer across model widths by:
    1. Scaling weight initialization appropriately
    2. Scaling learning rates per parameter type
    3. Scaling output logits

    Args:
        enabled: Whether muP is active
        base_dim: Reference width where hyperparameters are tuned
        target_dim: Current model width
    """
    def __init__(self, enabled: bool = False, base_dim: int = 64, target_dim: int = None):
        self.enabled = enabled
        self.base_dim = base_dim
        self.target_dim = target_dim or base_dim
        self.width_mult = self.target_dim / self.base_dim if enabled else 1.0

    def scale_init_std(self, std: float, param_type: Literal['hidden', 'output', 'embedding']) -> float:
        """Scale initialization standard deviation based on muP rules

        Args:
            std: Base standard deviation
            param_type: Type of parameter (hidden/output/embedding)

        Returns:
            Scaled standard deviation
        """
        if not self.enabled:
            return std
        if param_type == 'hidden':
            # Hidden-to-hidden: use N(0, 1/fan_in) - no additional scaling needed
            return std
        elif param_type == 'output':
            # Output: additional 1/width scaling
            return std / self.width_mult
        elif param_type == 'embedding':
            # Embeddings: scale by 1/√width
            return std / math.sqrt(self.width_mult)
        return std

    def get_lr_mult(self, param_type: Literal['hidden', 'output', 'embedding', 'hc_special']) -> float:
        """Get learning rate multiplier for parameter type

        Args:
            param_type: Type of parameter

        Returns:
            Learning rate multiplier to apply to base LR
        """
        if not self.enabled:
            return 1.0
        if param_type in ['hidden', 'output']:
            # Hidden and output layers: scale by 1/width
            return 1.0 / self.width_mult
        else:  # embedding, hc_special
            # Embeddings and HC special params: no scaling
            return 1.0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        max_seq_len=512,
        qkv_bias=False,
        proj_bias=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.head_dim = dim // num_heads
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x):
        B, N, C = x.shape
        head_dim = self.head_dim

        qkv = self.qkv(x).reshape(B, N, self.num_heads, 3 * head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self.q_norm(q).to(dtype=v.dtype)
        k = self.k_norm(k).to(dtype=v.dtype)

        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)  # [B, H, N, D]
        v = v.transpose(1, 2)  # [B, H, N, D]

        cos, sin = self.rotary_emb(N)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, D]
        out = self.proj(out)
        return out


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, bias=True, drop=0.0):
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: int,
        max_seq_len: int = 512,
        qkv_bias: bool = False,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = Mlp(
            dim=dim,
            mlp_ratio=ffn_ratio,
            bias=proj_bias,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HCBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: int,
        hc_cls: nn.Module,
        max_seq_len: int = 512,
        qkv_bias: bool = False,
        proj_bias: bool = True,
    ):
        super().__init__()
        attn = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(
                dim=dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
            ),
        )
        self.attn = hc_cls(module=attn)
        ffn = nn.Sequential(
            nn.LayerNorm(dim),
            Mlp(
                dim=dim,
                mlp_ratio=ffn_ratio,
                bias=proj_bias,
            ),
        )
        self.ffn = hc_cls(module=ffn)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_grid_tokens: int,
        n_observations: int,
        n_actions: int,
        n_positions: int,
        seq_len: int,
        dim: int,
        num_heads: int,
        ffn_ratio: int,
        num_layers: int,
        hc_cls: nn.Module = None,
        input_dim: int | None = None,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        mup_config: MuPConfig | None = None,
    ):
        super().__init__()
        # input_dim is (n/m)*dim for HC; equals dim for standard transformer.
        input_dim = input_dim or dim
        self.mup_config = mup_config or MuPConfig(enabled=False)

        self.observation_embed = nn.Embedding(n_observations, dim)
        self.action_embed = nn.Embedding(n_actions, dim)
        self.grid_pos_embed = nn.Embedding(n_grid_tokens, dim)
        self.stream_proj = (
            nn.Linear(dim, input_dim, bias=False) if input_dim != dim else nn.Identity()
        )

        block_cls = partial(HCBlock, hc_cls=hc_cls) if hc_cls is not None else Block
        self.layers = nn.ModuleList(
            [
                block_cls(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    max_seq_len=n_grid_tokens + seq_len,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.head = nn.Linear(input_dim, n_positions)

        # Apply muP initialization if enabled
        if self.mup_config.enabled:
            self._apply_mup_init()

    def _apply_mup_init(self):
        """Apply Maximal Update Parametrization (muP) initialization

        muP uses different initialization scales for different parameter types:
        - Embeddings: scale by 1/√width_mult
        - Hidden layers: N(0, 1/fan_in) instead of N(0, 1/√fan_in)
        - Output layer: additional 1/width_mult scaling
        - HC parameters: preserved (already initialized by HC modules)
        """
        # Scale embeddings by 1/√width_mult
        for emb_name, emb in [
            ('observation_embed', self.observation_embed),
            ('action_embed', self.action_embed),
            ('grid_pos_embed', self.grid_pos_embed)
        ]:
            if isinstance(emb, nn.Embedding):
                std = emb.weight.std().item()
                new_std = self.mup_config.scale_init_std(std, 'embedding')
                with torch.no_grad():
                    emb.weight.mul_(new_std / std)

        # Reinitialize Linear layers with muP scaling
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Skip HC-specific parameters (they have custom initialization)
                # HC projections (proj_read_in, proj_write_out, etc.) should get muP init
                is_hc_static = any(x in name for x in ['read_in', 'write_out', 'stream_mixing', 'alpha_']) and 'proj_' not in name
                if is_hc_static:
                    continue

                # Determine parameter type based on module name
                if 'head' in name:
                    param_type = 'output'
                else:
                    param_type = 'hidden'

                # muP uses 1/fan_in variance instead of 1/√fan_in
                fan_in = module.weight.shape[1]
                std = 1.0 / fan_in
                std = self.mup_config.scale_init_std(std, param_type)

                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations, actions, grid):
        B = grid.shape[0]
        grid_flat = grid.view(B, -1)                                    # [B, n_grid_tokens]
        pos_ids = torch.arange(grid_flat.shape[1], device=grid.device)
        grid_emb = self.stream_proj(
            self.observation_embed(grid_flat) + self.grid_pos_embed(pos_ids)
        )                                                               # [B, n_grid_tokens, input_dim]

        obs_emb = self.observation_embed(observations)
        act_emb = self.action_embed(actions)
        x = self.stream_proj(obs_emb + act_emb)                         # [B, T+1, input_dim]

        x = torch.cat([grid_emb, x], dim=1)                             # [B, n_grid_tokens + T+1, input_dim]

        for layer in self.layers:
            x = layer(x)
        logits = self.head(x[:, -1])

        # Apply muP output scaling
        if self.mup_config.enabled:
            logits = logits / self.mup_config.width_mult

        return logits
