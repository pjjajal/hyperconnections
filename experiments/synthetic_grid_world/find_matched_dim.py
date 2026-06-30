"""Find the baseline num_layers whose parameter count matches an IHC model.

Keeps dim fixed and searches over depth, so thinking capacity per layer
stays identical — only the number of processing steps differs.

Prints the matched num_layers (int) to stdout; diagnostic info goes to stderr.

Usage:
    uv run --extra experiments python -m experiments.synthetic_grid_world.find_matched_dim \
        --config experiments/synthetic_grid_world/configs/sweep/h4_veryhard_ihc.yaml \
        --n 8 --dim 128 --num-layers 12
"""

import argparse
import sys
from omegaconf import OmegaConf

from experiments.synthetic_grid_world.train import build_hc_cls, N_ACTIONS
from experiments.synthetic_grid_world.model import Transformer


def count_params(cfg, dim, num_layers, hc_cls=None, input_dim=None):
    n_positions = cfg.data.n_rows * cfg.data.n_cols
    model = Transformer(
        n_grid_tokens=n_positions,
        n_observations=cfg.data.n_colours,
        n_actions=N_ACTIONS,
        n_positions=n_positions,
        seq_len=cfg.data.trajectory_length + 1,
        dim=dim,
        input_dim=input_dim or dim,
        num_heads=max(1, dim // 16),
        ffn_ratio=cfg.model.ffn_ratio,
        num_layers=num_layers,
        hc_cls=hc_cls,
        qkv_bias=cfg.model.qkv_bias,
        proj_bias=cfg.model.proj_bias,
    )
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--n", type=int, required=True, help="Number of IHC streams")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--dim", type=int, required=True, help="Backbone dim (fixed for both models)")
    parser.add_argument("--num-layers", type=int, default=12, help="IHC depth")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    hc_cfg = OmegaConf.create({
        "type": "identity_hc", "n": args.n, "m": args.m,
        "bias": False, "elementwise_affine": False,
    })
    hc_cls, input_dim = build_hc_cls(hc_cfg, args.dim)
    target = count_params(cfg, args.dim, args.num_layers, hc_cls, input_dim)

    # Binary search over num_layers (integer), dim stays fixed
    lo, hi = args.num_layers, args.num_layers * (args.n * args.n)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_params(cfg, args.dim, mid) < target:
            lo = mid + 1
        else:
            hi = mid

    matched_layers = lo
    matched_params = count_params(cfg, args.dim, matched_layers)

    print(
        f"IHC n={args.n} m={args.m} dim={args.dim} layers={args.num_layers}: {target:,} params  →  "
        f"baseline dim={args.dim} layers={matched_layers}: {matched_params:,} params",
        file=sys.stderr,
    )
    print(matched_layers)


if __name__ == "__main__":
    main()
