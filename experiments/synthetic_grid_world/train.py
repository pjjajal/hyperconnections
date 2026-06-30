import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.synthetic_grid_world.grid_gen import GridWorldDataset
from experiments.synthetic_grid_world.logger import ExperimentLogger
from experiments.synthetic_grid_world.model import Transformer
from hyperconnections import cghc, ghc, identity_hc, mhc

torch.set_float32_matmul_precision('high')

# Actions 0-3 are NSEW; 4 is the null token appended to the final observation.
NULL_ACTION = 4
N_ACTIONS = 5

HC_CLASSES = {
    "ghc": ghc.GeneralizedHyperConnections,
    "mhc": mhc.ManifoldHyperConnections,
    "cghc": cghc.ContinuousGenHyperConnections,
    "identity_hc": identity_hc.IdentityHyperConnections,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train grid-world localization model")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g., training.lr=1e-4)",
    )
    return parser.parse_args()


def load_config(args):
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return cfg


def build_hc_cls(hc_cfg, dim: int):
    """Return (hc_cls, input_dim). hc_cls is None for the baseline."""
    if hc_cfg.type is None:
        return None, dim
    hc_class = HC_CLASSES[hc_cfg.type]
    n, m = hc_cfg.n, hc_cfg.m
    input_dim = (n // m) * dim
    extra = {k: v for k, v in hc_cfg.items() if k not in ("type", "n", "m")}
    return partial(hc_class, n=n, m=m, input_dim=input_dim, embed_dim=dim, **extra), input_dim


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        obs = batch.trajectory.observations.to(device)
        actions = batch.trajectory.actions.to(device)
        final_pos = batch.trajectory.final_position.to(device)

        B = obs.shape[0]
        null = torch.full((B, 1), NULL_ACTION, dtype=torch.long, device=device)
        actions_padded = torch.cat([actions, null], dim=1)

        optimizer.zero_grad()
        logits = model(obs, actions_padded, batch.grid.to(device))
        loss = F.cross_entropy(logits, final_pos)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == final_pos).sum().item()
        total += B

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        obs = batch.trajectory.observations.to(device)
        actions = batch.trajectory.actions.to(device)
        final_pos = batch.trajectory.final_position.to(device)

        B = obs.shape[0]
        null = torch.full((B, 1), NULL_ACTION, dtype=torch.long, device=device)
        actions_padded = torch.cat([actions, null], dim=1)

        logits = model(obs, actions_padded, batch.grid.to(device))
        loss = F.cross_entropy(logits, final_pos)

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == final_pos).sum().item()
        total += B

    return total_loss / len(dataloader), correct / total


def main():
    args = parse_args()
    cfg = load_config(args)

    torch.manual_seed(cfg.data.seed)

    device = resolve_device(cfg.training.device)
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────
    dataset_kwargs = dict(
        n_rows=cfg.data.n_rows,
        n_cols=cfg.data.n_cols,
        n_colours=cfg.data.n_colours,
        trajectory_length=cfg.data.trajectory_length,
        ambiguous_threshold=cfg.data.ambiguous_threshold,
        final_unique=cfg.data.final_unique,
    )
    train_cache = cfg.data.get("train_cache_path") or None
    val_cache = cfg.data.get("val_cache_path") or None
    train_dataset = GridWorldDataset(
        n_samples=cfg.data.n_samples, seed=cfg.data.seed,
        cache_path=train_cache, **dataset_kwargs
    )
    val_dataset = GridWorldDataset(
        n_samples=cfg.data.n_samples // 5, seed=cfg.data.seed + 1,
        cache_path=val_cache, **dataset_kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=4,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    n_positions = cfg.data.n_rows * cfg.data.n_cols
    n_grid_tokens = cfg.data.n_rows * cfg.data.n_cols
    seq_len = cfg.data.trajectory_length + 1

    hc_cls, input_dim = build_hc_cls(cfg.model.hc, cfg.model.dim)

    dim = cfg.model.dim
    num_heads = cfg.model.get("num_heads") or max(1, dim // 16)
    num_layers = cfg.model.get("num_layers") or max(2, dim // 10)

    model = Transformer(
        n_grid_tokens=n_grid_tokens,
        n_observations=cfg.data.n_colours,
        n_actions=N_ACTIONS,
        n_positions=n_positions,
        seq_len=seq_len,
        dim=dim,
        input_dim=input_dim,
        num_heads=num_heads,
        ffn_ratio=cfg.model.ffn_ratio,
        num_layers=num_layers,
        hc_cls=hc_cls,
        qkv_bias=cfg.model.qkv_bias,
        proj_bias=cfg.model.proj_bias,
    ).to(device)
    model.compile()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # ── Learning Rate Scheduler ───────────────────────────────────────────
    warmup_steps = 1000
    total_steps = cfg.training.epochs * len(train_loader)

    # Linear warmup from 0 to base_lr over warmup_steps
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-10,  # Start near 0
        end_factor=1.0,      # End at base_lr
        total_iters=warmup_steps
    )

    # Cosine annealing from base_lr to 0.1 * base_lr
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0.1 * cfg.training.lr
    )

    # Chain warmup and cosine annealing
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # ── Logging ───────────────────────────────────────────────────────────
    logger = ExperimentLogger(cfg.logging.log_dir, cfg.logging.run_name)
    logger.save_config(OmegaConf.to_container(cfg, resolve=True))

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    for epoch in range(cfg.training.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]['lr']
        logger.log_metrics(epoch, -1, {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        print(
            f"Epoch {epoch + 1:3d}/{cfg.training.epochs}"
            f"  train loss {train_loss:.4f}  acc {train_acc:.3f}"
            f"  |  val loss {val_loss:.4f}  acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), logger.run_dir / "best_model.pt")

    logger.save_metrics()
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Logs: {logger.run_dir}")


if __name__ == "__main__":
    main()
