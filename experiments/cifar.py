import argparse
from collections import defaultdict
from pathlib import Path

import trackio

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as tvt
from omegaconf import OmegaConf
from timm.layers import Mlp
from torch.utils.data import DataLoader, Subset

from hyperconnections.cghc import ContinuousGenHyperConnections
from hyperconnections.ghc import GeneralizedHyperConnections
from hyperconnections.mhc import ManifoldHyperConnections

HC_LAYERS = {
    "ghc": GeneralizedHyperConnections,
    "mhc": ManifoldHyperConnections,
    "cghc": ContinuousGenHyperConnections,
}


class TokenMixer(nn.Module):
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = Mlp(seq_len, int(seq_len * 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)


class ChannelMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm1(x))


class MixerBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int) -> None:
        super().__init__()
        self.token_mixer = TokenMixer(dim, seq_len)
        self.channel_mixer = ChannelMixer(dim)

    def forward(self, x: torch.Tensor, return_mixing: bool = False):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        if return_mixing:
            return x, None
        return x


class HyperConnectedMixerBlock(nn.Module):
    def __init__(
        self, dim: int, seq_len: int, hc_cls, n: int, m: int, **hc_kwargs
    ) -> None:
        super().__init__()
        input_dim = (n * dim) // m
        self.hc_token_mixer = hc_cls(
            n=n,
            m=m,
            input_dim=input_dim,
            embed_dim=dim,
            module=TokenMixer(dim, seq_len),
            **hc_kwargs,
        )
        self.hc_channel_mixer = hc_cls(
            n=n,
            m=m,
            input_dim=input_dim,
            embed_dim=dim,
            module=ChannelMixer(dim),
            **hc_kwargs,
        )

    @staticmethod
    def _mixing_matrix(hc, x: torch.Tensor) -> torch.Tensor:
        """Stream mixing [B, seq_len, n, n] for a single HC layer."""
        B, seq_len = x.shape[:2]
        x_blocks = x.reshape(-1, hc.n, hc.block_size)
        if hasattr(hc, "compute_transition"):  # CGHC: Phi = exp(dt * A)
            sm = hc.compute_transition(x_blocks)
        else:  # GHC / MHC
            _, _, sm = hc.compute_mixing_weights(x_blocks)
        return sm.reshape(B, seq_len, hc.n, hc.n)

    def forward(self, x: torch.Tensor, return_mixing: bool = False):
        if return_mixing:
            with torch.no_grad():
                token_mixing = self._mixing_matrix(self.hc_token_mixer, x).detach()
        x = self.hc_token_mixer(x)
        if return_mixing:
            with torch.no_grad():
                channel_mixing = self._mixing_matrix(self.hc_channel_mixer, x).detach()
            mixing = {"token": token_mixing, "channel": channel_mixing}
        x = self.hc_channel_mixer(x)
        return (x, mixing) if return_mixing else x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        return x.flatten(2).transpose(1, 2)  # [B, seq_len, embed_dim]


class HCExpand(nn.Module):
    def __init__(self, dim: int, n: int, m: int):
        super().__init__()
        self.dim = dim
        self.n = n
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 1, self.n // self.m)


class HCContract(nn.Module):
    def __init__(self, dim: int, n: int, m: int):
        super().__init__()
        self.dim = dim
        self.n = n
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, (self.n // self.m, self.dim)).mean(dim=-2)


class MLPMixer(nn.Module):
    def __init__(
        self,
        depth: int,
        aspect_ratio: float,
        patch_size: int = 4,
        img_size: int = 32,
        num_classes: int = 10,
        hc_cls=None,
        n: int = 4,
        m: int = 2,
        **hc_kwargs,
    ) -> None:
        super().__init__()
        hidden_dim = int(aspect_ratio * depth)
        seq_len = (img_size // patch_size) ** 2

        model_dim = hidden_dim
        self.patch_embed = PatchEmbed(3, model_dim, patch_size)

        if hc_cls is not None:
            self.blocks = nn.Sequential(*[
                HyperConnectedMixerBlock(hidden_dim, seq_len, hc_cls, n, m, **hc_kwargs)
                for _ in range(depth)
            ])
            self.expand = HCExpand(hidden_dim, n, m)
            self.contract = HCContract(hidden_dim, n, m)
        else:
            self.blocks = nn.Sequential(*[
                MixerBlock(hidden_dim, seq_len) for _ in range(depth)
            ])
            self.expand = nn.Identity()
            self.contract = nn.Identity()

        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, x: torch.Tensor, return_mixing: bool = False):
        x = self.patch_embed(x)
        x = self.expand(x)
        if return_mixing:
            mixing_matrices = []
            for block in self.blocks:
                x, m = block(x, return_mixing=True)
                mixing_matrices.append(m)
        else:
            x = self.blocks(x)
        x = self.contract(x)
        x = self.norm(x).mean(dim=1)
        logits = self.head(x)
        return (logits, mixing_matrices) if return_mixing else logits


CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)


def get_cifar10_loaders(batch_size: int, num_workers: int, data_dir: str = "./data"):
    train_transform = tvt.Compose([
        tvt.RandomHorizontalFlip(),
        tvt.ToImage(),
        tvt.ToDtype(torch.float32, scale=True),
        tvt.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    val_transform = tvt.Compose([
        tvt.ToImage(),
        tvt.ToDtype(torch.float32, scale=True),
        tvt.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_transform
    )
    val_ds = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=val_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, val_ds


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def make_run_dir(args) -> Path:
    name = f"{args.hc_type}_d{args.depth}_ar{args.aspect_ratio}_p{args.patch_size}"
    if args.hc_type != "none":
        name += f"_n{args.n}m{args.m}"
    run_dir = Path(__file__).parent / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(vars(args)), run_dir / "config.yaml")
    return run_dir


def sample_per_class(dataset, n_per_class: int = 10, seed: int = 42) -> list:
    """Return indices of n_per_class samples per class, reproducibly seeded."""
    class_indices = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_indices[label].append(idx)
    gen = torch.Generator()
    gen.manual_seed(seed)
    selected = []
    for cls in sorted(class_indices):
        idxs = class_indices[cls]
        perm = torch.randperm(len(idxs), generator=gen)
        selected.extend(idxs[perm[i].item()] for i in range(min(n_per_class, len(idxs))))
    return selected


def run_mixing_analysis(model, val_ds, device, run_dir: Path, seed: int = 42):
    """Sample 10 images per class and compute per-layer HC mixing matrices."""
    indices = sample_per_class(val_ds, n_per_class=10, seed=seed)
    subset = Subset(val_ds, indices)
    images = torch.stack([subset[i][0] for i in range(len(subset))])
    labels = torch.tensor([subset[i][1] for i in range(len(subset))])

    model.eval()
    with torch.no_grad():
        _, mixing_matrices = model(images.to(device), return_mixing=True)

    # Stack per-layer matrices: [depth, B*seq, n, n]
    token_mixing = torch.stack([m["token"] for m in mixing_matrices])
    channel_mixing = torch.stack([m["channel"] for m in mixing_matrices])

    torch.save(
        {
            "token_mixing": token_mixing,
            "channel_mixing": channel_mixing,
            "labels": labels,
            "indices": indices,
        },
        run_dir / "mixing_analysis.pt",
    )
    print(f"Mixing analysis saved: {token_mixing.shape} (depth, B, seq_len, n, n)")


def main():
    parser = argparse.ArgumentParser(description="MLPMixer CIFAR10 training")

    # Model
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument(
        "--aspect_ratio",
        type=float,
        default=8.0,
        help="hidden_dim = int(aspect_ratio * depth)",
    )
    parser.add_argument("--patch_size", type=int, default=4)

    # HC (shared)
    parser.add_argument(
        "--hc_type", type=str, default="none", choices=["none", "ghc", "mhc", "cghc"]
    )
    parser.add_argument("--n", type=int, default=4, help="number of HC streams")
    parser.add_argument(
        "--m", type=int, default=2, help="HC backbone divisor (must divide n)"
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--elementwise_affine", action="store_true")

    # MHC-specific
    parser.add_argument("--sinkhorn_iters", type=int, default=20)

    # CGHC-specific
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--dt_min", type=float, default=0.001)
    parser.add_argument("--dt_max", type=float, default=1.0)
    parser.add_argument("--learn_dt", action="store_true")
    parser.add_argument(
        "--generator_type",
        type=str,
        default="conservative_psd_diss",
        choices=[
            "conservative",
            "psd_diss",
            "diagonal_diss",
            "laplacian",
            "conservative_diag_diss",
            "conservative_psd_diss",
            "conservative_laplacian",
        ],
    )
    parser.add_argument(
        "--projection", type=str, default="none", choices=["none", "mean", "v"]
    )
    parser.add_argument("--use_triton", action="store_true", default=False)

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42, help="seed for analysis sampling")

    args = parser.parse_args()
    device = torch.device(args.device)

    hc_cls = HC_LAYERS.get(args.hc_type)
    hc_kwargs = dict(bias=args.bias, elementwise_affine=args.elementwise_affine)
    if args.hc_type == "mhc":
        hc_kwargs["sinkhorn_iters"] = args.sinkhorn_iters
    elif args.hc_type == "cghc":
        hc_kwargs.update(
            dt=args.dt,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            learn_dt=args.learn_dt,
            generator_type=args.generator_type,
            projection=args.projection,
            use_triton=args.use_triton,
        )

    model = MLPMixer(
        depth=args.depth,
        aspect_ratio=args.aspect_ratio,
        patch_size=args.patch_size,
        hc_cls=hc_cls,
        n=args.n,
        m=args.m,
        **hc_kwargs,
    ).to(device)

    run_dir = make_run_dir(args)

    hidden_dim = int(args.aspect_ratio * args.depth)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"depth={args.depth}  hidden_dim={hidden_dim}  hc_type={args.hc_type}  params={n_params:,}")
    print(f"run dir: {run_dir}")

    train_loader, val_loader, val_ds = get_cifar10_loaders(
        args.batch_size, args.num_workers, args.data_dir
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    run_name = run_dir.name
    trackio.init(project="mlpmixer-cifar10", name=run_name, config=vars(args))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        trackio.log(
            {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
            step=epoch,
        )
        print(
            f"epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"Weights saved to {run_dir / 'model.pt'}")

    if args.hc_type != "none":
        run_mixing_analysis(model, val_ds, device, run_dir, seed=args.seed)


if __name__ == "__main__":
    main()
