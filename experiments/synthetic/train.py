"""Training script for synthetic tasks."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hyperconnections import cghc, mhc, ghc, identity_hc
from experiments.synthetic import (
    SignalPreservationDataset,
    SignalRotationDataset,
    SignalFilteringDataset,
    StreamDynamicsModel,
    ZeroModule,
    mse_loss,
)
from einops import einsum
from experiments.synthetic.logger import ExperimentLogger


# Mapping from model name to class
HC_MODELS = {
    "cghc": cghc.ContinuousGenHyperConnections,
    "mhc": mhc.ManifoldHyperConnections,
    "ghc": ghc.GeneralizedHyperConnections,
    "identity_hc": identity_hc.IdentityHyperConnections,
}


def train_epoch(model, dataloader, optimizer, device, logger=None, epoch=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    step = 0

    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        noise = batch.get("noise", None)
        if noise is not None:
            noise = noise.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, noise=noise)
        loss = mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log per-batch metrics
        if logger is not None:
            metrics = {"loss": loss.item()}

            # Log energy statistics for filtering task
            if noise is not None:
                B = inputs.shape[0]
                signal_basis = batch.get("signal_basis", None)
                noise_basis = batch.get("noise_basis", None)

                # Input SNR: ||signal|| / ||total_noise||
                signal_norm = torch.norm(inputs.reshape(B, -1), dim=1).pow(2).mean().item()

                # Sum noise across all layers then compute norm
                # noise shape: [B, n_layers, n_streams, d]
                total_noise = noise.sum(dim=1)  # [B, n_streams, d]
                total_noise_norm = torch.norm(total_noise.reshape(B, -1), dim=1).pow(2).mean().item()

                input_snr = signal_norm / (total_noise_norm + 1e-8)

                # Output SNR: ||targets|| / ||outputs - targets||
                output_signal_norm = torch.norm(targets.reshape(B, -1), dim=1).pow(2).mean().item()
                output_error_norm = torch.norm((outputs - targets).reshape(B, -1), dim=1).pow(2).mean().item()
                output_snr = output_signal_norm / (output_error_norm + 1e-8)

                # Queried Signal Norm:
                vs = einsum(inputs, signal_basis, "bnd,bkn->bkd")
                outputs_vs = einsum(outputs, signal_basis, "bnd,bkn->bkd")

                vs_energy = vs.norm(dim=-1).pow(2).mean().item()
                outputs_vs_energy = outputs_vs.norm(dim=-1).pow(2).mean().item()
                residual_energy = (vs - outputs_vs).norm(dim=-1).pow(2).mean().item()

                snr_queried = vs_energy / (residual_energy + 1e-8)


                metrics.update({
                    "signal_norm": signal_norm,
                    "total_noise_norm": total_noise_norm,
                    "input_snr": input_snr,
                    "output_signal_norm": output_signal_norm,
                    "output_error_norm": output_error_norm,
                    "output_snr": output_snr,
                    "vs_energy_queried": vs_energy,
                    "outputs_vs_energy_queried": outputs_vs_energy,
                    "residual_energy_queried": residual_energy,
                    "snr_queried": snr_queried,
                })

            logger.log_metrics(epoch, step, metrics)
        step += 1

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        noise = batch.get("noise", None)
        if noise is not None:
            noise = noise.to(device)

        outputs = model(inputs, noise=noise)
        loss = mse_loss(outputs, targets)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train synthetic tasks")

    # Task settings
    parser.add_argument("--task", type=str, default="preservation",
                       choices=["preservation", "rotation", "filtering"],
                       help="Task type")
    parser.add_argument("--transform-type", type=str, default="permutation",
                       choices=["permutation", "rotation"],
                       help="Transformation type for rotation task")
    parser.add_argument("--permutation-mode", type=str, default="cyclic",
                       choices=["cyclic", "random", "reverse"],
                       help="Permutation mode")

    # Data settings
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--n-streams", type=int, default=4,
                       help="Number of streams (n)")
    parser.add_argument("--d", type=int, default=64,
                       help="Dimension of value vectors")
    parser.add_argument("--n-memories", type=int, default=None,
                       help="Number of key-value pairs (default: n_streams)")

    # Filtering task settings
    parser.add_argument("--n-signal-basis", type=int, default=2,
                       help="Number of signal basis vectors")
    parser.add_argument("--n-signal-memories", type=int, default=2,
                       help="Number of signal key-value pairs")
    parser.add_argument("--n-noise-basis", type=int, default=2,
                       help="Number of noise basis vectors")
    parser.add_argument("--n-noise-memories", type=int, default=2,
                       help="Number of noise key-value pairs per layer")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                       help="Noise scale for filtering task")

    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Model settings
    parser.add_argument("--model", type=str, default="cghc",
                       choices=["cghc", "mhc", "ghc", "identity_hc"],
                       help="Hyperconnection model type")
    parser.add_argument("--n-layers", type=int, default=1,
                       help="Number of hyperconnection layers")

    # CGHC-specific settings
    parser.add_argument("--generator-type", type=str, default="conservative_diag_diss",
                       help="Generator type for CGHC")
    parser.add_argument("--dt", type=float, default=0.01,
                       help="Time step for CGHC")
    parser.add_argument("--projection", type=str, default="none",
                       choices=["none", "mean", "v"],
                       help="Projection type for CGHC")

    # MHC-specific settings
    parser.add_argument("--sinkhorn-iters", type=int, default=20,
                       help="Sinkhorn iterations for MHC")

    # Training settings
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device")

    # Logging settings
    parser.add_argument("--log-dir", type=str, default="experiments/synthetic/runs",
                       help="Directory for experiment logs")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Run name (default: timestamp)")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Initialize logger
    logger = ExperimentLogger(args.log_dir, args.run_name)

    # Save configuration
    config = vars(args)
    logger.save_config(config)

    # Validate and set device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")
    if args.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif args.device == "mps":
        print("  Apple Silicon GPU (Metal)")
    print()

    # Create dataset
    if args.task == "preservation":
        dataset = SignalPreservationDataset(
            n_samples=args.n_samples,
            n_streams=args.n_streams,
            d=args.d,
            n_memories=args.n_memories,
            seed=args.seed,
        )
    elif args.task == "rotation":
        dataset = SignalRotationDataset(
            n_samples=args.n_samples,
            n_streams=args.n_streams,
            d=args.d,
            n_memories=args.n_memories,
            transform_type=args.transform_type,
            permutation_mode=args.permutation_mode,
            seed=args.seed,
        )
    elif args.task == "filtering":
        dataset = SignalFilteringDataset(
            n_samples=args.n_samples,
            n_streams=args.n_streams,
            d=args.d,
            n_layers=args.n_layers,
            n_signal_basis=args.n_signal_basis,
            n_signal_memories=args.n_signal_memories,
            n_noise_basis=args.n_noise_basis,
            n_noise_memories=args.n_noise_memories,
            noise_scale=args.noise_scale,
            seed=args.seed,
        )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )

    # Create model
    hc_class = HC_MODELS[args.model]

    # Build model-specific kwargs
    hc_kwargs = {}
    if args.model == "cghc":
        hc_kwargs = {
            "dt": args.dt,
            "generator_type": args.generator_type,
            "projection": args.projection,
            "use_triton": False,
            "vec_dt": True,
            "elementwise_affine": True,
            "learn_dt": True,
            "dt_min": 0.0001,
            "dt_max": 1.0,
        }
    elif args.model == "mhc":
        hc_kwargs = {
            "sinkhorn_iters": args.sinkhorn_iters,
        }
    elif args.model == "ghc":
        hc_kwargs = {}
    elif args.model == "identity_hc":
        hc_kwargs = {}

    model = StreamDynamicsModel(
        n_streams=args.n_streams,
        d=args.d,
        n_layers=args.n_layers,
        hc_class=hc_class,
        hc_kwargs=hc_kwargs,
        module_class=ZeroModule,
    ).to(args.device)

    model.compile()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, dataloader, optimizer, args.device, logger, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.6f}")

        if train_loss < best_loss:
            best_loss = train_loss
            print(f"  New best loss: {best_loss:.6f}")

    print(f"\nFinal loss: {best_loss:.6f}")

    # Save final metrics
    logger.save_metrics()
    print(f"\nExperiment logged to: {logger.run_dir}")


if __name__ == "__main__":
    main()
