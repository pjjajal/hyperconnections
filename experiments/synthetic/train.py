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
                # Signal energy (input)
                signal_energy = torch.norm(inputs).item()

                # Noise energy: average per layer and total
                # noise shape: [B, n_layers, n_streams, d]
                noise_per_layer = torch.stack([
                    torch.norm(noise[:, i]) for i in range(noise.shape[1])
                ])
                avg_noise_per_layer = noise_per_layer.mean().item()
                total_noise_added = noise_per_layer.sum().item()

                # Input SNR
                input_snr = signal_energy / (total_noise_added + 1e-8)

                # Output energy and error
                output_energy = torch.norm(outputs).item()
                error_energy = torch.norm(outputs - targets).item()

                # Option 3: Relative signal strength metrics
                signal_preservation = signal_energy / (output_energy + 1e-8)
                relative_error = error_energy / (signal_energy + 1e-8)
                output_snr_v3 = signal_preservation / (relative_error + 1e-8)

                metrics.update({
                    "signal_energy": signal_energy,
                    "avg_noise_per_layer": avg_noise_per_layer,
                    "total_noise_added": total_noise_added,
                    "output_energy": output_energy,
                    "error_energy": error_energy,
                    "input_snr": input_snr,
                    "signal_preservation": signal_preservation,
                    "relative_error": relative_error,
                    "output_snr_v3": output_snr_v3,
                })

                # Option 2: Subspace projection (if bases available)
                if "signal_basis" in batch and "noise_basis" in batch:
                    signal_basis = batch["signal_basis"].to(device)  # [B, n_signal_basis, n_streams]
                    noise_basis = batch["noise_basis"].to(device)    # [B, n_noise_basis, n_streams]

                    # Project output onto signal subspace
                    # outputs: [B, n_streams, d], signal_basis: [B, n_basis, n_streams]
                    # Project: for each sample and each d dimension
                    B = outputs.shape[0]
                    signal_components = []
                    noise_components = []

                    for b in range(B):
                        # outputs[b]: [n_streams, d], basis: [n_basis, n_streams]
                        # Project each column of outputs[b] onto the basis
                        out_b = outputs[b]  # [n_streams, d]

                        # Project onto signal basis
                        sig_basis = signal_basis[b]  # [n_signal_basis, n_streams]
                        # Projection: basis^T @ (basis @ basis^T)^{-1} @ basis @ out
                        # Simpler: basis^T @ out gives coefficients if basis is orthonormal
                        sig_coeffs = torch.matmul(sig_basis, out_b)  # [n_signal_basis, d]
                        sig_proj = torch.matmul(sig_basis.T, sig_coeffs)  # [n_streams, d]
                        signal_components.append(torch.norm(sig_proj).item())

                        # Project onto noise basis
                        noise_b_basis = noise_basis[b]  # [n_noise_basis, n_streams]
                        noise_coeffs = torch.matmul(noise_b_basis, out_b)  # [n_noise_basis, d]
                        noise_proj = torch.matmul(noise_b_basis.T, noise_coeffs)  # [n_streams, d]
                        noise_components.append(torch.norm(noise_proj).item())

                    signal_subspace_energy = sum(signal_components) / B
                    noise_subspace_energy = sum(noise_components) / B
                    output_snr_v2 = signal_subspace_energy / (noise_subspace_energy + 1e-8)

                    metrics.update({
                        "signal_subspace_energy": signal_subspace_energy,
                        "noise_subspace_energy": noise_subspace_energy,
                        "output_snr_v2": output_snr_v2,
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
