"""Quick example of using the synthetic tasks."""

import torch
from hyperconnections import cghc, mhc, ghc, identity_hc
from experiments.synthetic_dynamics import (
    SignalPreservationDataset,
    SignalRotationDataset,
    SignalFilteringDataset,
    StreamDynamicsModel,
    ZeroModule,
    mse_loss,
)


def create_model(hc_class, n_streams=4, d=64, n_layers=8, **hc_kwargs):
    """Helper to create a model with specified hyperconnection class."""
    return StreamDynamicsModel(
        n_streams=n_streams,
        d=d,
        n_layers=n_layers,
        hc_class=hc_class,
        hc_kwargs=hc_kwargs,
        module_class=ZeroModule,
    )


def example_preservation():
    """Example: Signal Preservation task."""
    print("="*60)
    print("Signal Preservation Task")
    print("="*60)

    # Create dataset
    dataset = SignalPreservationDataset(
        n_samples=100,
        n_streams=4,
        d=64,
        n_memories=4,
        seed=42,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Input shape: {dataset[0]['input'].shape}")
    print(f"Target shape: {dataset[0]['target'].shape}")

    # Create model with ZeroModule (pure stream dynamics)
    # Try different models: cghc, mhc, ghc, identity_hc
    model = create_model(
        cghc.ContinuousGenHyperConnections,
        dt=0.1,
        generator_type="conservative_psd_diss",
        projection="none",
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch = torch.stack([dataset[i]["input"] for i in range(4)])
    targets = torch.stack([dataset[i]["target"] for i in range(4)])

    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        loss = mse_loss(outputs, targets)

    print(f"\nInitial MSE loss: {loss.item():.6f}")
    print(f"Output shape: {outputs.shape}")
    print()


def example_rotation():
    """Example: Signal Rotation task."""
    print("="*60)
    print("Signal Rotation Task (Cyclic Permutation)")
    print("="*60)

    # Create dataset with cyclic permutation
    dataset = SignalRotationDataset(
        n_samples=100,
        n_streams=4,
        d=64,
        n_memories=4,
        transform_type="permutation",
        permutation_mode="cyclic",
        seed=42,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Transformation matrix (cyclic permutation):")
    print(dataset.T)

    # Create model
    model = create_model(
        cghc.ContinuousGenHyperConnections,
        dt=0.1,
        generator_type="conservative_psd_diss",
        projection="none",
    )

    # Test forward pass
    batch = torch.stack([dataset[i]["input"] for i in range(4)])
    targets = torch.stack([dataset[i]["target"] for i in range(4)])

    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        loss = mse_loss(outputs, targets)

    print(f"\nInitial MSE loss: {loss.item():.6f}")
    print()


def example_orthogonal_rotation():
    """Example: Signal Rotation with orthogonal matrix."""
    print("="*60)
    print("Signal Rotation Task (Orthogonal Rotation)")
    print("="*60)

    # Create dataset with random orthogonal rotation
    dataset = SignalRotationDataset(
        n_samples=100,
        n_streams=4,
        d=64,
        n_memories=4,
        transform_type="rotation",
        seed=42,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Transformation matrix (first 3x3 block):")
    print(dataset.T[:3, :3])
    print(f"Is orthogonal: {torch.allclose(dataset.T @ dataset.T.T, torch.eye(4), atol=1e-5)}")

    # Test forward pass with different model types
    print("Testing with CGHC:")
    model = create_model(
        cghc.ContinuousGenHyperConnections,
        dt=0.1,
        generator_type="conservative_psd_diss",
        projection="none",
    )

    batch = torch.stack([dataset[i]["input"] for i in range(4)])
    targets = torch.stack([dataset[i]["target"] for i in range(4)])

    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        loss = mse_loss(outputs, targets)

    print(f"\nInitial MSE loss: {loss.item():.6f}")
    print()


def compare_models():
    """Compare different hyperconnection models on the same task."""
    print("="*60)
    print("Model Comparison on Signal Preservation")
    print("="*60)

    # Create dataset
    dataset = SignalPreservationDataset(
        n_samples=100,
        n_streams=4,
        d=64,
        seed=42,
    )

    batch = torch.stack([dataset[i]["input"] for i in range(4)])
    targets = torch.stack([dataset[i]["target"] for i in range(4)])

    models = {
        "CGHC": create_model(
            cghc.ContinuousGenHyperConnections,
            dt=0.1,
            generator_type="conservative_psd_diss",
        ),
        "MHC": create_model(mhc.ManifoldHyperConnections, sinkhorn_iters=20),
        "GHC": create_model(ghc.GeneralizedHyperConnections),
        "Identity HC": create_model(identity_hc.IdentityHyperConnections),
    }

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
            loss = mse_loss(outputs, targets)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:15s}: Loss={loss.item():.6f}, Params={n_params:,}")
    print()


def example_filtering():
    """Example: Signal Filtering task."""
    print("="*60)
    print("Signal Filtering Task")
    print("="*60)

    # Create dataset with orthogonal signal/noise subspaces
    dataset = SignalFilteringDataset(
        n_samples=100,
        n_streams=4,
        d=64,
        n_layers=8,
        n_signal_memories=2,  # Signal lives in 2D subspace
        n_noise_memories=2,   # Noise lives in orthogonal 2D subspace
        noise_scale=0.5,
        seed=42,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Signal memories: {dataset.n_signal_memories}")
    print(f"Noise memories: {dataset.n_noise_memories}")
    print(f"Noise scale: {dataset.noise_scale}")
    print(f"Noise injected at {dataset.n_layers} layers")

    # Verify subspace structure
    sample_idx = 0
    signal_basis = dataset.signal_basis[sample_idx]  # [n_signal_dim, n_streams]
    noise_basis = dataset.noise_basis[sample_idx]    # [n_noise_dim, n_streams]
    signal_keys = dataset.signal_keys[sample_idx]    # [n_signal_memories, n_streams]
    noise_keys = dataset.noise_keys[sample_idx]      # [n_noise_memories, n_streams]

    # Check if signal basis is orthonormal
    gram_signal = signal_basis @ signal_basis.T
    is_signal_orthonormal = torch.allclose(gram_signal, torch.eye(dataset.n_signal_memories), atol=1e-5)
    print(f"Signal basis orthonormal: {is_signal_orthonormal}")

    # Check if noise basis is orthonormal
    gram_noise = noise_basis @ noise_basis.T
    is_noise_orthonormal = torch.allclose(gram_noise, torch.eye(dataset.n_signal_memories), atol=1e-5)
    print(f"Noise basis orthonormal: {is_noise_orthonormal}")

    # Check if signal and noise bases are orthogonal
    cross_gram = signal_basis @ noise_basis.T
    is_orthogonal = torch.allclose(cross_gram, torch.zeros_like(cross_gram), atol=1e-5)
    print(f"Signal and noise bases orthogonal: {is_orthogonal}")

    # Verify signal keys live in signal subspace
    # Project signal keys onto signal basis
    signal_proj = torch.einsum('kn,mn->km', signal_keys, signal_basis)
    signal_reconstructed = torch.einsum('km,mn->kn', signal_proj, signal_basis)
    signal_in_subspace = torch.allclose(signal_keys, signal_reconstructed, atol=1e-5)
    print(f"Signal keys in signal subspace: {signal_in_subspace}")

    # Verify noise keys live in noise subspace
    noise_proj = torch.einsum('kn,mn->km', noise_keys, noise_basis)
    noise_reconstructed = torch.einsum('km,mn->kn', noise_proj, noise_basis)
    noise_in_subspace = torch.allclose(noise_keys, noise_reconstructed, atol=1e-5)
    print(f"Noise keys in noise subspace: {noise_in_subspace}")

    # Create model
    model = create_model(
        cghc.ContinuousGenHyperConnections,
        n_streams=4,
        d=64,
        n_layers=8,
        dt=0.1,
        generator_type="conservative_psd_diss",
    )

    # Test with noise
    batch = torch.stack([dataset[i]["input"] for i in range(4)])
    targets = torch.stack([dataset[i]["target"] for i in range(4)])
    noise = torch.stack([dataset[i]["noise"] for i in range(4)])

    print(f"\nNoise shape: {noise.shape}")  # [B, n_layers, n_streams, d]

    model.eval()
    with torch.no_grad():
        outputs = model(batch, noise=noise)
        loss = mse_loss(outputs, targets)

    print(f"\nInitial MSE loss (with noise): {loss.item():.6f}")
    print()


if __name__ == "__main__":
    example_preservation()
    example_rotation()
    example_orthogonal_rotation()
    example_filtering()
    compare_models()
