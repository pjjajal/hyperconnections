import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def load_mixing_analysis_data(path: Path):
    """Load the mixing analysis data from the given path."""
    data = torch.load(path, map_location="cpu")
    token_mixing = data["token_mixing"]  # List of (depth, B, seq_len, n, n)matrices
    channel_mixing = data["channel_mixing"]  # List of (depth, B, seq_len, n, n)matrices
    stacked_mixing = torch.stack([token_mixing, channel_mixing], dim=1) # (depth, 2, B, seq_len, n, n)
    stacked_mixing = stacked_mixing.flatten(0, 1)  # (depth*2, B, seq_len, n, n) # TODO: we should double check that this results in [token_layer1, channel_layer1, token_layer2, channel_layer2, ...]
    return {
        "mixing": stacked_mixing,
        "labels": data["labels"],
        "indices": list(range(stacked_mixing.shape[0])),
    }

def get_eigenvalues(mixing_matrices):
    # mixing matrices = [depth*2, B, seq_len, n, n]
    eigvals = torch.linalg.eigvals(mixing_matrices.contiguous())  # (depth*2, B, seq_len, n)
    eigvals = eigvals.flatten(1) # (depth*2, B*seq_len*n)
    return eigvals

def get_transition_table(mixing_matrices):
    L = len(mixing_matrices)
    n = mixing_matrices.shape[-1]
    Pi = torch.zeros((L, L, n, n))
    identity = torch.eye(n)
    for tau in range(L):
        Pi[tau, tau] = identity
        cur = identity.clone()
        for t in range(tau, L):
            cur = mixing_matrices[t] @ cur
            Pi[tau, t] = cur
    return Pi


def plot_eigenvalues_polar(mixing_matrices, ax=None, title="Eigenvalue Distribution"):
    """Cartesian scatter of eigenvalues in the complex plane, coloured by layer depth."""
    eigvals = get_eigenvalues(mixing_matrices.float())  # (L, B*seq_len*n) complex
    L = eigvals.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    cmap = plt.get_cmap("viridis")
    for layer_idx in range(L):
        ev = eigvals[layer_idx]
        ax.scatter(ev.real.numpy(), ev.imag.numpy(), s=2, alpha=0.3, color=cmap(layer_idx / L))

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "r--", linewidth=1.2, label="unit circle")
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, L))
    plt.colorbar(sm, ax=ax, label="layer")
    return ax


def plot_transition_norms(mixing_matrices, ax=None, title="Transition Matrix Norms"):
    """Heatmap of ||Pi[tau, t]|| (Frobenius) for tau <= t."""
    mean_mixing = mixing_matrices.float().mean(dim=(1, 2))  # (L, n, n)
    Pi = get_transition_table(mean_mixing)          # (L, L, n, n)
    norms = torch.linalg.matrix_norm(Pi.contiguous(), ord=2)

    # mask lower triangle (tau > t, undefined)
    L = norms.shape[0]
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)
    norms_np = norms.numpy().astype(float)
    norms_np[mask.numpy()] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    valid = norms_np[~np.isnan(norms_np) & (norms_np > 0)]
    use_log = valid.size > 0 and (valid.max() / valid.min()) > 100
    norm = plt.matplotlib.colors.LogNorm(vmin=valid.min(), vmax=valid.max()) if use_log else None
    im = ax.imshow(norms_np, origin="lower", aspect="auto", cmap="viridis", norm=norm)
    cb = plt.colorbar(im, ax=ax, label="Spectral Norm (log)" if use_log else "Spectral Norm")
    if not use_log:
        cb.formatter = plt.matplotlib.ticker.FormatStrFormatter("%.2f")
        cb.update_ticks()
    ax.set_xlabel("t  (end layer)")
    ax.set_ylabel("τ  (start layer)")
    ax.set_title(title)
    return ax


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./experiments/runs")
    parser.add_argument("--output_dir", type=str, default="./cifar_plots")
    parser.add_argument("--init_data", action="store_true", help="Whether to generate plots from data at init.")
    return parser.parse_args()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    filename = "mixing_analysis_init.pt" if args.init_data else "mixing_analysis.pt"
    data_dir = Path(args.data_dir)

    # support passing a single run dir or a parent dir containing run dirs
    if (data_dir / filename).exists():
        run_dirs = [data_dir]
    else:
        run_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    for run_dir in run_dirs:
        pt_path = run_dir / filename
        if not pt_path.exists():
            continue

        data = load_mixing_analysis_data(pt_path)
        mixing = data["mixing"]
        run_name = run_dir.name

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(run_name)

        plot_eigenvalues_polar(mixing, ax=axes[0], title="Eigenvalue Distribution")
        plot_transition_norms(mixing, ax=axes[1], title="Transition Matrix Norms")

        fig.tight_layout()
        out_path = output_dir / f"{run_name}_{'init' if args.init_data else 'trained'}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
