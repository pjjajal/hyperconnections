import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
    }
)


def _stack_mixing(token, channel):
    """Interleave token/channel per depth → (depth*2, B, seq_len, n, n)."""
    return torch.stack([token, channel], dim=1).flatten(0, 1)


def _show_diverged(ax, fontsize=16):
    """Mark an axis as 'diverged' and hide its frame."""
    ax.text(0.5, 0.5, "diverged", ha="center", va="center",
            fontsize=fontsize, color="red", transform=ax.transAxes)
    ax.set_axis_off()


def _save_both(fig, path):
    """Save a figure as both PNG (preview) and PDF (paper) with tight bbox."""
    path = Path(path)
    for ext in (".png", ".pdf"):
        out = path.with_suffix(ext)
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")


def load_mixing_analysis_data(path: Path):
    """Load the mixing analysis data from the given path."""
    data = torch.load(path, map_location="cpu")
    stacked = _stack_mixing(data["token_mixing"], data["channel_mixing"])
    result = {
        "mixing": stacked,
        "labels": data["labels"],
        "indices": list(range(stacked.shape[0])),
    }
    if "token_mixing_proj" in data:
        result["mixing_proj"] = _stack_mixing(
            data["token_mixing_proj"], data["channel_mixing_proj"]
        )
    return result


def get_eigenvalues(mixing_matrices):
    # mixing matrices = [depth*2, B, seq_len, n, n]
    eigvals = torch.linalg.eigvals(
        mixing_matrices.contiguous()
    )  # (depth*2, B, seq_len, n)
    eigvals = eigvals.flatten(1)  # (depth*2, B*seq_len*n)
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


def plot_eigenvalues_polar(
    mixing_matrices,
    ax=None,
    title="Eigenvalue Distribution",
    density=False,
    show_colorbar=True,
    xlim=None,
    ylim=None,
    count_norm=None,
):
    """Cartesian plot of eigenvalues in the complex plane.

    density=False: scatter coloured by layer depth.
    density=True:  log-count hexbin, all layers merged — better for summary comparisons.
    xlim/ylim: force axis limits (pass shared limits for comparable summary panels).
    count_norm: pass a shared LogNorm to align hexbin colour scales across panels.
    """
    eigvals = get_eigenvalues(mixing_matrices.float())  # (L, B*seq_len*n) complex
    L = eigvals.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if density:
        all_re = eigvals.real.numpy().flatten()
        all_im = eigvals.imag.numpy().flatten()
        extent = (xlim[0], xlim[1], ylim[0], ylim[1]) if xlim and ylim else None
        hb = ax.hexbin(
            all_re,
            all_im,
            gridsize=80,
            cmap="inferno",
            norm=count_norm
            if count_norm is not None
            else plt.matplotlib.colors.LogNorm(vmin=1),
            mincnt=1,
            extent=extent,
            linewidths=0.1,
            edgecolors="face",
        )
        if show_colorbar:
            plt.colorbar(hb, ax=ax, label="count")
        artist = hb
    else:
        cmap = plt.get_cmap("viridis")
        for layer_idx in range(L):
            ev = eigvals[layer_idx]
            ax.scatter(
                ev.real.numpy(),
                ev.imag.numpy(),
                s=0.8,
                alpha=0.3,
                color=cmap(layer_idx / L),
            )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, L))
        if show_colorbar:
            plt.colorbar(sm, ax=ax, label="layer")
        artist = sm

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "r--", linewidth=0.6, alpha=0.7)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)
    return ax, artist


def plot_eigenvalue_trajectories(mixing_matrices, title="Eigenmode Trajectories"):
    """One subplot per eigenmode: trace its path across depth on the complex plane.

    Averaged over batch and seq_len. Segments coloured by depth (viridis).
    """
    mean_mixing = mixing_matrices.float().mean(dim=(1, 2))  # (L, n, n)
    L, n = mean_mixing.shape[0], mean_mixing.shape[1]

    eigvals = torch.linalg.eigvals(mean_mixing.contiguous())  # (L, n) complex

    # Sort eigenvalues by angle at each layer for consistent mode ordering
    sorted_idx = eigvals.angle().argsort(dim=1)  # (L, n)
    eigvals = torch.gather(eigvals, 1, sorted_idx)

    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    depth_cmap = plt.get_cmap("viridis")
    theta = np.linspace(0, 2 * np.pi, 300)

    for mode in range(n):
        ax = axes_flat[mode]
        re = eigvals[:, mode].real.numpy()
        im = eigvals[:, mode].imag.numpy()

        for i in range(L - 1):
            c = depth_cmap(i / max(L - 1, 1))
            ax.plot(re[i : i + 2], im[i : i + 2], color=c, linewidth=1.5, alpha=0.9)

        ax.scatter(
            re[0],
            im[0],
            marker="o",
            s=60,
            color="limegreen",
            zorder=5,
            edgecolors="black",
            linewidths=0.6,
            label="start",
        )
        ax.scatter(
            re[-1],
            im[-1],
            marker="*",
            s=100,
            color="red",
            zorder=5,
            edgecolors="black",
            linewidths=0.6,
            label="end",
        )

        ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=0.8, alpha=0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(f"Mode {mode}")
        ax.legend(fontsize=7, loc="upper left")

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=depth_cmap, norm=plt.Normalize(0, L - 1))
    fig.colorbar(sm, ax=axes_flat[:n].tolist(), label="layer (depth)", shrink=0.8)
    fig.suptitle(title)
    return fig


def plot_transition_norms(
    mixing_matrices,
    ax=None,
    title="Transition Matrix Norms",
    vmin=None,
    vmax=None,
    show_colorbar=True,
):
    """Heatmap of ||Pi[tau, t]|| (spectral norm) for tau <= t.

    Pass vmin/vmax to share a colour scale across multiple subplots.
    Set show_colorbar=False when the caller will add a single shared colorbar.
    """
    mean_mixing = mixing_matrices.float().mean(dim=(1, 2))  # (L, n, n)
    Pi = get_transition_table(mean_mixing)  # (L, L, n, n)
    norms = torch.linalg.matrix_norm(Pi.contiguous(), ord=2)

    L = norms.shape[0]
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)
    norms_np = norms.numpy().astype(float)
    norms_np[mask.numpy()] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    valid = norms_np[~np.isnan(norms_np) & (norms_np > 0)]
    _vmin = vmin if vmin is not None else valid.min()
    _vmax = vmax if vmax is not None else valid.max()
    use_log = _vmin > 0 and (_vmax / _vmin) > 100
    norm = (
        plt.matplotlib.colors.LogNorm(vmin=_vmin, vmax=_vmax)
        if use_log
        else plt.matplotlib.colors.Normalize(vmin=_vmin, vmax=_vmax)
    )
    im = ax.imshow(norms_np, origin="lower", aspect="auto", cmap="viridis", norm=norm)
    if show_colorbar:
        cb = plt.colorbar(
            im,
            ax=ax,
            label=r"$\|\Pi(t, \tau)\|$",
        )
        if not use_log:
            cb.formatter = plt.matplotlib.ticker.FormatStrFormatter("%.2f")
            cb.update_ticks()
    ax.set_xlabel(r"$t$ (end layer)")
    ax.set_ylabel(r"$\tau$ (start layer)")
    ax.set_title(title)
    return ax, im


DIVERSITY_METRICS = {
    "orth_frac_total": "||x_orth||² / ||x||²",
    "orth_vs_mean": "||x_orth||² / n·||x_mean||²",
    "orth_vs_x0": "||x_orth||² / n·||x0_mean||²",
}


def load_stream_diversity_data(path: Path):
    data = torch.load(path, map_location="cpu")
    epochs = sorted(data["epochs"].keys())
    # Build a matrix per metric: shape (n_epochs, n_layers)
    matrices = {
        metric: torch.tensor([data["epochs"][e][metric] for e in epochs])
        for metric in DIVERSITY_METRICS
    }
    return matrices, epochs, data["labels"]


def plot_stream_diversity_heatmap(matrix, epochs, labels, ax=None, title=""):
    """Heatmap: rows = layers, cols = epochs."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(epochs) * 0.6), 7))
    im = ax.imshow(matrix.T.numpy(), aspect="auto", cmap="viridis", vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("epoch  (0 = init)")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs)
    n_layers = matrix.shape[1]
    step = max(1, n_layers // 16)
    ax.set_yticks(range(0, n_layers, step))
    ax.set_yticklabels(labels[::step], fontsize=7)
    ax.set_title(title)
    return ax


def plot_stream_diversity_by_epoch(matrix, epochs, ax=None, title=""):
    """Line plot: one curve per epoch across layer depth."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    n_epochs = len(epochs)
    cmap = plt.get_cmap("plasma")
    for i, epoch in enumerate(epochs):
        ax.plot(
            matrix[i].numpy(),
            color=cmap(i / max(n_epochs - 1, 1)),
            alpha=0.8,
            linewidth=1.2,
            label="init" if epoch == 0 else f"epoch {epoch}",
        )
    ax.set_xlabel("layer index  (token/channel interleaved)")
    ax.set_ylabel("value")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(epochs)))
    plt.colorbar(sm, ax=ax, label="epoch")
    return ax


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./experiments/runs")
    parser.add_argument("--output_dir", type=str, default="./cifar_plots")
    parser.add_argument(
        "--init_data",
        action="store_true",
        help="Whether to generate plots from data at init.",
    )
    return parser.parse_args()


def _norm_range(mixing):
    """Return (vmin, vmax) of spectral norms for the transition table of one run."""
    mean_mixing = mixing.float().mean(dim=(1, 2))
    Pi = get_transition_table(mean_mixing)
    norms = torch.linalg.matrix_norm(Pi.contiguous(), ord=2).numpy().astype(float)
    L = norms.shape[0]
    mask = np.tril(np.ones((L, L), dtype=bool), k=-1)
    norms[mask] = np.nan
    valid = norms[~np.isnan(norms) & (norms > 0)]
    return (float(valid.min()), float(valid.max())) if valid.size > 0 else (None, None)


def plot_summary_norms(run_data, output_path):
    """Single-row summary: spectral norm heatmaps, one shared colorbar on the right."""
    n = len(run_data)

    vmin, vmax = np.inf, -np.inf
    for _, mixing in run_data:
        if not mixing.isnan().any():
            lo, hi = _norm_range(mixing)
            if lo is not None:
                vmin = min(vmin, lo)
                vmax = max(vmax, hi)
    shared_vmin = vmin if not np.isinf(vmin) else None
    shared_vmax = vmax if not np.isinf(vmax) else None
    use_log = (
        shared_vmin
        and shared_vmax
        and shared_vmin > 0
        and (shared_vmax / shared_vmin) > 100
    )

    output_path = Path(output_path)
    layouts = [
        (2, math.ceil(n / 2), "2x2", (9.5, 4.5 * math.ceil(n / 2))),
        (1, n, "row", (7.5, 2.2)),
    ]
    for nrows, ncols, tag, figsize in layouts:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.flatten()
        for i in range(n, nrows * ncols):
            axes_flat[i].set_visible(False)

        last_im = None
        for idx, (label, mixing) in enumerate(run_data):
            r, c = divmod(idx, ncols)
            ax = axes_flat[idx]
            if mixing.isnan().any():
                ax.set_title(label)
                _show_diverged(ax)
            else:
                _, im = plot_transition_norms(
                    mixing,
                    ax=ax,
                    vmin=shared_vmin,
                    vmax=shared_vmax,
                    title=label,
                    show_colorbar=False,
                )
                if c > 0:
                    ax.set_ylabel("")
                if r < nrows - 1:
                    ax.set_xlabel("")
                last_im = im

        if last_im is not None:
            if tag == "row":
                # Use Axes.inset_axes (child axes) to avoid the PDF+bbox_inches="tight"
                # displacement bug in axes_grid1.inset_locator.inset_axes (mpl #27143).
                last_ax = axes_flat[n - 1]
                cax = last_ax.inset_axes([1.05, 0, 0.08, 1.0])
                cb = fig.colorbar(last_im, cax=cax, label=r"$\|\Pi(t, \tau)\|$")
            else:
                cb = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8,
                                  aspect=30, label=r"$\|\Pi(t, \tau)\|$")
            if not use_log:
                cb.formatter = plt.matplotlib.ticker.FormatStrFormatter("%.2f")
                cb.update_ticks()

        out = output_path.with_name(f"{output_path.stem}_{tag}{output_path.suffix}")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

def plot_summary_eigenvalues(run_data, output_path):
    """Single-row summary: eigenvalue density hexbins, shared axes and colorbar."""
    n = len(run_data)

    # Compute shared axis limits and hexbin count scale across all runs
    all_re, all_im_vals = [], []
    for _, mixing in run_data:
        if not mixing.isnan().any():
            ev = get_eigenvalues(mixing.float())
            all_re.append(ev.real.numpy().flatten())
            all_im_vals.append(ev.imag.numpy().flatten())

    shared_xlim = shared_ylim = count_norm = None
    if all_re:
        re_cat = np.concatenate(all_re)
        im_cat = np.concatenate(all_im_vals)
        margin = 0.1
        half = (
            max(re_cat.max() - re_cat.min(), im_cat.max() - im_cat.min()) / 2 + margin
        )
        cx, cy = (re_cat.min() + re_cat.max()) / 2, (im_cat.min() + im_cat.max()) / 2
        shared_xlim = (max(cx - half, -1.25), min(cx + half, 1.25))
        shared_ylim = (max(cy - half, -1.25), min(cy + half, 1.25))

        # shared hexbin count scale: estimate max bin count via histogram2d
        gridsize = 60
        global_max = 0
        for re, im_v in zip(all_re, all_im_vals):
            counts, _, _ = np.histogram2d(
                re, im_v, bins=gridsize, range=[shared_xlim, shared_ylim]
            )
            global_max = max(global_max, int(counts.max()))
        count_norm = plt.matplotlib.colors.LogNorm(vmin=1, vmax=max(global_max, 1))

    output_path = Path(output_path)
    layouts = [
        (2, math.ceil(n / 2), "2x2", (7.5, 3.5 * math.ceil(n / 2))),
        (1, n, "row", (7.5, 2.5)),
    ]
    for nrows, ncols, tag, figsize in layouts:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.flatten()
        for i in range(n, nrows * ncols):
            axes_flat[i].set_visible(False)

        last_hb = None
        for idx, (label, mixing) in enumerate(run_data):
            r, c = divmod(idx, ncols)
            ax = axes_flat[idx]
            if mixing.isnan().any():
                ax.set_title(label)
                _show_diverged(ax)
            else:
                _, hb = plot_eigenvalues_polar(
                    mixing,
                    ax=ax,
                    density=True,
                    title=label,
                    show_colorbar=False,
                    xlim=shared_xlim,
                    ylim=shared_ylim,
                    count_norm=count_norm,
                )
                if c > 0:
                    ax.set_ylabel("")
                if r < nrows - 1:
                    ax.set_xlabel("")
                last_hb = hb

        if last_hb is not None:
            if tag == "row":
                # Use Axes.inset_axes (child axes) to avoid the PDF+bbox_inches="tight"
                # displacement bug in axes_grid1.inset_locator.inset_axes (mpl #27143).
                last_ax = axes_flat[n - 1]
                cax = last_ax.inset_axes([1.05, 0, 0.08, 1.0])
                fig.colorbar(last_hb, cax=cax, label="Count")
            else:
                fig.colorbar(last_hb, ax=axes.ravel().tolist(), shrink=0.8, aspect=30,
                             label="Count")

        out = output_path.with_name(f"{output_path.stem}_{tag}{output_path.suffix}")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

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

    suffix = "init" if args.init_data else "trained"
    summary_data = []

    for run_dir in run_dirs:
        pt_path = run_dir / filename
        if not pt_path.exists():
            continue

        data = load_mixing_analysis_data(pt_path)
        mixing = data["mixing"]
        mixing_proj = data.get("mixing_proj")
        run_name = run_dir.name
        summary_data.append((run_name, mixing))

        run_out = output_dir / run_name
        run_out.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(run_name)

        if mixing.isnan().any():
            for ax in axes:
                _show_diverged(ax, fontsize=20)
        else:
            plot_eigenvalues_polar(mixing, ax=axes[0], title="Eigenvalue Distribution")
            plot_transition_norms(mixing, ax=axes[1], title="Transition Matrix Norms")

        fig.tight_layout()
        _save_both(fig, run_out / f"{suffix}.png")
        plt.close(fig)

        if mixing.isnan().any():
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            fig2.suptitle(run_name)
            _show_diverged(ax2, fontsize=20)
        else:
            fig2 = plot_eigenvalue_trajectories(
                mixing, title=f"{run_name}  —  Eigenmode Trajectories"
            )
        _save_both(fig2, run_out / f"{suffix}_trajectories.png")
        plt.close(fig2)

        if mixing_proj is not None:
            fig_p, axes_p = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            fig_p.suptitle(run_name)
            if mixing_proj.isnan().any():
                for ax in axes_p:
                    _show_diverged(ax, fontsize=20)
            else:
                plot_eigenvalues_polar(mixing, ax=axes_p[0], title="A  (raw)")
                plot_eigenvalues_polar(
                    mixing_proj, ax=axes_p[1], title="P + A(I − P)  (projected)"
                )
            _save_both(fig_p, run_out / f"{suffix}_eigenvalues_proj.png")
            plt.close(fig_p)

        diversity_path = run_dir / "stream_diversity_layerwise.pt"
        if diversity_path.exists():
            matrices, epochs, labels = load_stream_diversity_data(diversity_path)
            fig, axes = plt.subplots(
                len(DIVERSITY_METRICS), 2, figsize=(16, 5 * len(DIVERSITY_METRICS))
            )
            fig.suptitle(f"{run_name}  —  stream diversity")
            for row, (metric, ylabel) in enumerate(DIVERSITY_METRICS.items()):
                plot_stream_diversity_heatmap(
                    matrices[metric],
                    epochs,
                    labels,
                    ax=axes[row, 0],
                    title=f"{metric}  (heatmap)",
                )
                plot_stream_diversity_by_epoch(
                    matrices[metric],
                    epochs,
                    ax=axes[row, 1],
                    title=f"{metric}: {ylabel}",
                )
            fig.tight_layout()
            _save_both(fig, run_out / "diversity.png")
            plt.close(fig)

    SUMMARY_ORDER = [
        ("ghc_d48_ar2.0_p4_n4m1", "GHC"),
        ("mhc_d48_ar2.0_p4_n4m1", "mHC"),
        ("cghc_d48_ar2.0_p4_n4m1_cons_psd", "CGHC (PSD)"),
        ("cghc_d48_ar2.0_p4_n4m1_cons_psd_proj_mean", "CGHC (PSD + proj)"),
    ]
    lookup = {name: mixing for name, mixing in summary_data}
    ordered = [(label, lookup[key]) for key, label in SUMMARY_ORDER if key in lookup]
    if len(ordered) > 1:
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(exist_ok=True)
        plot_summary_norms(ordered, summary_dir / f"norms_{suffix}.png")
        plot_summary_eigenvalues(ordered, summary_dir / f"eigenvalues_{suffix}.png")
        plot_summary_norms(ordered, summary_dir / f"norms_{suffix}.pdf")
        plot_summary_eigenvalues(ordered, summary_dir / f"eigenvalues_{suffix}.pdf")


if __name__ == "__main__":
    args = parse_args()
    main(args)
