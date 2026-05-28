"""Plot results from the grid-world localization sweep.

Figures saved to --out-dir (default experiments/synthetic_grid_world/plots/):
  fig1_acc_vs_dim.{png,pdf}      -- Accuracy vs backbone dim      (1×4 subplots by level)
  fig4_acc_vs_params.{png,pdf}   -- Accuracy vs param count       (1×4 subplots by level)
  fig5_acc_vs_ressize.{png,pdf}  -- Accuracy vs effective residual size
  fig6_acc_vs_flops.{png,pdf}    -- Accuracy vs forward-pass FLOPs

Baseline line: uses depth-matched baseline where available, standard baseline otherwise.
HC lines: tonal shading (lighter → darker) for increasing n.

Run from the repo root:
  uv run --extra experiments python -m experiments.synthetic_grid_world.plot_results
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ── Style (matches experiments/synthetic_dynamics/plotting.py) ─────────────────
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "savefig.dpi": 200,
})

HC_LABELS = {
    None:          "Baseline",
    "identity_hc": "Identity HC",
    "ghc":         "GHC",
    "mhc":         "MHC",
    "cghc":        "CGHC",
}

HC_COLORS = {
    None:          "black",
    "identity_hc": "tab:red",
    "ghc":         "tab:green",
    "mhc":         "tab:orange",
    "cghc":        "tab:blue",
}

MODEL_ORDER = [None, "identity_hc", "ghc", "mhc", "cghc"]

_N_MARKERS: dict[Optional[int], str] = {None: "o", 4: "s", 8: "^", 16: "D", 32: "P"}
_MARKER_CYCLE = ["X", "v", "<", ">", "h", "H"]
_HC_COLOR_CYCLE = ["tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
_auto_hc_colors: dict[str, str] = {}

LEVELS = ["h1_easy", "h2_medium", "h3_hard", "h4_veryhard"]
LEVEL_LABELS = {
    "h1_easy":     "Easy",
    "h2_medium":   "Medium",
    "h3_hard":     "Hard",
    "h4_veryhard": "Very Hard",
}
DIMS = [16, 32, 64, 128]

BASELINE_STYLE = dict(color="black", ls="-", marker="o", label="Baseline",
                      alpha=0.9, markersize=6)
FIG_SIZE = (13, 3.5)  # 1×4 layout


def _marker(n: Optional[int]) -> str:
    if n not in _N_MARKERS:
        idx = len(_N_MARKERS) - len({None, 4, 8, 16, 32})
        _N_MARKERS[n] = _MARKER_CYCLE[idx % len(_MARKER_CYCLE)]
    return _N_MARKERS[n]


def _base_color(hc_type: Optional[str]) -> str:
    if hc_type in HC_COLORS:
        return HC_COLORS[hc_type]
    if hc_type not in _auto_hc_colors:
        _auto_hc_colors[hc_type] = _HC_COLOR_CYCLE[len(_auto_hc_colors) % len(_HC_COLOR_CYCLE)]
    return _auto_hc_colors[hc_type]


def _tonal_color(base: str, level: float) -> tuple:
    """Blend base color toward white. level=1.0 → full color, 0.0 → white."""
    r, g, b, a = mcolors.to_rgba(base)
    f = level
    return (r + (1 - r) * (1 - f), g + (1 - g) * (1 - f), b + (1 - b) * (1 - f), a)


def _n_tone(n: int, all_ns: list[int]) -> float:
    """Map n to [0.45, 1.0]: smaller n → lighter, larger n → darker."""
    if len(all_ns) <= 1:
        return 1.0
    idx = sorted(all_ns).index(n)
    return 0.45 + 0.55 * idx / (len(all_ns) - 1)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class RunInfo:
    run_dir: Path
    level: str
    dim: int
    num_layers: int
    hc_type: Optional[str]
    n: Optional[int]
    m: Optional[int]
    is_matched: bool
    matched_n: Optional[int]
    best_acc: float
    _param_count: Optional[int] = field(default=None, repr=False, compare=False)
    _flops: Optional[int] = field(default=None, repr=False, compare=False)

    @property
    def series_key(self) -> tuple:
        return (self.hc_type, self.n, self.is_matched, self.matched_n)

    def eff_res(self) -> int:
        return self.dim if self.hc_type is None else (self.n or 1) * self.dim

    def param_count(self) -> Optional[int]:
        if self._param_count is None:
            self._param_count = _compute_param_count(self)
        return self._param_count

    def flops(self) -> Optional[int]:
        if self._flops is None:
            self._flops = _compute_flops(self)
        return self._flops


def _load_run(run_dir: Path) -> Optional[RunInfo]:
    cfg_path = run_dir / "config.json"
    met_path = run_dir / "metrics.json"
    if not cfg_path.exists() or not met_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    metrics = json.loads(met_path.read_text())
    if not metrics:
        return None

    train_cache = cfg["data"].get("train_cache_path", "")
    if not train_cache:
        return None
    level = Path(train_cache).stem.replace("_train", "")
    if level not in LEVELS:
        return None

    dim = cfg["model"]["dim"]
    num_layers = cfg["model"].get("num_layers") or max(2, dim // 10)
    hc = cfg["model"].get("hc", {})
    hc_type = hc.get("type")
    n = hc.get("n") if hc_type else None
    m = hc.get("m") if hc_type else None

    match = re.search(r"_baseline_n(\d+)eq_", run_dir.name)
    is_matched = match is not None and hc_type is None
    matched_n = int(match.group(1)) if match else None

    best_acc = max(e["val_acc"] for e in metrics)
    return RunInfo(
        run_dir=run_dir, level=level, dim=dim, num_layers=num_layers,
        hc_type=hc_type, n=n, m=m,
        is_matched=is_matched, matched_n=matched_n,
        best_acc=best_acc,
    )


def scan_runs(runs_dir: Path) -> list[RunInfo]:
    best: dict[tuple, RunInfo] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        run = _load_run(d)
        if run is None:
            continue
        key = (run.level, run.dim, run.hc_type, run.n, run.is_matched, run.matched_n)
        if key not in best or run.best_acc > best[key].best_acc:
            best[key] = run
    return list(best.values())


def _compute_param_count(run: RunInfo) -> Optional[int]:
    try:
        from omegaconf import OmegaConf
        from experiments.synthetic_grid_world.train import build_hc_cls, N_ACTIONS
        from experiments.synthetic_grid_world.model import Transformer
        c = OmegaConf.create(json.loads((run.run_dir / "config.json").read_text()))
        dim = c.model.dim
        num_layers = c.model.num_layers or max(2, dim // 10)
        num_heads = c.model.num_heads or max(1, dim // 16)
        hc_cls, input_dim = build_hc_cls(c.model.hc, dim)
        n_pos = c.data.n_rows * c.data.n_cols
        model = Transformer(
            n_grid_tokens=n_pos, n_observations=c.data.n_colours,
            n_actions=N_ACTIONS, n_positions=n_pos,
            seq_len=c.data.trajectory_length + 1,
            dim=dim, input_dim=input_dim, num_heads=num_heads,
            ffn_ratio=c.model.ffn_ratio, num_layers=num_layers,
            hc_cls=hc_cls, qkv_bias=c.model.qkv_bias, proj_bias=c.model.proj_bias,
        )
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None


def _compute_flops(run: RunInfo) -> Optional[int]:
    """Measure forward-pass FLOPs via torch.utils.flop_counter.FlopCounterMode."""
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
        from omegaconf import OmegaConf
        from experiments.synthetic_grid_world.train import build_hc_cls, N_ACTIONS
        from experiments.synthetic_grid_world.model import Transformer

        c = OmegaConf.create(json.loads((run.run_dir / "config.json").read_text()))
        dim = c.model.dim
        num_layers = c.model.num_layers or max(2, dim // 10)
        num_heads = c.model.num_heads or max(1, dim // 16)
        hc_cls, input_dim = build_hc_cls(c.model.hc, dim)
        n_pos = c.data.n_rows * c.data.n_cols

        model = Transformer(
            n_grid_tokens=n_pos, n_observations=c.data.n_colours,
            n_actions=N_ACTIONS, n_positions=n_pos,
            seq_len=c.data.trajectory_length + 1,
            dim=dim, input_dim=input_dim, num_heads=num_heads,
            ffn_ratio=c.model.ffn_ratio, num_layers=num_layers,
            hc_cls=hc_cls, qkv_bias=c.model.qkv_bias, proj_bias=c.model.proj_bias,
        ).eval()

        obs  = torch.zeros(1, c.data.trajectory_length + 1, dtype=torch.long)
        acts = torch.zeros(1, c.data.trajectory_length + 1, dtype=torch.long)
        grid = torch.zeros(1, c.data.n_rows, c.data.n_cols, dtype=torch.long)

        with FlopCounterMode(display=False) as counter:
            model(obs, acts, grid)

        return counter.get_total_flops()
    except Exception:
        return None


# ── Style computation ──────────────────────────────────────────────────────────

def make_styles(runs: list[RunInfo]) -> dict[tuple, dict]:
    """Pre-compute per-series-key style with tonal shading for increasing n."""
    ns_per_type: dict[str, set] = {}
    for r in runs:
        if r.hc_type is not None and not r.is_matched:
            ns_per_type.setdefault(r.hc_type, set()).add(r.n)
    ns_per_type = {k: sorted(v) for k, v in ns_per_type.items()}

    styles: dict[tuple, dict] = {}
    for r in runs:
        key = r.series_key
        if key in styles:
            continue
        hc_type, n, is_matched, matched_n = key
        base = _base_color(hc_type)

        if hc_type is None:
            # Handled separately via BASELINE_STYLE / best-baseline logic
            styles[key] = dict(**BASELINE_STYLE)
        elif is_matched:
            styles[key] = dict(color=base, ls="--", marker=_marker(matched_n),
                               label=f"Baseline (matched n={matched_n})", alpha=0.9, markersize=6)
        else:
            all_ns = ns_per_type.get(hc_type, [n])
            color = _tonal_color(base, _n_tone(n, all_ns))
            lbl = HC_LABELS.get(hc_type, hc_type or "Unknown")
            styles[key] = dict(color=color, ls="-", marker=_marker(n),
                               label=f"{lbl} n={n}", alpha=0.9, markersize=6)
    return styles


# ── Baseline helpers ───────────────────────────────────────────────────────────

def _best_baseline_per_dim(runs: list[RunInfo], level: str) -> list[RunInfo]:
    """For each dim, use the highest-accuracy matched baseline if one exists,
    otherwise fall back to the standard (unmatched) baseline."""
    result = []
    for dim in DIMS:
        candidates = [r for r in runs if r.level == level and r.dim == dim and r.hc_type is None]
        matched = [r for r in candidates if r.is_matched]
        standard = [r for r in candidates if not r.is_matched]
        best = max(matched, key=lambda r: r.best_acc) if matched else (standard[0] if standard else None)
        if best:
            result.append(best)
    return result


# ── Plot helpers ───────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf}}")


def legend_dedupe(axes) -> tuple[list, list]:
    seen: set[str] = set()
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    return handles, labels


def _series_order(runs: list[RunInfo]) -> list[tuple]:
    keys: dict[tuple, tuple] = {}
    for r in runs:
        k = r.series_key
        hc_type, n, is_matched, matched_n = k
        keys[k] = (
            1 if is_matched else 0,
            MODEL_ORDER.index(hc_type) if hc_type in MODEL_ORDER else 99,
            n or 0,
        )
    return [k for k, _ in sorted(keys.items(), key=lambda x: x[1])]


def _setup_axes(axes: np.ndarray, titles: list[str], xlabel: str, ylabel: str):
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel(ylabel)


def _add_legend(fig, axes, ncol: int = 5):
    h, l = legend_dedupe(axes)
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.13, 1, 0.97])


def _plot_baseline(ax, bl_runs: list[RunInfo], x_fn):
    if not bl_runs:
        return
    pts = sorted((x_fn(r), r.best_acc) for r in bl_runs)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, **BASELINE_STYLE)


def _plot_hc_series(ax, level_runs: list[RunInfo], styles: dict, x_fn):
    hc_runs = [r for r in level_runs if r.hc_type is not None and not r.is_matched]
    order = [k for k in _series_order(hc_runs) if k[0] is not None]
    raw: dict[tuple, list] = {}
    for r in hc_runs:
        val = x_fn(r)
        if val is not None:
            raw.setdefault(r.series_key, []).append((val, r.best_acc))
    for key in order:
        if key not in raw or not raw[key]:
            continue
        xs, ys = zip(*sorted(raw[key]))
        s = styles.get(key, {})
        ax.plot(xs, ys, color=s["color"], ls=s["ls"], marker=s["marker"],
                label=s["label"], alpha=s.get("alpha", 0.9), markersize=s.get("markersize", 6))


# ── Figure 1: Accuracy vs backbone dim ────────────────────────────────────────

def fig1_acc_vs_dim(runs: list[RunInfo], styles: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=FIG_SIZE, sharey=True)
    for ax, level in zip(axes, LEVELS):
        lr = [r for r in runs if r.level == level]
        _plot_baseline(ax, _best_baseline_per_dim(runs, level), x_fn=lambda r: r.dim)
        _plot_hc_series(ax, lr, styles, x_fn=lambda r: r.dim)
        ax.set_xticks(DIMS)
    _setup_axes(axes, [LEVEL_LABELS[lv] for lv in LEVELS], "Backbone dim", "Best val accuracy")
    _add_legend(fig, axes, ncol=5)
    fig.suptitle("Accuracy vs backbone dim", y=1.01)
    save_fig(fig, out_dir, "fig1_acc_vs_dim")


# ── Figure 4: Accuracy vs param count ─────────────────────────────────────────

def fig4_acc_vs_params(runs: list[RunInfo], styles: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=FIG_SIZE, sharey=True)
    for ax, level in zip(axes, LEVELS):
        lr = [r for r in runs if r.level == level]
        _plot_baseline(ax, _best_baseline_per_dim(runs, level),
                       x_fn=lambda r: r.param_count())
        _plot_hc_series(ax, lr, styles, x_fn=lambda r: r.param_count())
        ax.set_xscale("log")
    _setup_axes(axes, [LEVEL_LABELS[lv] for lv in LEVELS], "Parameters", "Best val accuracy")
    _add_legend(fig, axes, ncol=5)
    fig.suptitle("Accuracy vs parameter count", y=1.01)
    save_fig(fig, out_dir, "fig4_acc_vs_params")


# ── Figure 5: Accuracy vs effective residual size ─────────────────────────────

def fig5_acc_vs_ressize(runs: list[RunInfo], styles: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=FIG_SIZE, sharey=True)
    for ax, level in zip(axes, LEVELS):
        lr = [r for r in runs if r.level == level]
        _plot_baseline(ax, _best_baseline_per_dim(runs, level),
                       x_fn=lambda r: r.eff_res())
        _plot_hc_series(ax, lr, styles, x_fn=lambda r: r.eff_res())
        ax.set_xscale("log", base=2)
    _setup_axes(axes, [LEVEL_LABELS[lv] for lv in LEVELS],
                "Effective residual size  (n × dim)", "Best val accuracy")
    _add_legend(fig, axes, ncol=5)
    fig.suptitle("Accuracy vs effective residual size", y=1.01)
    save_fig(fig, out_dir, "fig5_acc_vs_ressize")


# ── Figure 6: Accuracy vs FLOPs ───────────────────────────────────────────────

def fig6_acc_vs_flops(runs: list[RunInfo], styles: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 4, figsize=FIG_SIZE, sharey=True)
    for ax, level in zip(axes, LEVELS):
        lr = [r for r in runs if r.level == level]
        _plot_baseline(ax, _best_baseline_per_dim(runs, level),
                       x_fn=lambda r: r.flops())
        _plot_hc_series(ax, lr, styles, x_fn=lambda r: r.flops())
        ax.set_xscale("log")
    _setup_axes(axes, [LEVEL_LABELS[lv] for lv in LEVELS],
                "Forward-pass FLOPs", "Best val accuracy")
    _add_legend(fig, axes, ncol=5)
    fig.suptitle("Accuracy vs FLOPs", y=1.01)
    save_fig(fig, out_dir, "fig6_acc_vs_flops")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser("Plot grid-world localization sweep results.")
    parser.add_argument("--runs-dir", default="experiments/synthetic_grid_world/runs")
    parser.add_argument("--out-dir",  default="experiments/synthetic_grid_world/plots")
    return parser.parse_args()


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)

    runs = scan_runs(runs_dir)
    print(f"Loaded {len(runs)} runs from {runs_dir}")
    for r in sorted(runs, key=lambda r: (r.level, r.hc_type or "", r.n or 0, r.dim)):
        tag = (r.hc_type or "baseline") + (f" n={r.n}" if r.n else "")
        if r.is_matched:
            tag = f"matched(n={r.matched_n})"
        print(f"  {r.level:20s}  dim={r.dim:3d}  {tag:25s}  acc={r.best_acc:.3f}")

    styles = make_styles(runs)

    fig1_acc_vs_dim(runs, styles, out_dir)
    fig4_acc_vs_params(runs, styles, out_dir)
    fig5_acc_vs_ressize(runs, styles, out_dir)
    fig6_acc_vs_flops(runs, styles, out_dir)
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
