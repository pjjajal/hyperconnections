import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

CSV_PATH = "benchmark_reports/benchmark_expm_norm_correctness_2026_07_06.CSV"
OUT_PATH = "benchmark_reports/expm_squaring_offset_error_2026_07_06.png"

# Use ["science"] only if you have LaTeX installed.
# Use ["science", "no-latex"] otherwise.
plt.style.use(["science", "ieee", "no-latex"])

df = pd.read_csv(CSV_PATH)

# Squaring-count offset from T18 recommendation
df["delta_S"] = df["S"] - df["s_req"]

# Match each row against the baseline row with S == s_req
keys = ["config", "variant", "check"]

baseline = (
    df[df["S"] == df["s_req"]]
    .loc[:, keys + ["rel_err"]]
    .rename(columns={"rel_err": "rel_err_baseline"})
)

plot_df = df.merge(baseline, on=keys, how="inner")

# Avoid divide-by-zero / log issues
eps = np.finfo(float).tiny
plot_df["log2_rel_err_ratio"] = np.log2(
    np.maximum(plot_df["rel_err"], eps)
    / np.maximum(plot_df["rel_err_baseline"], eps)
)

checks = list(plot_df["check"].drop_duplicates())
fig, axes = plt.subplots(
    1,
    len(checks),
    figsize=(3.2 * len(checks), 2.8),
    sharey=True,
    constrained_layout=True,
)

if len(checks) == 1:
    axes = [axes]

for ax, check in zip(axes, checks):
    sub = plot_df[plot_df["check"] == check].copy()

    offsets = sorted(sub["delta_S"].unique())

    # Box plot: distribution over configs at each offset
    data = [
        sub.loc[sub["delta_S"] == d, "log2_rel_err_ratio"].to_numpy()
        for d in offsets
    ]

    ax.boxplot(
        data,
        positions=offsets,
        widths=0.55,
        showfliers=False,
        patch_artist=False,
    )

    # Jittered points: individual test cases
    rng = np.random.default_rng(0)
    for d in offsets:
        y = sub.loc[sub["delta_S"] == d, "log2_rel_err_ratio"].to_numpy()
        x = d + rng.normal(0.0, 0.045, size=len(y))
        ax.scatter(x, y, s=8, alpha=0.35, linewidths=0)

    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.axvline(0.0, linestyle=":", linewidth=0.8)

    ax.set_title(check)
    ax.set_xlabel(r"$\Delta S = S - s_{\mathrm{req}}$")
    ax.set_xticks(offsets)

axes[0].set_ylabel(
    r"$\log_2\left(\mathrm{relerr}(S) / "
    r"\mathrm{relerr}(S=s_{\mathrm{req}})\right)$"
)

fig.suptitle("Effect of T18 Squaring Count (S) Offset on Relative Numerical Error")
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
# plt.show()

# print(f"Saved figure to {OUT_PATH}")