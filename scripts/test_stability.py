import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from hyperconnections.cghc_fdyn import ContinuousHyperConnectionsFeatDyn


class ZeroModule(nn.Module):
    def forward(self, x, **kwargs):
        return torch.zeros_like(x)

N, M, EMBED_DIM = 4, 1, 100
INPUT_DIM = (N // M) * EMBED_DIM
DEPTH = 200
BATCH = 8
SEED = 42

CONFIGS = [
    "conservative",
    "conservative_psd_diss",
    "conservative_diag_diss",
    "conservative_laplacian",
]


def stability_test(generator_type: str) -> list[float]:
    torch.manual_seed(SEED)
    layer = ContinuousHyperConnectionsFeatDyn(
        n=N, m=M, input_dim=INPUT_DIM, embed_dim=EMBED_DIM,
        module=ZeroModule(),
        generator_type=generator_type,
        use_triton=False,
        projection="mean"
    )
    layer.eval()

    x = torch.randn(BATCH, INPUT_DIM)
    norms = [x.norm(dim=-1).mean().item()]
    with torch.no_grad():
        for _ in range(DEPTH):
            x = layer(x)
            norms.append(x.norm(dim=-1).mean().item())
    return norms


results = {cfg: stability_test(cfg) for cfg in CONFIGS}

fig, ax = plt.subplots(figsize=(8, 4))
for cfg, norms in results.items():
    ax.plot(norms, label=cfg)
ax.set_xlabel("depth")
ax.set_ylabel("mean L2 norm")
ax.set_title("Signal norm over depth (stability check)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("stability.png", dpi=150)
print("saved stability.png")

print()
for cfg, norms in results.items():
    ratio = norms[-1] / (norms[0] + 1e-8)
    status = "OK" if ratio < 100 else "EXPLODED"
    print(f"{cfg:35s}  init={norms[0]:.3f}  final={norms[-1]:.3f}  ratio={ratio:.3f}  {status}")
