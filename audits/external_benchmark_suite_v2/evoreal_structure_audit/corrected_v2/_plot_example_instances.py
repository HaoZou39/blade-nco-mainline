import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "datasets" / "external_v2"
OUT = Path(__file__).resolve().parent
df = pd.read_csv(OUT / "instance_features_corrected.csv")

picks = {
    "S1": "EvoReal_S1_100_0746",
    "S2": "EvoReal_S2_100_0890",
    "S3": "EvoReal_S3_100_0653",
}
colors = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c"}
datasets = {"S1": "EvoReal_S1_100", "S2": "EvoReal_S2_100", "S3": "EvoReal_S3_100"}

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
for ax, split in zip(axes, ["S1", "S2", "S3"]):
    iid = picks[split]
    ds = datasets[split]
    coords = np.load(DATA / ds / "coordinates.npy")
    manifest = pd.read_csv(DATA / ds / "instances.csv")
    idx = int(manifest.index[manifest.instance_id.astype(str) == iid][0])
    xy = coords[idx]
    row = df[df.instance_id == iid].iloc[0]
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=28,
        c=colors[split],
        alpha=0.9,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"{split}: {iid}\nFFT={row.fft_energy:.1f}, NN-ratio={row.nn_ratio:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("x")
    if ax is axes[0]:
        ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

fig.suptitle(
    "EvoReal frozen N=100 examples (one each from S1 / S2 / S3)",
    fontsize=12,
    y=1.02,
)
fig.tight_layout()
path = OUT / "example_s1_s2_s3_instances.png"
fig.savefig(path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(path)
