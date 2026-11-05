import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import NAMES, PRETTY

out_dir = sys.argv[1]
files = sys.argv[2:]

rows = []
for f in files:
    rows += json.load(open(f))
df = pd.DataFrame(rows)

MODEL_NAMES = {"logreg": "Logistic regression", "rf": "Random forest", "mlp": "MLP", "attn": "Self-attention"}
models = [m for m in MODEL_NAMES if m in df["model"].unique()]

single = df[~df["source"].str.contains(r"\+")]
agg = single.groupby(["model", "source", "target"])[["f1", "recall", "fpr", "precision", "auc", "accuracy"]].agg(["mean", "std"]).reset_index()
agg.columns = ["_".join(c).strip("_") for c in agg.columns]
agg.to_csv(f"{out_dir}/single_source_summary.csv", index=False)

gap = []
for m in models:
    s = single[single["model"] == m]
    per_seed = []
    for seed, g in s.groupby("seed"):
        ind = g[g["in_dataset"]]["f1"].mean()
        cross = g[~g["in_dataset"]]["f1"].mean()
        per_seed.append((ind, cross, ind - cross))
    a = np.array(per_seed)
    gap.append({
        "model": m,
        "in_f1": a[:, 0].mean(), "cross_f1": a[:, 1].mean(), "gap": a[:, 2].mean(), "gap_std": a[:, 2].std(),
        "in_auc": s[s["in_dataset"]]["auc"].mean(), "cross_auc": s[~s["in_dataset"]]["auc"].mean(),
        "in_recall": s[s["in_dataset"]]["recall"].mean(), "cross_recall": s[~s["in_dataset"]]["recall"].mean(),
        "in_fpr": s[s["in_dataset"]]["fpr"].mean(), "cross_fpr": s[~s["in_dataset"]]["fpr"].mean(),
    })
gap = pd.DataFrame(gap)
gap.to_csv(f"{out_dir}/gap_summary.csv", index=False)
print(gap.round(3).to_string())

multi = df[df["source"].str.contains(r"\+")]
if len(multi):
    held = multi[~multi["in_dataset"]]
    magg = held.groupby(["model", "source", "target"])[["f1", "recall", "fpr", "auc"]].agg(["mean", "std"]).reset_index()
    magg.columns = ["_".join(c).strip("_") for c in magg.columns]
    magg.to_csv(f"{out_dir}/multi_source_summary.csv", index=False)
    print(magg.round(3).to_string())

fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.0), constrained_layout=True)
axes = np.atleast_1d(axes)
for ax, m in zip(axes, models):
    mat = np.zeros((3, 3))
    for i, s in enumerate(NAMES):
        for j, t in enumerate(NAMES):
            mat[i, j] = single[(single["model"] == m) & (single["source"] == s) & (single["target"] == t)]["f1"].mean()
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if mat[i, j] > 0.6 else "#1a1a1a")
    ax.set_xticks(range(3), [PRETTY[n] for n in NAMES], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(3), [PRETTY[n] for n in NAMES], fontsize=8)
    ax.set_title(MODEL_NAMES[m], fontsize=9)
    ax.set_xlabel("Test dataset", fontsize=8)
    if m == models[0]:
        ax.set_ylabel("Training dataset", fontsize=8)
fig.colorbar(im, ax=axes, shrink=0.8, label="Attack-class F1")
fig.savefig(f"{out_dir}/f1_matrix.pdf")
fig.savefig(f"{out_dir}/f1_matrix.png", dpi=150)
