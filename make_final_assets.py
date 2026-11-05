import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import NAMES, PRETTY

res_dir, out_dir = sys.argv[1], sys.argv[2]

rows = []
for f in ["phase1_sklearn", "phase1_mlp", "multi_full", "single_60k", "multi_60k", "no_ttl"]:
    try:
        rows += json.load(open(f"{res_dir}/{f}.json"))
    except FileNotFoundError:
        print("missing", f)
df = pd.DataFrame(rows)
df["multi"] = df["source"].str.contains(r"\+")

LABEL = {"logreg": "Logistic regression", "rf": "Random forest", "mlp": "MLP",
         "mlp_60k": "MLP-60k", "attn_60k": "Attention-60k"}
MAIN = [m for m in LABEL if m in df["model"].unique()]


def f3(x):
    return f"{x:.3f}"


single = df[~df["multi"]]
summary = []
for m in MAIN:
    s = single[single["model"] == m]
    ind, cross = s[s["in_dataset"]], s[~s["in_dataset"]]
    gaps = [ind[ind["seed"] == k]["f1"].mean() - cross[cross["seed"] == k]["f1"].mean() for k in sorted(s["seed"].unique())]
    summary.append({
        "model": m, "in_f1": ind["f1"].mean(), "cross_f1": cross["f1"].mean(), "gap": np.mean(gaps), "gap_std": np.std(gaps),
        "in_auc": ind["auc"].mean(), "cross_auc": cross["auc"].mean(), "cross_recall": cross["recall"].mean(),
        "cross_fpr": cross["fpr"].mean(), "cross_acc": cross["accuracy"].mean(), "in_fpr": ind["fpr"].mean(),
    })
summary = pd.DataFrame(summary)
base = json.load(open(f"{res_dir}/baseline.json"))
flag = {t: base[t]["flag_all"] for t in NAMES}
flag_f1 = np.mean([flag[t]["f1"] for t in NAMES])
flag_acc = np.mean([flag[t]["accuracy"] for t in NAMES])
beats = {}
for m in MAIN:
    c = single[(single["model"] == m) & ~single["in_dataset"]].groupby(["source", "target"])["f1"].mean().reset_index()
    beats[m] = int(sum(r["f1"] > flag[r["target"]]["f1"] for _, r in c.iterrows()))
summary["cross_cells_beating_flag_all"] = summary["model"].map(beats)
summary.to_csv(f"{out_dir}/summary.csv", index=False)
with open(f"{out_dir}/tab_summary.tex", "w") as f:
    for _, r in summary.iterrows():
        f.write(f"{LABEL[r['model']]} & {f3(r['in_f1'])} & {f3(r['cross_f1'])} & {f3(r['gap'])} & {f3(r['in_auc'])} & {f3(r['cross_auc'])} & {f3(r['cross_recall'])} & {f3(r['cross_fpr'])} & {f3(r['cross_acc'])} \\\\\n")
    f.write("\\midrule\n")
    f.write(f"Flag every flow & {f3(flag_f1)} & {f3(flag_f1)} & 0.000 & 0.500 & 0.500 & 1.000 & 1.000 & {f3(flag_acc)} \\\\\n")

with open(f"{out_dir}/tab_indataset.tex", "w") as f:
    for m in MAIN:
        cells = []
        for n in NAMES:
            g = single[(single["model"] == m) & (single["source"] == n) & (single["target"] == n)]
            cells += [f3(g["f1"].mean()), f3(g["recall"].mean()), f3(g["fpr"].mean())]
        f.write(f"{LABEL[m]} & " + " & ".join(cells) + " \\\\\n")

grid = single.groupby(["model", "source", "target"])[["f1", "recall", "fpr", "auc", "accuracy", "precision"]].mean().reset_index()
grid.to_csv(f"{out_dir}/grid.csv", index=False)

multi = df[df["multi"] & ~df["in_dataset"]]
mrows = []
for m in MAIN:
    for tgt in NAMES:
        g = multi[(multi["model"] == m) & (multi["target"] == tgt)]
        if not len(g):
            continue
        others = [n for n in NAMES if n != tgt]
        singles = [single[(single["model"] == m) & (single["source"] == o) & (single["target"] == tgt)]["f1"].mean() for o in others]
        mrows.append({"model": m, "target": tgt, "single_best": max(singles), "single_mean": np.mean(singles),
                      "multi_f1": g["f1"].mean(), "multi_f1_std": g.groupby("seed")["f1"].mean().std(ddof=0),
                      "multi_recall": g["recall"].mean(), "multi_fpr": g["fpr"].mean(), "multi_auc": g["auc"].mean()})
mrows = pd.DataFrame(mrows)
mrows.to_csv(f"{out_dir}/multi.csv", index=False)
if len(mrows):
    with open(f"{out_dir}/tab_multi.tex", "w") as f:
        for m in MAIN:
            g = mrows[mrows["model"] == m]
            if not len(g):
                continue
            cells = []
            for tgt in NAMES:
                r = g[g["target"] == tgt].iloc[0]
                cells += [f3(r["single_mean"]), f3(r["multi_f1"])]
            f.write(f"{LABEL[m]} & " + " & ".join(cells) + f" & {f3(g['single_mean'].mean())} & {f3(g['multi_f1'].mean())} \\\\\n")

ablation = []
for m in ["logreg", "rf", "mlp"]:
    base = single[single["model"] == m]
    abl = single[single["model"] == m + "_nottl"]
    if not len(abl):
        continue
    row = {"model": m}
    for name, d in (("base", base), ("nottl", abl)):
        row[f"{name}_in"] = d[d["in_dataset"]]["f1"].mean()
        row[f"{name}_cross"] = d[~d["in_dataset"]]["f1"].mean()
        row[f"{name}_cross_auc"] = d[~d["in_dataset"]]["auc"].mean()
        u = d[d["source"] == "unsw"]
        row[f"{name}_unsw_in"] = u[u["target"] == "unsw"]["f1"].mean()
        row[f"{name}_unsw_out"] = u[u["target"] != "unsw"]["f1"].mean()
        row[f"{name}_unsw_out_auc"] = u[u["target"] != "unsw"]["auc"].mean()
    ablation.append(row)
ablation = pd.DataFrame(ablation)
ablation.to_csv(f"{out_dir}/ablation.csv", index=False)
if len(ablation):
    with open(f"{out_dir}/tab_ablation.tex", "w") as f:
        for _, r in ablation.iterrows():
            f.write(f"{LABEL[r['model']]} & {f3(r['base_unsw_in'])} & {f3(r['nottl_unsw_in'])} & {f3(r['base_unsw_out'])} & {f3(r['nottl_unsw_out'])} & {f3(r['base_cross'])} & {f3(r['nottl_cross'])} \\\\\n")

plt.rcParams.update({"font.size": 11})
heat_models = [m for m in ["logreg", "rf", "mlp", "attn_60k"] if m in MAIN]
fig, axes = plt.subplots(1, len(heat_models), figsize=(3.2 * len(heat_models), 3.4), constrained_layout=True)
for ax, m in zip(np.atleast_1d(axes), heat_models):
    mat = np.zeros((3, 3))
    for i, s in enumerate(NAMES):
        for j, t in enumerate(NAMES):
            mat[i, j] = grid[(grid["model"] == m) & (grid["source"] == s) & (grid["target"] == t)]["f1"].iloc[0]
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if mat[i, j] > 0.6 else "#1a1a1a")
    ax.set_xticks(range(3), [PRETTY[n] for n in NAMES], rotation=30, ha="right")
    ax.set_yticks(range(3), [PRETTY[n] for n in NAMES])
    ax.set_title(LABEL[m])
    ax.set_xlabel("Test dataset")
    if m == heat_models[0]:
        ax.set_ylabel("Training dataset")
fig.colorbar(im, ax=axes, shrink=0.85, label="Attack-class F1")
fig.savefig(f"{out_dir}/fig_matrix.pdf")
fig.savefig(f"{out_dir}/fig_matrix.png", dpi=130)

if len(mrows):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True, constrained_layout=True)
    x = np.arange(len(MAIN))
    for ax, tgt in zip(axes, NAMES):
        g = mrows[mrows["target"] == tgt].set_index("model").reindex(MAIN)
        ax.bar(x - 0.2, g["single_mean"], 0.4, label="One source, average", color="#9ecae1")
        ax.bar(x + 0.2, g["multi_f1"], 0.4, label="Two sources combined", color="#08519c")
        ax.set_xticks(x, [LABEL[m] for m in MAIN], rotation=35, ha="right", fontsize=9)
        ax.set_title(f"Held out {PRETTY[tgt]}")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Attack-class F1")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.savefig(f"{out_dir}/fig_multi.pdf")
    fig.savefig(f"{out_dir}/fig_multi.png", dpi=130)

from scipy.stats import spearmanr

feat = json.load(open(f"{res_dir}/features.json"))
imp, shift = feat["importance"], feat["shift"]
with open(f"{out_dir}/tab_features.tex", "w") as f:
    for kind in ["rf", "mlp"]:
        for n in NAMES:
            drops = imp[f"{kind}|{n}|{n}"]["drops"]
            top = sorted(drops.items(), key=lambda kv: -kv[1])[:3]
            cells = " & ".join(f"{k.replace('_', chr(92) + '_')} & {v:.3f}" for k, v in top)
            f.write(f"{'Random forest' if kind == 'rf' else 'MLP'} & {PRETTY[n]} & {cells} \\\\\n")
corr = {}
for kind in ["rf", "mlp"]:
    cols = list(imp[f"{kind}|unsw|unsw"]["drops"])
    for i, a in enumerate(NAMES):
        for b in NAMES[i + 1:]:
            va = [imp[f"{kind}|{a}|{a}"]["drops"][c] for c in cols]
            vb = [imp[f"{kind}|{b}|{b}"]["drops"][c] for c in cols]
            corr[f"{kind}|{a}|{b}"] = spearmanr(va, vb).statistic
ks = {c: np.mean([v for k, v in e.items() if k.startswith("ks_")]) for c, e in shift.items()}
json.dump({"spearman": corr, "ks_mean_by_feature": ks, "ks_median": float(np.median(list(ks.values()))),
           "ks_share_over_half": float(np.mean([v > 0.5 for v in ks.values()]))}, open(f"{out_dir}/feature_stats.json", "w"), indent=1)

print(summary.round(3).to_string())
print(mrows.round(3).to_string())
print(ablation.round(3).to_string())
