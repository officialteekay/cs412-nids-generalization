import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from common import NAMES, Prep, load_splits
from models import make_model

sample_dir, out_path = sys.argv[1], sys.argv[2]
splits = load_splits(sample_dir)
columns = splits["unsw"]["columns"]
rng = np.random.default_rng(0)

importance = {}
for kind in ["rf", "mlp"]:
    for src in NAMES:
        X_train, y_train = splits[src]["train"]
        X_val, y_val = splits[src]["val"]
        prep = Prep().fit(X_train)
        model = make_model(kind, 0).fit(prep.transform(X_train), y_train, prep.transform(X_val), y_val)
        for tgt in NAMES:
            X_test, y_test, _ = splits[tgt]["test"]
            keep = rng.choice(len(y_test), size=min(10000, len(y_test)), replace=False)
            Z, y = prep.transform(X_test.iloc[keep]), y_test[keep]
            base = roc_auc_score(y, model.score(Z))
            drops = {}
            for j, col in enumerate(columns):
                vals = []
                for _ in range(3):
                    Zp = Z.copy()
                    Zp[:, j] = Zp[rng.permutation(len(Zp)), j]
                    vals.append(base - roc_auc_score(y, model.score(Zp)))
                drops[col] = float(np.mean(vals))
            importance[f"{kind}|{src}|{tgt}"] = {"base_auc": base, "drops": drops}
            top = sorted(drops.items(), key=lambda kv: -kv[1])[:5]
            print(kind, src, "->", tgt, round(base, 3), [(k, round(v, 3)) for k, v in top], flush=True)

shift = {}
for col in columns:
    entry = {}
    for name in NAMES:
        X, y = splits[name]["train"]
        entry[name] = {"benign_median": float(X[col][y == 0].median()), "attack_median": float(X[col][y == 1].median())}
    for i, a in enumerate(NAMES):
        for b in NAMES[i + 1:]:
            Xa, ya = splits[a]["train"]
            Xb, yb = splits[b]["train"]
            entry[f"ks_benign_{a}_{b}"] = float(ks_2samp(Xa[col][ya == 0].values, Xb[col][yb == 0].values).statistic)
    shift[col] = entry

with open(out_path, "w") as f:
    json.dump({"importance": importance, "shift": shift}, f, indent=1)
