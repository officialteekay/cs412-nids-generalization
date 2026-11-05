import argparse
import json
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from common import NAMES, SPLIT_SEED, Prep, load_splits, metrics
from models import make_model

parser = argparse.ArgumentParser()
parser.add_argument("--samples", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--models", nargs="+", required=True)
parser.add_argument("--mode", choices=["single", "multi"], default="single")
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--subsample", type=int, default=0)
parser.add_argument("--tag", default="")
parser.add_argument("--drop", nargs="*", default=[])
args = parser.parse_args()

splits = load_splits(args.samples)
for name in splits:
    for part in ("train", "val", "test"):
        X, *rest = splits[name][part]
        splits[name][part] = (X.drop(columns=args.drop), *rest)


def take(X, y, n):
    if not n or n >= len(y):
        return X, y
    keep, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=SPLIT_SEED)
    return X.iloc[keep], y[keep]

if args.mode == "single":
    sources = [[n] for n in NAMES]
else:
    sources = [[a, b] for i, a in enumerate(NAMES) for b in NAMES[i + 1:]]

results = []
for kind in args.models:
    for seed in args.seeds:
        for src in sources:
            per_source = args.subsample // len(src)
            train_parts = [take(*splits[n]["train"], per_source) for n in src]
            val_parts = [take(*splits[n]["val"], per_source // 4) for n in src]
            X_train = pd.concat([p[0] for p in train_parts])
            y_train = np.concatenate([p[1] for p in train_parts])
            X_val = pd.concat([p[0] for p in val_parts])
            y_val = np.concatenate([p[1] for p in val_parts])
            prep = Prep().fit(X_train)
            start = time.time()
            model = make_model(kind, seed).fit(prep.transform(X_train), y_train, prep.transform(X_val), y_val)
            train_time = time.time() - start
            for tgt in NAMES:
                X_test, y_test, _ = splits[tgt]["test"]
                m = metrics(y_test, model.score(prep.transform(X_test)))
                m.update({
                    "model": kind + args.tag, "seed": seed, "train_rows": len(y_train), "source": "+".join(src), "target": tgt,
                    "in_dataset": tgt in src, "train_time_s": train_time,
                    "epochs": getattr(model, "epochs_run", None),
                })
                results.append(m)
                print(f"{kind} seed={seed} {'+'.join(src)} -> {tgt} f1={m['f1']:.3f} rec={m['recall']:.3f} fpr={m['fpr']:.4f} auc={m['auc']:.3f}", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
