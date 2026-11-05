import json
import sys
from collections import defaultdict

import numpy as np

files = sys.argv[1:]
rows = []
for f in files:
    rows += json.load(open(f))

single = [r for r in rows if "+" not in r["source"]]
for m in sorted(set(r["model"] for r in single)):
    s = [r for r in single if r["model"] == m]
    ind = [r for r in s if r["in_dataset"]]
    cross = [r for r in s if not r["in_dataset"]]
    line = [m]
    for name, group in (("in", ind), ("cross", cross)):
        for k in ("f1", "auc", "recall", "fpr", "precision", "accuracy"):
            line.append(f"{name}_{k}={np.mean([r[k] for r in group]):.3f}")
    gaps = []
    for seed in sorted(set(r["seed"] for r in s)):
        a = [r["f1"] for r in ind if r["seed"] == seed]
        b = [r["f1"] for r in cross if r["seed"] == seed]
        gaps.append(np.mean(a) - np.mean(b))
    line.append(f"gap={np.mean(gaps):.3f}")
    spread = defaultdict(list)
    for r in ind:
        spread[r["source"]].append(r["f1"])
    line.append(f"max_in_seed_spread={max(max(v) - min(v) for v in spread.values()):.4f}")
    print(" ".join(line))

cells = defaultdict(list)
for r in rows:
    cells[(r["model"], r["source"], r["target"])].append(r)
for key in sorted(cells):
    g = cells[key]
    print(key, " ".join(f"{k}={np.mean([r[k] for r in g]):.3f}" for k in ("f1", "recall", "fpr", "precision", "auc", "accuracy")),
          "f1_by_seed=" + ",".join(f"{r['f1']:.3f}" for r in g))
