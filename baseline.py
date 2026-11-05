import json
import sys

import numpy as np

from common import NAMES, load_splits, metrics

splits = load_splits(sys.argv[1])
out = {}
for t in NAMES:
    X, y, _ = splits[t]["test"]
    out[t] = {"flag_all": metrics(y, np.ones(len(y))), "ttl_rule": metrics(y, (X["MIN_TTL"].values != 31).astype(float)),
              "attack_share": float(y.mean())}
json.dump(out, open(sys.argv[2], "w"), indent=1, default=float)
print({t: round(v["flag_all"]["f1"], 3) for t, v in out.items()})
