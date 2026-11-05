import json
import subprocess
import sys

import numpy as np
import pandas as pd

DATA_DIR = sys.argv[1]
OUT_DIR = sys.argv[2]

DATASETS = {
    "unsw": ("unsw.zip", "*/data/NF-UNSW-NB15-v2.csv", 2390275),
    "ton": ("ton.zip", "*/data/NF-ToN-IoT-v2.csv", 16940496),
    "cic": ("cic.zip", "*/data/NF-CSE-CIC-IDS2018-v2.csv", 18893708),
}

DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "DNS_QUERY_ID"]
SAMPLE_SIZE = 300000

stats = {}
for name, (zip_name, member, approx_rows) in DATASETS.items():
    rng = np.random.default_rng(412)
    rate = SAMPLE_SIZE / approx_rows
    proc = subprocess.Popen(["unzip", "-p", f"{DATA_DIR}/{zip_name}", member], stdout=subprocess.PIPE)
    parts = []
    total = 0
    attacks = 0
    attack_types = {}
    for chunk in pd.read_csv(proc.stdout, chunksize=500000, low_memory=False):
        total += len(chunk)
        attacks += int(chunk["Label"].sum())
        for k, v in chunk["Attack"].value_counts().items():
            attack_types[k] = attack_types.get(k, 0) + int(v)
        keep = rng.random(len(chunk)) < rate
        parts.append(chunk[keep].drop(columns=DROP))
    proc.wait()
    sample = pd.concat(parts, ignore_index=True)
    sample.to_parquet(f"{OUT_DIR}/{name}.parquet", index=False)
    stats[name] = {
        "total_flows": total,
        "attack_flows": attacks,
        "attack_share": attacks / total,
        "attack_types": attack_types,
        "sample_flows": len(sample),
        "sample_attack_share": float(sample["Label"].mean()),
    }
    print(name, stats[name]["total_flows"], stats[name]["attack_share"], len(sample), flush=True)

with open(f"{OUT_DIR}/dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
