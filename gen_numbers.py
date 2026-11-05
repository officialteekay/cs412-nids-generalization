import json
import sys

import pandas as pd

assets = sys.argv[1]
grid = pd.read_csv(f"{assets}/grid.csv")
summary = pd.read_csv(f"{assets}/summary.csv")
fs = json.load(open(f"{assets}/feature_stats.json"))

main = ["logreg", "rf", "mlp", "mlp_60k", "attn_60k"]
grid = grid[grid["model"].isin(main)]
ind = grid[grid["source"] == grid["target"]]
cross = grid[grid["source"] != grid["target"]]
summary = summary[summary["model"].isin(main)]
mlp_sp = [v for k, v in fs["spearman"].items() if k.startswith("mlp|")]

values = {
    "INFMIN": f"{ind['f1'].min():.3f}",
    "INFMAX": f"{ind['f1'].max():.3f}",
    "CROSSMAX": f"{cross['f1'].max():.3f}",
    "CROSSMIN": f"{summary['cross_f1'].min():.3f}",
    "CROSSMAXAVG": f"{summary['cross_f1'].max():.3f}",
    "SPMLPMIN": f"{min(mlp_sp):.2f}",
    "SPMLPMAX": f"{max(mlp_sp):.2f}",
    "KSMED": f"{fs['ks_median']:.2f}",
    "KSSHARE": f"{100 * fs['ks_share_over_half']:.0f}",
}
with open(f"{assets}/numbers_auto.tex", "w") as f:
    for k, v in values.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(values)
