import json
import sys

import numpy as np
import pandas as pd

res, assets = sys.argv[1], sys.argv[2]
rows = []
for f in ["phase1_sklearn", "phase1_mlp", "multi_full", "single_60k", "multi_60k", "no_ttl"]:
    rows += json.load(open(f"{res}/{f}.json"))
df = pd.DataFrame(rows)
single = df[~df["source"].str.contains(r"\+")]
multi = df[df["source"].str.contains(r"\+")]
base = json.load(open(f"{res}/baseline.json"))
summ = pd.read_csv(f"{assets}/summary.csv").set_index("model")
mrows = pd.read_csv(f"{assets}/multi.csv")
abl = pd.read_csv(f"{assets}/ablation.csv").set_index("model")
feat = json.load(open(f"{res}/features.json"))


def cell(m, s, t, k, d=single):
    return d[(d["model"] == m) & (d["source"] == s) & (d["target"] == t)][k].mean()


claims = [
    ("LR ToN in FPR 0.154", 0.154, cell("logreg", "ton", "ton", "fpr")),
    ("LR cross F1 0.194", 0.194, summ.loc["logreg", "cross_f1"]),
    ("MLP cross F1 0.157", 0.157, summ.loc["mlp", "cross_f1"]),
    ("RF cross F1 0.049", 0.049, summ.loc["rf", "cross_f1"]),
    ("Attn cross F1 0.232", 0.232, summ.loc["attn_60k", "cross_f1"]),
    ("MLP60k cross F1 0.146", 0.146, summ.loc["mlp_60k", "cross_f1"]),
    ("MLP60k in F1 0.964", 0.964, summ.loc["mlp_60k", "in_f1"]),
    ("MLP in F1 0.968", 0.968, summ.loc["mlp", "in_f1"]),
    ("RF in F1 0.977", 0.977, summ.loc["rf", "in_f1"]),
    ("RF in AUC 0.996", 0.996, summ.loc["rf", "in_auc"]),
    ("RF cross AUC 0.413", 0.413, summ.loc["rf", "cross_auc"]),
    ("RF cross acc 0.649", 0.649, summ.loc["rf", "cross_acc"]),
    ("LR gap 0.736", 0.736, summ.loc["logreg", "gap"]),
    ("Attn gap 0.730", 0.730, summ.loc["attn_60k", "gap"]),
    ("Attn gap std 0.049", 0.049, summ.loc["attn_60k", "gap_std"]),
    ("MLP60k gap std 0.002", 0.002, summ.loc["mlp_60k", "gap_std"]),
    ("Attn cross AUC 0.496", 0.496, summ.loc["attn_60k", "cross_auc"]),
    ("MLP60k cross AUC 0.495", 0.495, summ.loc["mlp_60k", "cross_auc"]),
    ("Attn cross FPR 0.398", 0.398, summ.loc["attn_60k", "cross_fpr"]),
    ("MLP60k cross FPR 0.322", 0.322, summ.loc["mlp_60k", "cross_fpr"]),
    ("LR ton->unsw FPR 0.937", 0.937, cell("logreg", "ton", "unsw", "fpr")),
    ("LR ton->cic FPR 0.728", 0.728, cell("logreg", "ton", "cic", "fpr")),
    ("RF cic->unsw recall 0", 0.0, cell("rf", "cic", "unsw", "recall")),
    ("RF ton->unsw recall 0.003", 0.003, cell("rf", "ton", "unsw", "recall")),
    ("RF cic->unsw acc 0.960", 0.960, cell("rf", "cic", "unsw", "accuracy")),
    ("RF ton->unsw AUC 0.217", 0.217, cell("rf", "ton", "unsw", "auc")),
    ("flag all unsw 0.077", 0.077, base["unsw"]["flag_all"]["f1"]),
    ("flag all cic 0.214", 0.214, base["cic"]["flag_all"]["f1"]),
    ("flag all ton 0.780", 0.780, base["ton"]["flag_all"]["f1"]),
    ("flag all avg 0.357", 0.357, np.mean([base[t]["flag_all"]["f1"] for t in base])),
    ("TTL rule F1 0.832", 0.832, base["unsw"]["ttl_rule"]["f1"]),
    ("TTL rule FPR 0.017", 0.017, base["unsw"]["ttl_rule"]["fpr"]),
    ("TTL rule recall 1.0", 1.0, base["unsw"]["ttl_rule"]["recall"]),
    ("RF ToN L7_PROTO drop 0.111", 0.111, feat["importance"]["rf|ton|ton"]["drops"]["L7_PROTO"]),
    ("multi RF in-source 0.976", 0.976, multi[(multi["model"] == "rf") & multi["in_dataset"]]["f1"].mean()),
    ("multi MLP in-source 0.965", 0.965, multi[(multi["model"] == "mlp") & multi["in_dataset"]]["f1"].mean()),
    ("multi LR in-source 0.748", 0.748, multi[(multi["model"] == "logreg") & multi["in_dataset"]]["f1"].mean()),
    ("multi LR avg 0.173", 0.173, mrows[mrows["model"] == "logreg"]["multi_f1"].mean()),
    ("multi MLP avg 0.139", 0.139, mrows[mrows["model"] == "mlp"]["multi_f1"].mean()),
    ("multi RF avg 0.061", 0.061, mrows[mrows["model"] == "rf"]["multi_f1"].mean()),
    ("multi MLP60k avg 0.134", 0.134, mrows[mrows["model"] == "mlp_60k"]["multi_f1"].mean()),
    ("multi attn avg 0.306", 0.306, mrows[mrows["model"] == "attn_60k"]["multi_f1"].mean()),
    ("multi attn ton 0.608", 0.608, mrows[(mrows["model"] == "attn_60k") & (mrows["target"] == "ton")]["multi_f1"].iloc[0]),
    ("multi attn ton AUC 0.663", 0.663, mrows[(mrows["model"] == "attn_60k") & (mrows["target"] == "ton")]["multi_auc"].iloc[0]),
    ("multi attn cic AUC 0.806", 0.806, mrows[(mrows["model"] == "attn_60k") & (mrows["target"] == "cic")]["multi_auc"].iloc[0]),
    ("multi attn cic F1 0.297", 0.297, mrows[(mrows["model"] == "attn_60k") & (mrows["target"] == "cic")]["multi_f1"].iloc[0]),
    ("multi attn cic FPR 0.573", 0.573, mrows[(mrows["model"] == "attn_60k") & (mrows["target"] == "cic")]["multi_fpr"].iloc[0]),
    ("multi attn unsw AUC 0.182", 0.182, mrows[(mrows["model"] == "attn_60k") & (mrows["target"] == "unsw")]["multi_auc"].iloc[0]),
    ("abl RF unsw in 0.952", 0.952, abl.loc["rf", "base_unsw_in"]),
    ("abl RF unsw in no ttl 0.951", 0.951, abl.loc["rf", "nottl_unsw_in"]),
    ("abl MLP unsw in 0.940", 0.940, abl.loc["mlp", "nottl_unsw_in"]),
    ("abl LR unsw in no ttl 0.869", 0.869, abl.loc["logreg", "nottl_unsw_in"]),
    ("abl LR unsw out 0.227 -> 0.174", (0.227, 0.174), (abl.loc["logreg", "base_unsw_out"], abl.loc["logreg", "nottl_unsw_out"])),
    ("abl RF unsw out 0.088 -> 0.023", (0.088, 0.023), (abl.loc["rf", "base_unsw_out"], abl.loc["rf", "nottl_unsw_out"])),
    ("abl MLP unsw out 0.263 -> 0.091", (0.263, 0.091), (abl.loc["mlp", "base_unsw_out"], abl.loc["mlp", "nottl_unsw_out"])),
    ("abl AUC LR 0.614 -> 0.580", (0.614, 0.580), (abl.loc["logreg", "base_cross_auc"], abl.loc["logreg", "nottl_cross_auc"])),
    ("abl AUC RF 0.413 -> 0.413", (0.413, 0.413), (abl.loc["rf", "base_cross_auc"], abl.loc["rf", "nottl_cross_auc"])),
    ("abl AUC MLP 0.493 -> 0.473", (0.493, 0.473), (abl.loc["mlp", "base_cross_auc"], abl.loc["mlp", "nottl_cross_auc"])),
    ("MLP unsw->cic recall 0.690 fpr 0.468", (0.690, 0.468), (cell("mlp", "unsw", "cic", "recall"), cell("mlp", "unsw", "cic", "fpr"))),
    ("MLP noTTL unsw->cic recall 0.024 fpr 0.157", (0.024, 0.157), (cell("mlp_nottl", "unsw", "cic", "recall"), cell("mlp_nottl", "unsw", "cic", "fpr"))),
]

bad = 0
for name, want, got in claims:
    w = np.atleast_1d(want)
    g = np.atleast_1d(got)
    ok = all(abs(a - b) < 0.0005 for a, b in zip(w, np.round(g, 3)))
    bad += not ok
    print(("OK  " if ok else "BAD ") + name, np.round(g, 4))

auc_cells = single[~single["in_dataset"]].groupby(["model", "source", "target"])["auc"].mean().reset_index()
print("cells with AUC <= 0.5:", auc_cells[auc_cells["auc"] <= 0.5].groupby("model").size().to_dict())
print(f"{bad} bad claims")
