import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

NAMES = ["unsw", "ton", "cic"]
PRETTY = {"unsw": "UNSW-NB15", "ton": "ToN-IoT", "cic": "CIC-IDS2018"}
SPLIT_SEED = 412


def load_splits(sample_dir):
    splits = {}
    for name in NAMES:
        df = pd.read_parquet(f"{sample_dir}/{name}.parquet")
        y = df["Label"].astype(int).values
        attack = df["Attack"].values
        X = df.drop(columns=["Label", "Attack"]).astype(float)
        idx = np.arange(len(df))
        train_idx, rest_idx = train_test_split(idx, test_size=0.3, stratify=y, random_state=SPLIT_SEED)
        val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=SPLIT_SEED)
        splits[name] = {
            "train": (X.iloc[train_idx], y[train_idx]),
            "val": (X.iloc[val_idx], y[val_idx]),
            "test": (X.iloc[test_idx], y[test_idx], attack[test_idx]),
            "columns": list(X.columns),
        }
    return splits


class Prep:
    def fit(self, X):
        self.scaler = StandardScaler().fit(np.log1p(np.clip(X.values, 0, None)))
        return self

    def transform(self, X):
        Z = self.scaler.transform(np.log1p(np.clip(X.values, 0, None)))
        return np.clip(Z, -10, 10).astype(np.float32)


def metrics(y, score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "auc": roc_auc_score(y, score) if len(set(y)) > 1 else float("nan"),
        "accuracy": (tp + tn) / len(y),
    }
