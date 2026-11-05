import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

torch.set_num_threads(2)


class SklearnModel:
    def __init__(self, kind, seed):
        if kind == "logreg":
            self.model = LogisticRegression(max_iter=2000, random_state=seed)
        else:
            self.model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=2, random_state=seed)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class MLPNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AttentionNet(nn.Module):
    def __init__(self, n_features, d=16, heads=2, layers=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d) * 0.1)
        self.bias = nn.Parameter(torch.zeros(n_features, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0]).squeeze(-1)


class TorchModel:
    def __init__(self, kind, seed, max_epochs=30, patience=3, batch_size=1024):
        self.kind = kind
        self.seed = seed
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = MLPNet(X.shape[1]) if self.kind == "mlp" else AttentionNet(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
        Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32))
        best, best_state, bad = float("inf"), None, 0
        self.epochs_run = 0
        for epoch in range(self.max_epochs):
            self.net.train()
            order = rng.permutation(len(Xt))
            for i in range(0, len(order), self.batch_size):
                b = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[b]), yt[b])
                loss.backward()
                opt.step()
            self.epochs_run = epoch + 1
            val_loss = loss_fn(self._logits(Xv), yv).item()
            if val_loss < best - 1e-4:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def _logits(self, X):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                out.append(self.net(X[i:i + 8192]))
        return torch.cat(out)

    def score(self, X):
        return torch.sigmoid(self._logits(torch.from_numpy(X))).numpy()


def make_model(kind, seed):
    if kind in ("logreg", "rf"):
        return SklearnModel(kind, seed)
    if kind == "attn":
        return TorchModel(kind, seed, max_epochs=20)
    return TorchModel(kind, seed)
