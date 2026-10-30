"""Graph models for the top-10% target, evaluated exactly like walkforward.py (same data, test years, metrics).

Runs in the native arm64 env:  .venv-gnn/bin/python src/gnn.py   (reads data/datasets/ from export_datasets.py)

  mlp        no graph: isolates what the graph adds
  sage       2-layer GraphSAGE (mean aggregator) on an 8-nearest-county graph built per year
  sage_gru   GRU over each county's last 4 years of features, then GraphSAGE on the GRU state

Walk-forward: for target year Y, train on all years whose labels were known at origin Y-h. A small configuration
search (hidden size x epochs) is scored on the validation years only.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data" / "datasets"
OUT = ROOT / "outputs" / "results"
K, SEQ, SEEDS, TOP_K = 8, 4, 3, 100
CONFIGS = [dict(hidden=64, epochs=60), dict(hidden=64, epochs=150), dict(hidden=128, epochs=100)]
torch.set_num_threads(8)


class Net(nn.Module):
    def __init__(self, f_in, hidden, kind):
        super().__init__()
        self.kind = kind
        d = f_in
        if kind == "sage_gru":
            self.gru = nn.GRU(f_in, hidden, batch_first=True)
            d = hidden
        if kind == "mlp":
            self.l1, self.l2 = nn.Linear(d, hidden), nn.Linear(hidden, hidden)
        else:  # GraphSAGE: concat(self, mean of neighbours)
            self.l1, self.l2 = nn.Linear(2 * d, hidden), nn.Linear(2 * hidden, hidden)
        self.out, self.drop = nn.Linear(hidden, 1), nn.Dropout(0.2)

    def forward(self, x, A, seq=None):
        if self.kind == "sage_gru":
            _, hN = self.gru(seq)
            x = hN[-1]
        if self.kind == "mlp":
            h = self.drop(torch.relu(self.l1(x)))
            h = self.drop(torch.relu(self.l2(h)))
        else:
            h = self.drop(torch.relu(self.l1(torch.cat([x, torch.sparse.mm(A, x)], 1))))
            h = self.drop(torch.relu(self.l2(torch.cat([h, torch.sparse.mm(A, h)], 1))))
        return self.out(h).squeeze(1)


def knn_adj(lat, lon):
    xy = np.c_[lon * np.cos(np.radians(lat)), lat]
    idx = NearestNeighbors(n_neighbors=min(K + 1, len(xy))).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    n = len(xy)
    rows = np.repeat(np.arange(n), idx.shape[1])
    i = torch.tensor(np.vstack([rows, idx.ravel()]), dtype=torch.long)
    return torch.sparse_coo_tensor(i, torch.full((i.shape[1],), 1.0 / idx.shape[1]), (n, n)).coalesce()


class Prep:
    """Standardise with training-window statistics; NaN -> 0 plus missing-value indicators."""

    def __init__(self, df, cols):
        self.cols = cols
        self.med, self.std = df[cols].median(), df[cols].std().replace(0, 1).fillna(1)
        self.miss = [c for c in cols if df[c].isna().any()]

    def __call__(self, df):
        z = ((df[self.cols] - self.med) / self.std).clip(-5, 5).fillna(0).values
        m = df[self.miss].isna().values.astype(float) if self.miss else np.zeros((len(df), 0))
        return np.c_[z, m].astype(np.float32)


def graphs(d, prep, origins, kind):
    """One graph per origin year: features, adjacency, labels, (optional) feature sequence of the last SEQ years."""
    X_all = pd.DataFrame(prep(d), index=pd.MultiIndex.from_arrays([d.fips.values, d.origin.values]))
    out = []
    for o in origins:
        g = d[d.origin == o]
        if len(g) < K + 2:
            continue
        x = torch.tensor(X_all.loc[list(zip(g.fips, g.origin))].values)
        seq = None
        if kind == "sage_gru":
            steps = []
            for lag in range(SEQ - 1, -1, -1):
                keys = list(zip(g.fips, g.origin - lag))
                steps.append(X_all.reindex(keys).fillna(0).values)  # missing earlier year -> zeros
            seq = torch.tensor(np.stack(steps, 1), dtype=torch.float32)
        out.append((x, knn_adj(g.lat.values, g.lon.values), torch.tensor(g.top.values, dtype=torch.float32), seq, g))
    return out


def train_predict(kind, cfg, d, cols, train_mask, test_origin, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    prep = Prep(d[train_mask], cols)
    tr = graphs(d[train_mask], prep, sorted(d[train_mask].origin.unique()), kind)
    # the test graph needs earlier years for the GRU sequence, so pass the full history up to the test origin
    hist = d[d.origin <= test_origin]
    te = [g for g in graphs(hist, prep, [test_origin], kind)]
    if not tr or not te:
        return None
    net = Net(tr[0][0].shape[1], cfg["hidden"], kind)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(cfg["epochs"]):
        net.train()
        for i in np.random.permutation(len(tr)):
            x, A, y, seq, _ = tr[i]
            opt.zero_grad()
            loss_fn(net(x, A, seq), y).backward()
            opt.step()
    net.eval()
    x, A, y, seq, g = te[0]
    with torch.no_grad():
        return g.assign(score=torch.sigmoid(net(x, A, seq)).numpy())


def walk(kind, cfg, d, cols, years, h):
    out = []
    for Y in years:
        mask = (d.ty <= Y - h).values
        if d[mask].top.sum() < 20:
            continue
        runs = [train_predict(kind, cfg, d, cols, mask, Y - h, s) for s in range(SEEDS)]
        runs = [r for r in runs if r is not None]
        if runs:
            r = runs[0].copy()
            r["score"] = np.mean([x.score.values for x in runs], axis=0)  # average over seeds
            out.append(r)
    return pd.concat(out) if out else None


def scores(p):
    pk = np.mean([g.nlargest(TOP_K, "score").top.mean() for _, g in p.groupby("ty")])
    return {"pr_auc": float(average_precision_score(p.top, p.score)), "p_at_100": float(pk), "base_rate": float(p.top.mean())}


def main(only=None, kinds=("mlp", "sage", "sage_gru")):
    meta = json.loads((DS / "meta.json").read_text())
    recs, preds = [], []
    if only and (OUT / "gnn_metrics.json").exists():  # partial rerun: keep results for everything else
        old = pd.read_json(OUT / "gnn_metrics.json")
        recs = old[~old.target.astype(str).str.startswith(only.split("|")[0]) | ~old.horizon_years.astype(str).isin([only.split("|")[1]] if "|" in only else old.horizon_years.astype(str))].to_dict("records")
        oldp = pd.read_parquet(OUT / "gnn_predictions.parquet")
        preds = [oldp[oldp.set_index(["target", "horizon_years"]).index.isin([(r["target"], r["horizon_years"]) for r in recs])]]
    for key, m in meta.items():
        if only and not key.startswith(only):
            continue
        target, h = key.split("|")
        h = int(h)
        d = pd.read_parquet(DS / f"{target}_h{h}.parquet").reset_index(drop=True)
        cols = m["features"]
        val_years = list(range(m["train_end"] + 1, m["val_end"] + 1))
        test_years = sorted(y for y in d.ty.unique() if y > m["val_end"])
        for kind in kinds:
            t0 = time.time()
            vals = [(scores(walk(kind, c, d, cols, val_years, h))["pr_auc"], i) for i, c in enumerate(CONFIGS)]
            v, i = max(vals)
            p = walk(kind, CONFIGS[i], d, cols, test_years, h)
            s = scores(p)
            recs.append({"target": target, "horizon_years": h, "task": "walkforward", "model": kind, "config": CONFIGS[i], "val_pr_auc": v,
                         "test_years": f"{min(test_years)}-{max(test_years)}", **{f"test_{k}": x for k, x in s.items()}, "train_s": time.time() - t0})
            preds.append(p[["fips", "origin", "ty", "top", "score"]].assign(target=target, horizon_years=h, model=kind))
            print(f"{key} {kind:9s} PR-AUC {s['pr_auc']:.3f}  P@100 {s['p_at_100']:.2f}  (cfg {i}, {time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(recs).to_json(OUT / "gnn_metrics.json", orient="records", indent=1)
        pd.concat(preds).to_parquet(OUT / "gnn_predictions.parquet")


if __name__ == "__main__":
    # e.g.  gnn.py residential_permits|2 mlp,sage
    main(sys.argv[1] if len(sys.argv) > 1 else None, tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("mlp", "sage", "sage_gru"))
