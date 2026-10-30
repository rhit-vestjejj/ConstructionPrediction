"""Walk-forward evaluation, learning-to-rank and tuning for the "top-10% absolute increase" target.

Walk-forward: to predict target year Y at horizon h (origin Y-h), train on every row whose label was already known at the
origin (target year <= Y-h), i.e. the model is retrained each year, as it would be in real use.
Tuning: random search scored on the validation years only (walk-forward inside them); test years are touched once.
Test years are the same as forecasting.py's fixed split, so the numbers compare directly.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

import forecasting as fc
from geo import ROOT

OUT = ROOT / "outputs" / "results"
N_CONFIGS = 10
RNG = np.random.default_rng(0)

SPACE = {
    "xgb_clf": dict(max_depth=[3, 4, 5, 6, 8], learning_rate=[0.02, 0.04, 0.08], n_estimators=[200, 400, 800], min_child_weight=[1, 5, 20, 50],
                    subsample=[0.7, 0.9], colsample_bytree=[0.6, 0.8, 1.0]),
    "lgbm_rank": dict(num_leaves=[15, 31, 63], learning_rate=[0.02, 0.05, 0.1], n_estimators=[200, 400, 800], min_child_samples=[10, 20, 50, 100],
                      subsample=[0.7, 0.9], colsample_bytree=[0.6, 0.8, 1.0]),
}
DEFAULTS = {
    "xgb_clf": dict(max_depth=5, learning_rate=0.03, n_estimators=400, min_child_weight=20, subsample=0.8, colsample_bytree=0.8),
    "lgbm_rank": dict(num_leaves=31, learning_rate=0.05, n_estimators=400, min_child_samples=20, subsample=0.8, colsample_bytree=0.8),
}


def relevance(inc: pd.Series, origin: pd.Series) -> np.ndarray:
    """Graded label per year: 3 = top 10% absolute increase, 2 = next 10%, 1 = next 20%, 0 = rest."""
    q = inc.groupby(origin).rank(pct=True)
    return np.select([q > 0.9, q > 0.8, q > 0.6], [3, 2, 1], 0)


def fit_score(kind: str, params: dict, tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if kind == "xgb_clf":
        m = xgb.XGBClassifier(**params, n_jobs=-1, random_state=0).fit(tr[cols], tr.top)
        return m.predict_proba(te[cols])[:, 1]
    if kind == "lgbm_clf":
        m = lgb.LGBMClassifier(**params, subsample_freq=1, n_jobs=-1, verbose=-1, random_state=0).fit(tr[cols], tr.top)
        return m.predict_proba(te[cols])[:, 1]
    if kind == "rf":
        return fc.make_model("random_forest", "clf").fit(tr[cols], tr.top).predict_proba(te[cols])[:, 1]
    tr = tr.sort_values("origin")
    if kind == "lgbm_rank":
        m = lgb.LGBMRanker(objective="lambdarank", lambdarank_truncation_level=200, **params, subsample_freq=1, n_jobs=-1, verbose=-1, random_state=0)
        m.fit(tr[cols], tr.rel, group=tr.groupby("origin", sort=True).size().values)
        return m.predict(te[cols])
    if kind == "xgb_rank":
        m = xgb.XGBRanker(objective="rank:ndcg", lambdarank_num_pair_per_sample=8, **params, tree_method="hist", n_jobs=-1, random_state=0)
        m.fit(tr[cols], tr.rel, qid=tr.origin.values)
        return m.predict(te[cols])
    raise ValueError(kind)


def walk(kind, params, d, years, h, cols):
    """Out-of-sample scores for each target year in `years`, each from a model trained on labels known at its origin."""
    out = []
    for Y in years:
        tr, te = d[d.ty <= Y - h], d[d.ty == Y]
        if len(te) == 0 or tr.top.sum() < 20:
            continue
        out.append(te.assign(score=fit_score(kind, params, tr, te, cols)))
    return pd.concat(out) if out else None


def scores(p: pd.DataFrame) -> dict:
    pk = np.mean([g.nlargest(fc.TOP_K, "score").top.mean() for _, g in p.groupby("ty")])
    return {"pr_auc": float(average_precision_score(p.top, p.score)), "p_at_100": float(pk), "base_rate": float(p.top.mean())}


def sample(kind):
    return {k: v[RNG.integers(len(v))] for k, v in SPACE[kind].items()}


def rank_avg(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0][["fips", "origin", "ty", "top"]].copy()
    base["score"] = np.mean([f.groupby("ty").score.rank(pct=True).values for f in frames], axis=0)
    return base


def run():
    topf = json.loads((OUT / "top_features.json").read_text())
    fixed = pd.read_json(OUT / "metrics.json")
    recs, preds, best_cfg = [], [], {}
    for t in fc.TARGETS:
        data = fc.build_dataset(t)
        groups = fc.feature_groups(t)
        for h in fc.HORIZONS:
            d = data.dropna(subset=[f"inc{h}"]).copy()
            d["ty"] = d.origin + h
            d["top"] = d.groupby("origin")[f"inc{h}"].transform(lambda s: (s >= s.quantile(1 - fc.TOP_SHARE)).astype(float))
            d["rel"] = relevance(d[f"inc{h}"], d.origin)
            cols = groups[topf.get(f"{t.key}|{h}", "own+spatial+econ")]
            val_years = list(range(t.train_end + 1, t.val_end + 1))
            test_years = sorted(y for y in d.ty.unique() if y > t.val_end)
            t0 = time.time()
            # ---- tune on validation years (walk-forward inside them)
            chosen = {}
            if REUSE and f"{t.key}|{h}" in REUSE:  # skip the (slow) search; use the configs chosen on validation last time
                chosen = REUSE[f"{t.key}|{h}"]
            for kind in (() if chosen else ("xgb_clf", "lgbm_rank")):
                cands = [DEFAULTS[kind]] + [sample(kind) for _ in range(N_CONFIGS)]
                res = [(scores(walk(kind, c, d, val_years, h, cols))["pr_auc"], i, c) for i, c in enumerate(cands)]
                v, i, c = max(res, key=lambda r: r[0])
                chosen[kind] = c
                recs.append({"target": t.key, "horizon_years": h, "task": "tuning", "model": kind, "val_pr_auc_default": res[0][0], "val_pr_auc_tuned": v,
                             "config": json.dumps(c, default=int), "picked_default": i == 0})
            best_cfg[f"{t.key}|{h}"] = {k: {kk: (float(vv) if isinstance(vv, float) else int(vv)) for kk, vv in c.items()} for k, c in chosen.items()}
            # ---- walk-forward test: baselines, defaults, rankers, tuned, rank-average ensemble
            te_all = d[d.ty.isin(test_years)]
            runs = {
                "baseline_size": te_all.assign(score=te_all["size"]),
                "baseline_recent_increase": te_all.assign(score=np.expm1(te_all.lu_0) - te_all["size"]),
                "xgb_clf_default": walk("xgb_clf", DEFAULTS["xgb_clf"], d, test_years, h, cols),
                "lgbm_clf_default": walk("lgbm_clf", dict(num_leaves=31, learning_rate=0.03, n_estimators=400, min_child_samples=20, subsample=0.8, colsample_bytree=0.8), d, test_years, h, cols),
                "rf_default": walk("rf", {}, d, test_years, h, cols),
                "xgb_rank_default": walk("xgb_rank", DEFAULTS["xgb_clf"], d, test_years, h, cols),
                "lgbm_rank_default": walk("lgbm_rank", DEFAULTS["lgbm_rank"], d, test_years, h, cols),
                "xgb_clf_tuned": walk("xgb_clf", chosen["xgb_clf"], d, test_years, h, cols),
                "lgbm_rank_tuned": walk("lgbm_rank", chosen["lgbm_rank"], d, test_years, h, cols),
            }
            runs["ensemble_wf"] = rank_avg([runs["xgb_clf_tuned"], runs["lgbm_rank_tuned"], runs["rf_default"]])
            fx = fixed[(fixed.target == t.key) & (fixed.horizon_years == h) & (fixed.task == "top") & (fixed.model == "ensemble")]
            for name, p in runs.items():
                recs.append({"target": t.key, "horizon_years": h, "task": "walkforward", "model": name, "test_years": f"{min(test_years)}-{max(test_years)}",
                             **{f"test_{k}": v for k, v in scores(p).items()}, "fixed_split_ensemble_pr_auc": float(fx.test_pr_auc.iloc[0]) if len(fx) else None,
                             "fixed_split_ensemble_p_at_100": float(fx.test_p_at_k.iloc[0]) if len(fx) else None})
                preds.append(p[["fips", "origin", "ty", "top", "score"]].assign(target=t.key, horizon_years=h, model=name))
            r = {k: scores(v) for k, v in runs.items()}
            print(f"{t.key} h={h} ({time.time() - t0:.0f}s)  fixed-split ens PR-AUC {fx.test_pr_auc.iloc[0]:.3f} | WF: "
                  + "  ".join(f"{k} {v['pr_auc']:.3f}" for k, v in r.items() if k in ("baseline_size", "xgb_clf_default", "lgbm_rank_default", "xgb_clf_tuned", "lgbm_rank_tuned", "ensemble_wf")), flush=True)
    pd.DataFrame(recs).to_json(OUT / "walkforward_metrics.json", orient="records", indent=1)
    pd.concat(preds).to_parquet(OUT / "walkforward_predictions.parquet")
    (OUT / "walkforward_configs.json").write_text(json.dumps(best_cfg, indent=1))


REUSE = None

if __name__ == "__main__":
    import sys
    if "--reuse-configs" in sys.argv:
        REUSE = json.loads((OUT / "walkforward_configs.json").read_text())
    run()
