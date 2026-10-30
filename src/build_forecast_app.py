"""Build outputs/forecast_app.html: map of model predictions (forecast + backtest) by category, horizon and model.

Forward forecasts: each model family is refit on every labelled row, then applied to the latest data year.
Top-10% view: walk-forward ensemble (walkforward.py) = rank average of tuned XGBoost, tuned LightGBM ranker and Random
Forest; probabilities come from isotonic maps fit on the walk-forward out-of-sample predictions.
Spike/growth views: fixed-split models from forecasting.py (validation-learned ensemble weights, raw probabilities).
Backtests: saved out-of-sample predictions only (walk-forward for the top-10% view), never in-sample.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

import forecasting as fc
from geo import ROOT, centroids, load_geojson

RES = ROOT / "outputs" / "results"
ML = ("xgboost", "lightgbm", "random_forest")
MODELS_BT = ("baseline_persistence", *ML, "ensemble")
FEAT = "own+spatial+econ"
TOPF = json.loads((RES / "top_features.json").read_text())
WF_CFG = json.loads((RES / "walkforward_configs.json").read_text())
WF = pd.read_parquet(RES / "walkforward_predictions.parquet")
WF_MAP = {"ensemble": "ensemble_wf", "xgboost": "xgb_clf_tuned", "lightgbm": "lgbm_rank_tuned", "random_forest": "rf_default",
          "baseline_size": "baseline_size", "baseline_recent_increase": "baseline_recent_increase"}


def wf_calibrators(target, h):
    """Isotonic map: within-year percentile rank of each model's score -> P(top 10%), fit on walk-forward out-of-sample predictions."""
    from sklearn.isotonic import IsotonicRegression
    out = {}
    for key, name in WF_MAP.items():
        g = WF[(WF.target == target) & (WF.horizon_years == h) & (WF.model == name)]
        pct = g.score if name == "ensemble_wf" else g.groupby("ty").score.rank(pct=True)
        out[key] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pct, g.top)
    return out


def ints(a, scale):
    return [None if not np.isfinite(v) else int(round(v * scale)) for v in a]


def forward(t, data, metrics):
    """Refit on all labelled rows, predict the latest origin. Returns {h: payload}."""
    cols = fc.feature_groups(t)[FEAT]
    latest_origin = int(data.origin.max())
    latest = data[data.origin == latest_origin].reset_index(drop=True)
    out = {}
    for h in fc.HORIZONS:
        y, b = f"y{h}", f"boom{h}"
        lab = data.dropna(subset=[y, b])
        if not len(metrics[(metrics.target == t.key) & (metrics.horizon_years == h)]):
            continue
        reg, clf, contrib = {}, {}, None
        for name in ML:
            reg[name] = fc.make_model(name, "reg").fit(lab[cols], lab[y]).predict(latest[cols])
            m = fc.make_model(name, "clf").fit(lab[cols], lab[b])
            clf[name] = m.predict_proba(latest[cols])[:, 1]
            if name == "xgboost":
                c = m.get_booster().predict(xgb.DMatrix(latest[cols]), pred_contribs=True)[:, :-1]
                top = np.argsort(-np.abs(c), axis=1)[:, :3]
                contrib = [[[cols[j], round(float(c[i, j]), 3), None if np.isnan(latest.iloc[i][cols[j]]) else round(float(latest.iloc[i][cols[j]]), 3)] for j in top[i]] for i in range(len(latest))]
                imp = dict(zip(cols, m.feature_importances_.round(4).tolist()))
        w = {k: metrics[(metrics.target == t.key) & (metrics.horizon_years == h) & (metrics.task == k) & (metrics.model == "ensemble")].ensemble_weights.iloc[0] for k in ("reg", "clf")}
        reg["ensemble"] = sum(w["reg"][n] * reg[n] for n in ML)
        clf["ensemble"] = sum(w["clf"][n] * clf[n] for n in ML)
        # "top" target: walk-forward ensemble retrained on every labelled year (tuned configs from walkforward.py)
        import walkforward as wfm
        tcols = fc.feature_groups(t)[TOPF.get(f"{t.key}|{h}", FEAT)]
        labt = lab.assign(top=lab.groupby("origin")[f"inc{h}"].transform(lambda s: (s >= s.quantile(1 - fc.TOP_SHARE)).astype(float)),
                          rel=wfm.relevance(lab[f"inc{h}"], lab.origin))
        cfg = WF_CFG[f"{t.key}|{h}"]
        raw = {"xgboost": wfm.fit_score("xgb_clf", cfg["xgb_clf"], labt, latest, tcols), "lightgbm": wfm.fit_score("lgbm_rank", cfg["lgbm_rank"], labt, latest, tcols),
               "random_forest": wfm.fit_score("rf", {}, labt, latest, tcols)}
        pct = {k: pd.Series(v).rank(pct=True).values for k, v in raw.items()}
        pct["ensemble"] = np.mean([pct[k] for k in ML], axis=0)
        cal = wf_calibrators(t.key, h)
        ptop = {k: cal[k].predict(v) for k, v in pct.items()}
        m = xgb.XGBClassifier(**cfg["xgb_clf"], n_jobs=-1, random_state=0).fit(labt[tcols], labt.top)  # explain the top-10% view
        c = m.get_booster().predict(xgb.DMatrix(latest[tcols]), pred_contribs=True)[:, :-1]
        top = np.argsort(-np.abs(c), axis=1)[:, :3]
        contrib = [[[tcols[j], round(float(c[i, j]), 3), None if np.isnan(latest.iloc[i][tcols[j]]) else round(float(latest.iloc[i][tcols[j]]), 3)] for j in top[i]] for i in range(len(latest))]
        imp = dict(zip(tcols, m.feature_importances_.round(4).tolist()))
        out[h] = {"origin": latest_origin, "fips": latest.fips.tolist(), "prob": {k: ints(v, 1000) for k, v in clf.items()}, "ptop": {k: ints(v, 1000) for k, v in ptop.items()},
                  "growth": {k: ints(v, 100) for k, v in reg.items()}, "drivers": contrib, "importance": imp, "base_rate": round(float(lab[b].mean()), 3),
                  "n_train": int(len(lab)), "ensemble_weights": {k: {a: float(c) for a, c in v.items()} for k, v in w.items()}}
    return out


def backtest(t, preds, preds_top):
    p = preds[preds.target == t.key]
    ptp = preds_top[preds_top.target == t.key]
    out, cals = {}, {}
    for h, g in p.groupby("horizon_years"):
        rows = {}
        for (origin, split), gg in g.groupby(["origin", "split"]):
            piv = {}
            for task, col in (("prob", "prob"), ("pred", "pred")):
                piv[task] = gg.pivot_table(index="fips", columns="model", values=col)
            base = gg.drop_duplicates("fips").set_index("fips")[["y", "boom"]].sort_index()
            fips = base.index
            # top-10% view: walk-forward out-of-sample predictions (test years only), calibrated like the forecast
            gw = WF[(WF.target == t.key) & (WF.horizon_years == h) & (WF.origin == origin)]
            topd = {}
            if len(gw):
                cal = cals.setdefault(h, wf_calibrators(t.key, h))
                for key, name in WF_MAP.items():
                    gm = gw[gw.model == name].set_index("fips")
                    if len(gm):
                        sc = gm.score if name == "ensemble_wf" else gm.score.rank(pct=True)
                        topd[key] = ints(pd.Series(cal[key].predict(sc), index=gm.index).reindex(fips).values, 1000)
                tact = gw.drop_duplicates("fips").set_index("fips").top.reindex(fips)
            rows[int(origin)] = {"split": split, "fips": fips.tolist(), "boom": [int(v) for v in base.boom], "growth": ints(base.y.values, 100),
                                 **({"top": [None if pd.isna(v) else int(v) for v in tact], "ptop": topd} if topd else {}),
                                 "prob": {m: ints(piv["prob"][m].reindex(fips).values, 1000) for m in MODELS_BT if m in piv["prob"]},
                                 "pred": {m: ints(piv["pred"][m].reindex(fips).values, 100) for m in MODELS_BT if m in piv["pred"]}}
        out[int(h)] = rows
    return out


def metro_payload():
    """H3-cell permit forecasts for the metros with open permit data (see metro_cells.py). Forward = refit on all labelled rows."""
    import h3
    import metro_cells as mc
    from geo import assign_counties

    df = pd.read_parquet(ROOT / "data" / "metro_cells.parquet")
    pr = pd.read_parquet(RES / "metro_predictions.parquet")
    mm = pd.read_json(RES / "metro_metrics.json")
    w = mm[(mm.model == "ensemble") & (mm.horizon_years == 1) & (mm.task.isna() if "task" in mm else True)].ensemble_weights.iloc[0]
    lab = df.dropna(subset=["y1"])
    cols = mc.FEATS + mc.SITE if json.loads((RES / "metro_features.json").read_text())["1"] == "history+site" else mc.FEATS
    models = {n: mc.make(n).fit(lab[cols], lab.y1) for n in ("xgboost", "lightgbm", "random_forest")}
    out = {}
    labels = {"fortworth": "Fort Worth, TX", "mecklenburg": "Charlotte / Mecklenburg, NC", "columbus": "Columbus, OH"}
    for metro, last in mc.LAST_COMPLETE.items():
        cur = df[(df.metro_name == metro) & (df.origin == last)].reset_index(drop=True)
        fwd = {n: m.predict_proba(cur[cols])[:, 1] for n, m in models.items()}
        fwd["ensemble"] = sum(w[n] * fwd[n] for n in models)
        bt = {}
        for o, g in pr[(pr.metro == metro) & (pr.horizon_years == 1)].groupby("origin"):
            piv = g.pivot_table(index="cell", columns="model", values="prob")
            y = g.drop_duplicates("cell").set_index("cell").y.reindex(piv.index)
            bt[int(o)] = {"cells": piv.index.tolist(), "y": y.astype(int).tolist(), "split": g.split.iloc[0],
                          "prob": {m: ints(piv[m].values, 1000) for m in piv.columns}}
        cells = sorted(set(cur.cell) | {c for b in bt.values() for c in b["cells"]})
        feats = [{"type": "Feature", "id": c, "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[round(lng, 5), round(lat, 5)] for lat, lng in h3.cell_to_boundary(c)] + [[round(h3.cell_to_boundary(c)[0][1], 5), round(h3.cell_to_boundary(c)[0][0], 5)]]]}} for c in cells]
        ll = np.array([h3.cell_to_latlng(c) for c in cells])
        counties = sorted({f for f in assign_counties(ll[:, 1], ll[:, 0]) if f})
        out[metro] = {"label": labels[metro], "origin": last, "target_year": last + 1, "geojson": {"type": "FeatureCollection", "features": feats}, "counties": counties,
                      "fwd": {"cells": cur.cell.tolist(), "prob": {k: ints(v, 1000) for k, v in fwd.items()}, "hist": cur[["n0", "n1", "n2"]].values.tolist(), "n_cum": cur.n_cum.tolist(),
                              "indwh3": cur.indwh3.tolist(), "dist": [round(v, 1) for v in cur.dist_center_km]}, "bt": bt}
    return out


def dc_timed_payload(payload):
    """Timed data center forecast (dc_timed.py) + the 915-project layer (data_dc_projects.py). Probabilities via an
    isotonic map fit on the walk-forward out-of-sample predictions (2021-2025)."""
    from sklearn.isotonic import IsotonicRegression
    dt_all = pd.read_parquet(ROOT / "data" / "dc_timed_forecast.parquet")
    dt, dt_prev = dt_all[dt_all.origin == dt_all.origin.max()], dt_all[dt_all.origin == dt_all.origin.max() - 1]
    from sklearn.metrics import average_precision_score
    allp = pd.read_parquet(RES / "dc_timed_predictions.parquet")
    allp["ty"] = allp.origin + 1
    # choose the model on 2021-2022 (validation); 2023-2025 is the untouched test
    split = {m: {"val_pr_auc": float(average_precision_score(g[g.ty <= 2022].y, g[g.ty <= 2022].score)),
                 "test_pr_auc": float(average_precision_score(g[g.ty >= 2023].y, g[g.ty >= 2023].score)),
                 "test_top50": float(g[g.ty >= 2023].groupby("ty").apply(lambda x: x.nlargest(50, "score").y.sum(), include_groups=False).mean())}
             for m, g in allp.groupby("model")}
    chosen = max((m for m in split if not m.startswith("baseline")), key=lambda m: split[m]["val_pr_auc"])
    xg = allp[allp.model == chosen]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xg.score, xg.y)
    payload["dct"] = {"year": int(dt.origin.max()) + 1, "fips": dt.fips.tolist(), "prob": ints(iso.predict(dt[f"p_{chosen}"]), 1000), "proj_cum": dt.proj_cum.astype(int).tolist(),
                      "chosen": chosen, "split": split,
                      "bt": {int(o) + 1: {"fips": g.fips.tolist(), "prob": ints(iso.predict(g.score), 1000), "y": g.y.astype(int).tolist()} for o, g in xg.groupby("origin")},
                      "metrics": json.loads((RES / "dc_timed_metrics.json").read_text()), "importance": json.loads((RES / "dc_timed_importance.json").read_text()),
                      # expected number of counties with a new project in the current year, as forecast from the year before
                      "prev": {"year": int(dt_prev.origin.max()) + 1, "expected_counties": round(float(iso.predict(dt_prev[f"p_{chosen}"]).sum()), 1)} if len(dt_prev) else None}
    pj = pd.read_parquet(ROOT / "data" / "dc_projects.parquet")
    cen = centroids().set_index("fips")
    pj = pj[pj.fips.isin(cen.index)]
    s = lambda v, n=80: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)[:n]
    if (RES / "dc_scenarios.json").exists():  # scenario grids (scenarios.py) + national capex link
        payload["dcs"] = {**json.loads((RES / "dc_scenarios.json").read_text()), "capex": json.loads((RES / "capex_link.json").read_text())}
    payload["projects"] = [{"name": s(r.name), "operator": s(r.operator, 60), "fips": r.fips, "year": int(r.year), "type": "data center", "status": s(r.status, 30),
                            "mw": None if pd.isna(r.mw) else float(r.mw), "source": s(r.source, 300), "lat": float(cen.lat[r.fips]), "lon": float(cen.lon[r.fips])} for r in pj.itertuples()]


def main():
    metrics = pd.read_json(RES / "metrics.json")
    preds = pd.read_parquet(RES / "predictions.parquet")
    preds_top = pd.read_parquet(RES / "predictions_top.parquet")
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    dc = pd.read_parquet(ROOT / "data" / "datacenters.parquet")
    geo = load_geojson()
    names = panel.drop_duplicates("fips", keep="last").set_index("fips").name.to_dict()
    best = json.loads((RES / "best_models.json").read_text())

    payload = {"targets": {}, "best": best, "names": names, "dc_counts": dc.fips.value_counts().to_dict(),
               "dc": {"lat": dc.lat.round(4).tolist(), "lon": dc.lon.round(4).tolist(), "text": (dc.name + np.where(dc.operator != "", " (" + dc.operator + ")", "")).tolist()}}
    keep_cols = ["target", "horizon_years", "task", "model", "features", "n_train", "n_val", "n_test", "train_years", "val_years", "test_years", "val_mae", "test_mae", "test_rmse", "test_p_at_k",
                 "val_pr_auc", "test_pr_auc", "test_base_rate", "test_brier", "test_ece", "test_precision_topk", "test_recall_topk", "train_s", "infer_ms_per_1k", "reg_mae", "clf_pr_auc", "top_pr_auc", "top_p_at_k"]
    for t in fc.TARGETS:
        print("forecasting", t.key)
        data = fc.build_dataset(t)
        S, _ = fc.load_series(t)
        years = [int(y) for y in S.columns]
        payload["targets"][t.key] = {
            "label": t.label, "unit": "permitted units" if t.source == "permits" else "jobs", "forward": forward(t, data, metrics), "backtest": backtest(t, preds, preds_top),
            "history": {"years": years, "values": {f: ints(S.loc[f].values, 1) for f in data[data.origin == data.origin.max()].fips}},
        }
    payload["metrics"] = json.loads(metrics[metrics.task.isin(["reg", "clf", "ablation", "top", "ablation_top"])][[c for c in keep_cols if c in metrics.columns]].round(4).to_json(orient="records"))
    wfm_ = pd.read_json(RES / "walkforward_metrics.json")
    payload["wf_metrics"] = json.loads(wfm_[wfm_.task == "walkforward"][["target", "horizon_years", "model", "test_years", "test_pr_auc", "test_p_at_100", "fixed_split_ensemble_pr_auc"]].round(4).to_json(orient="records"))
    if (RES / "gnn_metrics.json").exists():
        g = pd.read_json(RES / "gnn_metrics.json")
        payload["gnn_metrics"] = json.loads(g[["target", "horizon_years", "model", "test_pr_auc", "test_p_at_100"]].round(4).to_json(orient="records"))
    dc_timed_payload(payload)
    # surge view = max(top-10% chance, spike chance): measured on backtest years where both exist
    from sklearn.metrics import average_precision_score
    sm = []
    for key, T in payload["targets"].items():
        for h, rows in T["backtest"].items():
            ys, ss, k100, yrs = [], [], [], []
            for o, r in rows.items():
                if "ptop" not in r or "ensemble" not in r["ptop"]:
                    continue
                top, boom = np.array([np.nan if v is None else v for v in r["top"]], float), np.array(r["boom"], float)
                sc = np.fmax(np.array([np.nan if v is None else v for v in r["ptop"]["ensemble"]], float), np.array([np.nan if v is None else v for v in r["prob"]["ensemble"]], float))
                ok = ~np.isnan(top) & ~np.isnan(sc)
                y = np.maximum(top[ok], boom[ok])
                ys.append(y); ss.append(sc[ok]); yrs.append(o + h)
                k100.append(y[np.argsort(-sc[ok])[:100]].mean())
            if ys:
                y, sc = np.concatenate(ys), np.concatenate(ss)
                sm.append({"target": key, "horizon_years": int(h), "years": f"{min(yrs)}-{max(yrs)}", "pr_auc": float(average_precision_score(y, sc)), "base_rate": float(y.mean()), "p_at_100": float(np.mean(k100))})
    payload["surge_metrics"] = sm
    for t in fc.TARGETS:  # the top-10% view uses the walk-forward ensemble
        for h in fc.HORIZONS:
            payload["best"][f"{t.key}|{h}|top"] = "ensemble"
    payload["geo_holdout"] = json.loads(metrics[metrics.task == "geo_holdout"][["target", "model", "region", "mae_standard", "mae_heldout_region", "mae_persistence"]].round(4).to_json(orient="records"))

    # ---- data center siting propensity (see datacenter_model.py): not a timed forecast
    sc = pd.read_parquet(ROOT / "data" / "datacenter_scores.parquet")
    ep = pd.read_parquet(ROOT / "data" / "epoch_sites.parquet")
    tl = pd.read_parquet(ROOT / "data" / "epoch_timelines.parquet")
    cent = centroids().set_index("fips")
    last = tl.groupby("Data center").Date.max()
    sites = []
    for _, r in ep.iterrows():
        f = r.fips if isinstance(r.fips, str) else None
        lat, lon = (r.lat, r.lon) if pd.notna(r.lat) else ((cent.lat.get(f), cent.lon.get(f)) if f else (None, None))
        if lat is None or pd.isna(lat):
            continue
        sites.append({"name": r.Name, "owner": str(r.Owner).split(" #")[0], "fips": f, "lat": round(float(lat), 4), "lon": round(float(lon), 4), "loc": r.location_source,
                      "mw_now": None if pd.isna(r["Current power (MW)"]) else round(float(r["Current power (MW)"])), "peak_mw": None if pd.isna(r.planned_peak_mw) else round(float(r.planned_peak_mw)),
                      "first": None if pd.isna(r.first_dated_milestone) else str(r.first_dated_milestone.date()), "last": str(last.get(r.Name, pd.NaT))[:10]})
    models = ["logistic", "xgboost", "lightgbm", "random_forest"]
    payload["dcm"] = {"fips": sc.fips.tolist(), "like": ints(sc.likelihood.values, 1000), "pct": ints(sc.percentile.values, 10), "n_dc": sc.n_dc_osm.tolist(),
                      "epoch": sc.epoch_sites.tolist(), "ws": [int(v) for v in sc.white_space], "drivers": [json.loads(d) for d in sc.drivers],
                      "scores": {m: ints(sc[f"score_{m}"].values, 1000) for m in models + ["ensemble"]}}
    payload["dc_metrics"] = json.loads((RES / "datacenter_metrics.json").read_text())
    payload["epoch_sites"] = sites
    payload["metro"] = metro_payload()
    mm = pd.read_json(RES / "metro_metrics.json")
    payload["metro_metrics"] = json.loads(mm[mm.task.isna() & mm.test_roc_auc.notna()][["horizon_years", "model", "n_train", "n_val", "n_test", "train_years", "val_years", "test_years", "test_pr_auc", "test_roc_auc", "test_precision_top5pct",
                                                              "test_recall_top5pct", "test_base_rate", "test_ece", "quiet_pr_auc", "quiet_recall_top5pct", "train_s"]].round(4).to_json(orient="records"))
    payload["metro_holdout"] = json.loads(mm[mm.task == "metro_holdout"][["metro", "n_test", "base_rate", "pr_auc_heldout_metro", "pr_auc_trained_incl_metro", "pr_auc_baseline_recent_history"]].round(4).to_json(orient="records"))

    (RES / "app_payload.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    render(payload, geo)


def render(payload, geo):
    """Fill the template; `build_forecast_app.py --template-only` re-renders from the cached payload in seconds."""
    html = (ROOT / "src" / "forecast_template.html").read_text()
    html = html.replace("__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False)).replace("__GEO__", json.dumps(geo, separators=(",", ":")))
    out = ROOT / "outputs" / "forecast_app.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import sys
    if "--template-only" in sys.argv:
        render(json.loads((RES / "app_payload.json").read_text()), load_geojson())
    else:
        main()
