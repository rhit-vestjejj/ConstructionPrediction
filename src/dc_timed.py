"""Timed data center forecast: will a county get a NEW data center project (first public signal) next year?

Labels: data/dc_projects.parquet (news / trade press / public records, compiled by research agents + Epoch AI).
Features use only information available at the origin year t (projects announced <= t, jobs in t, ACS published by t).
Static siting factors (power plants 2023, prices, grid, land, climate, hazards) are slightly look-ahead for early
years; they change slowly. OpenStreetMap site counts are NOT used: they include sites built after t.
Evaluation: walk-forward. For each test year Y, train on every label year < Y, predict Y. Label years with a
partial year of news (the current year) are excluded.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_context import lean_asof
from forecasting import NO_WAGE_TAX_STATES
from geo import ROOT, centroids

OUT = ROOT / "outputs" / "results"
FIRST_ORIGIN, LAST_LABEL = 2016, 2025  # 2026 news is a partial year: never a label, but it is the forecast origin
FORECAST_ORIGIN = LAST_LABEL + 1  # forecast the year after the current one from everything known so far
TEST_YEARS = list(range(2021, LAST_LABEL + 1))
K = 8
STATIC = ["mw_total", "mw_150km", "ind_price", "kv345_dist_km", "kv230_dist_km", "n_utilities", "cdd", "risk_score",
          "interstate_dist_km", "rail_dist_km", "land_value_log"]
DYN = ["jobs518", "estabs518", "jobs23", "jobs2362", "jobs493", "g518", "units", "pop_lu", "inc_lu",
       "proj_cum", "proj_recent", "yrs_since_proj", "nbr_proj_cum", "nbr_proj_recent", "state_proj_recent",
       "county_gop_margin", "state_gop_margin", "prop_tax_rate", "no_wage_income_tax", "lat", "lon"]
POWER = ["op_mw_100km", "prop_mw_100km", "prop_gas_mw_100km", "prop_mw_25km", "prop_mw_100km_chg"]  # EIA-860, vintage t-1
# Planned-power features were tested (Oct 2026): validation PR-AUC fell 0.244 -> 0.231 and the 2026 blind test fell
# from 26 to 19 hits in the top 50, so they are kept out of the live model. Set USE_POWER = True to re-test.
USE_POWER = False
FEATS = DYN + STATIC + (POWER if USE_POWER else [])


def panel() -> pd.DataFrame:
    cen = centroids().set_index("fips")
    q = pd.read_parquet(ROOT / "data" / "qcew.parquet")
    jobs = {c: q[q.industry_code == c].pivot_table(index="area_fips", columns="year", values="annual_avg_emplvl") for c in ("518", "23", "2362", "493")}
    est518 = q[q.industry_code == "518"].pivot_table(index="area_fips", columns="year", values="annual_avg_estabs")
    units = pd.read_parquet(ROOT / "data" / "panel.parquet").pivot_table(index="fips", columns="year", values="units", aggfunc="sum")
    acs = pd.read_parquet(ROOT / "data" / "acs.parquet")
    A = {v: acs.pivot_table(index="fips", columns="vintage", values=v) for v in ("pop", "med_income", "med_re_taxes", "med_home_value")}
    E = pd.read_parquet(ROOT / "data" / "elections.parquet")
    static = pd.read_parquet(ROOT / "data" / "infra_features.parquet").join(pd.read_parquet(ROOT / "data" / "context_static.parquet"), rsuffix="_c")
    pr = pd.read_parquet(ROOT / "data" / "dc_projects.parquet")
    pw = pd.read_parquet(ROOT / "data" / "power_pipeline.parquet")
    PW = {v: g.set_index("fips") for v, g in pw.groupby("vintage")}
    keep = cen.index
    xy = cen[["lon", "lat"]].values
    nbr = NearestNeighbors(n_neighbors=K + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    state = pd.Series(keep.str[:2], index=keep)
    rows = []
    for t in range(FIRST_ORIGIN, FORECAST_ORIGIN + 1):  # rows after LAST_LABEL - 1 have no complete label year
        jt = min(t, max(jobs["23"].columns))  # QCEW annual ends 2024: carry the latest year forward
        f = pd.DataFrame(index=keep)
        for c, nm in (("518", "jobs518"), ("23", "jobs23"), ("2362", "jobs2362"), ("493", "jobs493")):
            f[nm] = np.log1p(jobs[c].reindex(keep)[jt])
        f["estabs518"] = np.log1p(est518.reindex(keep)[jt])
        f["g518"] = f.jobs518 - np.log1p(jobs["518"].reindex(keep)[jt - 2])
        f["units"] = np.log1p(units.reindex(keep)[min(t, max(units.columns))])  # permits end 2025: carry forward
        v = t - 2
        if v in A["pop"].columns:
            f["pop_lu"], f["inc_lu"] = np.log(A["pop"].reindex(keep)[v]), np.log(A["med_income"].reindex(keep)[v])
            f["prop_tax_rate"] = (A["med_re_taxes"][v] / A["med_home_value"][v]).reindex(keep)
        known = pr[pr.year <= t]
        cnt = known.groupby("fips").size().reindex(keep).fillna(0)
        rec = known[known.year >= t - 1].groupby("fips").size().reindex(keep).fillna(0)
        last = known.groupby("fips").year.max().reindex(keep)
        f["proj_cum"], f["proj_recent"], f["yrs_since_proj"] = cnt.values, rec.values, (t - last).fillna(30).values
        f["nbr_proj_cum"], f["nbr_proj_recent"] = cnt.values[nbr].sum(1), rec.values[nbr].sum(1)
        f["state_proj_recent"] = rec.groupby(state).transform("sum").values - rec.values
        lean = lean_asof(E, t).reindex(keep)
        f["county_gop_margin"], f["state_gop_margin"] = pd.to_numeric(lean.get("county_gop_margin"), errors="coerce"), pd.to_numeric(lean.get("state_gop_margin"), errors="coerce")
        f["no_wage_income_tax"] = state.isin(NO_WAGE_TAX_STATES).astype(float).values
        f["lat"], f["lon"] = cen.lat, cen.lon
        for c in STATIC:
            f[c] = static[c].reindex(keep)
        v_pw = max(v for v in PW if v <= t - 1)  # EIA-860 vintage published before the forecast is made
        for c in ("op_mw_100km", "prop_mw_100km", "prop_gas_mw_100km", "prop_mw_25km"):
            f[c] = np.log1p(PW[v_pw][c].reindex(keep).values)
        prev = PW.get(v_pw - 1)
        f["prop_mw_100km_chg"] = f.prop_mw_100km - np.log1p(prev["prop_mw_100km"].reindex(keep).values) if prev is not None else np.nan
        f["origin"] = t
        if t + 1 <= LAST_LABEL:
            f["y"] = keep.isin(pr[pr.year == t + 1].fips).astype(float)
        rows.append(f)
    return pd.concat(rows).rename_axis("fips").reset_index()


def make(name):
    return {"xgboost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, min_child_weight=2, n_jobs=-1, random_state=0),
            "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.04, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=10, n_jobs=-1, verbose=-1, random_state=0),
            "logistic": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))}[name]()


def evaluate(p: pd.DataFrame) -> dict:
    pk = []
    rec = []
    for _, g in p.groupby("origin"):
        top = g.nlargest(50, "score")
        pk.append(top.y.mean())
        rec.append(g.nlargest(int(0.05 * len(g)), "score").y.sum() / max(g.y.sum(), 1))
    return {"pr_auc": float(average_precision_score(p.y, p.score)), "roc_auc": float(roc_auc_score(p.y, p.score)), "base_rate": float(p.y.mean()),
            "precision_top50": float(np.mean(pk)), "recall_top5pct": float(np.mean(rec)), "positives": int(p.y.sum())}


def main():
    d = panel()
    lab = d.dropna(subset=["y"])
    print("county-years", len(lab), "| positives by label year:", lab.groupby(lab.origin + 1).y.sum().astype(int).to_dict())
    preds, recs = [], []
    runs = {}
    for name in ("xgboost", "lightgbm", "logistic"):
        out = []
        for Y in TEST_YEARS:
            tr, te = lab[lab.origin + 1 < Y], lab[lab.origin + 1 == Y]
            if tr.y.sum() < 20:
                continue
            out.append(te.assign(score=make(name).fit(tr[FEATS], tr.y).predict_proba(te[FEATS])[:, 1]))
        runs[name] = pd.concat(out)
    test = lab[(lab.origin + 1).isin(runs["xgboost"].origin.unique() + 1)]
    runs["ensemble"] = runs["xgboost"].assign(score=np.mean([runs[n].groupby("origin").score.rank(pct=True).values for n in ("xgboost", "lightgbm", "logistic")], axis=0))
    runs["baseline_prior_projects"] = test.assign(score=test.proj_cum + 0.5 * test.proj_recent + 0.01 * test.nbr_proj_cum)
    runs["baseline_size"] = test.assign(score=test.jobs23.fillna(0))
    runs["baseline_hosting_jobs"] = test.assign(score=test.jobs518.fillna(0))
    for name, p in runs.items():
        m = evaluate(p)
        new = p[p.proj_cum == 0]  # counties with no earlier known project: genuinely new markets
        mn = evaluate(new) if new.y.sum() >= 5 else {}
        recs.append({"model": name, **{f"test_{k}": v for k, v in m.items()}, **{f"new_{k}": v for k, v in mn.items()}})
        preds.append(p[["fips", "origin", "y", "score"]].assign(model=name))
        print(f"{name:26s} PR-AUC {m['pr_auc']:.3f}  ROC {m['roc_auc']:.3f}  P@50 {m['precision_top50']:.2f}  recall top5% {m['recall_top5pct']:.2f}"
              + (f"  | new-market PR-AUC {mn['pr_auc']:.3f} recall top5% {mn['recall_top5pct']:.2f}" if mn else ""))
    pd.DataFrame(recs).to_json(OUT / "dc_timed_metrics.json", orient="records", indent=1)
    pd.concat(preds).to_parquet(OUT / "dc_timed_predictions.parquet")
    # forward forecast for the next year from the latest origin, all labelled years used
    cur = d[d.origin == d.origin.max()]
    fwd = {n: make(n).fit(lab[FEATS], lab.y).predict_proba(cur[FEATS])[:, 1] for n in ("xgboost", "lightgbm", "logistic")}
    out = cur[["fips", "origin", "proj_cum"]].assign(**{f"p_{k}": v for k, v in fwd.items()})
    out["p_ensemble"] = np.mean([pd.Series(v).rank(pct=True).values for v in fwd.values()], axis=0)
    out.to_parquet(ROOT / "data" / "dc_timed_forecast.parquet")
    imp = make("xgboost").fit(lab[FEATS], lab.y).feature_importances_
    (OUT / "dc_timed_importance.json").write_text(json.dumps(dict(sorted(zip(FEATS, map(float, imp.round(4))), key=lambda kv: -kv[1]))))
    print("forecast for", int(cur.origin.max()) + 1, "| top importance:", sorted(zip(FEATS, imp.round(3)), key=lambda kv: -kv[1])[:6])


if __name__ == "__main__":
    main()
