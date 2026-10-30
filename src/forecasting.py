"""Stage 1: chronological comparison of baselines, XGBoost, LightGBM, Random Forest and an ensemble.

Every (target, horizon) uses the same rows, features, splits and metrics for all models.
Splits are by *target year*, so no training label is ever later than the first validation label.
Regression target: log1p(units/jobs at t+h) - log1p(mean of t-2..t).  Boom label: value at t+h >= 1.25 x that mean.
"""
import json
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.neighbors import NearestNeighbors

from data_context import lean_asof
from geo import ROOT, centroids

OUT = ROOT / "outputs" / "results"
BOOM = 1.25
K_NEIGHBORS = 8
HORIZONS = (1, 2, 3)  # years; 6 months is not trainable with annual data
TOP_K = 100

REGIONS = {
    "Northeast": "09 23 25 33 44 50 34 36 42", "Midwest": "17 18 26 39 55 19 20 27 29 31 38 46",
    "South": "10 11 12 13 24 37 45 51 54 01 21 28 47 05 22 40 48", "West": "04 08 16 35 30 49 32 56 02 06 15 41 53",
}
STATE_REGION = {s: r for r, ss in REGIONS.items() for s in ss.split()}


@dataclass
class Target:
    key: str
    label: str
    source: str  # "qcew" or "permits"
    industry: str | None
    min_size: float  # ignore tiny counties: growth there is mostly noise
    lags: int
    train_end: int  # last target year used for training
    val_end: int  # last target year used for validation; later target years are test


TARGETS = [
    Target("all_construction", "All construction (jobs)", "qcew", "23", 20, 3, 2020, 2022),
    Target("nonres_building", "Commercial/industrial building (jobs)", "qcew", "2362", 20, 3, 2020, 2022),
    Target("infrastructure", "Infrastructure & civil (jobs)", "qcew", "237", 20, 3, 2020, 2022),
    Target("residential_permits", "Residential (permitted units)", "permits", None, 50, 5, 2012, 2017),
]


# ----------------------------------------------------------------------------- data
def _wide(df, value, index="fips"):
    return df.pivot_table(index=index, columns="year", values=value, aggfunc="sum")


def load_series(t: Target):
    """Return (levels, aux) where levels is counties x years (NaN = not observed)."""
    if t.source == "qcew":
        q = pd.read_parquet(ROOT / "data" / "qcew.parquet")
        s = q[q.industry_code == t.industry]
        aux = {"pay": _wide(s, "avg_annual_pay", "area_fips"), "estabs": _wide(s, "annual_avg_estabs", "area_fips"),
               "ctx": _wide(q[q.industry_code == "23"], "annual_avg_emplvl", "area_fips")}
        S = _wide(s, "annual_avg_emplvl", "area_fips")
        qq = pd.read_parquet(ROOT / "data" / "qcew_quarterly.parquet")
        aux["q1"] = qq[(qq.industry_code == t.industry) & (qq.qtr == 1)].pivot_table(index="fips", columns="year", values="emp")
        # 2025 annual is not published yet: estimate it from Q1-Q3 with each county's usual annual / Q1-Q3 ratio.
        # Used as a FEATURE for the forward forecast only; never as a training label (LABEL_LAST below).
        last = int(S.columns.max())
        for code, key in ((t.industry, None), ("23", "ctx")):
            est = estimate_next_annual(qq[qq.industry_code == code], last + 1)
            if est is not None:
                if key:
                    aux["ctx"][last + 1] = est.reindex(aux["ctx"].index)
                else:
                    S[last + 1] = est.reindex(S.index)
        for k in ("pay", "estabs"):  # carried forward one year (latest published values)
            aux[k][last + 1] = aux[k][last]
        aux["_label_last"] = last
        return S, aux
    p = pd.read_parquet(ROOT / "data" / "panel.parquet")
    units = _wide(p, "units")
    aux = {"mf": _wide(p, "multifamily_units"), "value": _wide(p, "value"),
           "ytd": pd.read_parquet(ROOT / "data" / "permits_ytd.parquet").pivot_table(index="fips", columns="year", values="units_ytd", aggfunc="sum")}
    aux["_label_last"] = int(units.columns.max())
    return units, aux


def estimate_next_annual(qq: pd.DataFrame, year: int) -> pd.Series | None:
    """Annual average for `year` from its first three quarters, scaled by the county's median historical ratio."""
    w = qq.pivot_table(index=["fips", "year"], columns="qtr", values="emp")
    if year not in w.index.get_level_values(1) or not {1, 2, 3}.issubset(w.columns):
        return None
    q123 = w[[1, 2, 3]].mean(axis=1)
    full = w[[1, 2, 3, 4]].mean(axis=1, skipna=False) if 4 in w.columns else None
    ratio = (full / q123).dropna()
    county_ratio = ratio.groupby(level=0).median()
    cur = q123.xs(year, level=1)
    return cur * county_ratio.reindex(cur.index).fillna(ratio.median())


def build_dataset(t: Target) -> pd.DataFrame:
    S, aux = load_series(t)
    label_last = aux.pop("_label_last")
    cent = centroids().set_index("fips")
    keep = S.index.intersection(cent.index)  # counties without a map centroid (e.g. CT planning regions) are dropped
    S = S.loc[keep]
    aux = {k: v.reindex(keep) for k, v in aux.items()}
    cent = cent.loc[keep]
    years = [int(y) for y in S.columns]
    LU = np.log1p(S)
    acs = pd.read_parquet(ROOT / "data" / "acs.parquet")
    A = {v: acs.pivot_table(index="fips", columns="vintage", values=v).reindex(keep) for v in ("pop", "med_income", "med_home_value", "housing_units", "med_re_taxes")}
    idx = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(cent[["lon", "lat"]]).kneighbors(cent[["lon", "lat"]], return_distance=False)[:, 1:]
    state = pd.Series(keep.str[:2], index=keep)
    E = pd.read_parquet(ROOT / "data" / "elections.parquet")
    static = pd.read_parquet(ROOT / "data" / "context_static.parquet").reindex(keep)  # 2023 roads/rail, 2022 land value
    hazard = pd.read_parquet(ROOT / "data" / "infra_features.parquet").reindex(keep)  # FEMA National Risk Index (static)
    projects = pd.read_parquet(ROOT / "data" / "dc_projects.parquet")  # 915 dated data center projects (data_dc_projects.py)

    rows = []
    for o in range(years[0] + t.lags - 1, years[-1] + 1):  # origin year: features use data <= o; the last year has no labels yet (forecast row)
        lu = {k: LU[o - k] for k in range(t.lags)}
        mean3 = S[[o, o - 1, o - 2]].mean(axis=1, skipna=False)
        avg3 = np.log1p(mean3)
        mom1, mom2 = lu[0] - lu[1], lu[0] - lu[2]
        f = pd.DataFrame({f"lu_{k}": lu[k] for k in range(t.lags)})
        f["avg3"] = avg3
        f["mom1"], f["mom2"] = mom1, mom2
        f["vol"] = pd.concat(lu.values(), axis=1).std(axis=1)
        # spatial
        lu0, lu2 = lu[0].values, lu[2].values
        f["nbr_lu"] = np.nanmean(lu0[idx], axis=1)
        f["nbr_mom2"] = np.nanmean(lu0[idx] - lu2[idx], axis=1)
        f["state_mom1"] = mom1.groupby(state).transform("median")
        f["lat"], f["lon"] = cent.lat, cent.lon
        # economic
        if t.source == "qcew":
            f["pay_g2"] = np.log(aux["pay"][o] / aux["pay"][o - 2])
            f["pay_lvl"] = np.log(aux["pay"][o])
            f["estab_g2"] = np.log1p(aux["estabs"][o]) - np.log1p(aux["estabs"][o - 2])
            f["ctx_lu"] = np.log1p(aux["ctx"][o])
            # nowcast: Q1 of the following year is published before the forecast is made
            f["q1_next_yoy"] = np.log1p(aux["q1"][o + 1]) - np.log1p(aux["q1"][o]) if {o, o + 1} <= set(aux["q1"].columns) else np.nan
        else:
            f["mf_share"] = aux["mf"][o] / S[o].clip(lower=1)
            f["val_per_unit"] = np.log1p(aux["value"][o] / S[o].clip(lower=1))
            # nowcast: January-August permits of the following year vs the same months a year earlier
            f["ytd_next_g"] = np.log1p(aux["ytd"][o + 1]) - np.log1p(aux["ytd"][o]) if {o, o + 1} <= set(aux["ytd"].columns) else np.nan
        v = o - 2  # ACS 5-year vintage v is released in Dec v+1, so it is public before origin o's year ends
        if v in A["pop"].columns:
            f["pop_lu"], f["inc_lu"], f["home_lu"] = np.log(A["pop"][v]), np.log(A["med_income"][v]), np.log(A["med_home_value"][v])
            if v - 3 in A["pop"].columns:
                f["pop_g3"], f["hu_g3"] = np.log(A["pop"][v] / A["pop"][v - 3]), np.log(A["housing_units"][v] / A["housing_units"][v - 3])
        lean = lean_asof(E, o).reindex(keep)  # latest presidential election held on or before o
        for c in ("county_gop_margin", "state_gop_margin", "state_red"):
            f[c] = pd.to_numeric(lean[c], errors="coerce") if c in lean else np.nan  # empty (object dtype) before 2008
        for c in ("interstate_dist_km", "rail_dist_km", "primary_km_25km", "land_value_log"):
            f[c] = static[c]
        if v in A["med_re_taxes"].columns:
            f["prop_tax_rate"] = A["med_re_taxes"][v] / A["med_home_value"][v]  # effective property tax rate, owner-occupied homes
        f["no_wage_income_tax"] = state.isin(NO_WAGE_TAX_STATES).astype(float).values
        for c in ("risk_score", "eal_log", "resl_score", "risk_hrcn", "risk_wfir", "risk_trnd", "risk_erqk"):
            f[c] = hazard[c]
        known = projects[projects.year <= o]  # announced on or before the origin year
        f["pipe_cum"] = known.groupby("fips").size().reindex(keep).fillna(0).values
        f["pipe_recent"] = known[known.year >= o - 1].groupby("fips").size().reindex(keep).fillna(0).values
        f["origin"] = o
        f["size"] = mean3
        for h in HORIZONS:
            if o + h <= label_last:  # labels only from published (not estimated) years
                fut = S[o + h]
                f[f"y{h}"] = np.log1p(fut) - avg3
                f[f"boom{h}"] = np.where(fut.isna(), np.nan, (fut >= BOOM * mean3).astype(float))
                f[f"inc{h}"] = fut - mean3  # absolute increase (jobs or units) vs the 3-year average
        rows.append(f)
    df = pd.concat(rows).rename_axis("fips").reset_index().replace([np.inf, -np.inf], np.nan)
    return df[(df["size"] >= t.min_size)]  # eligible: observed 3-yr baseline of meaningful size


def feature_groups(t: Target) -> dict[str, list[str]]:
    own = [f"lu_{k}" for k in range(t.lags)] + ["avg3", "mom1", "mom2", "vol"]
    spatial = ["nbr_lu", "nbr_mom2", "state_mom1", "lat", "lon"]
    econ = ["pay_g2", "pay_lvl", "estab_g2", "ctx_lu", "q1_next_yoy"] if t.source == "qcew" else ["mf_share", "val_per_unit", "ytd_next_g"]
    demo = ["pop_lu", "inc_lu", "home_lu", "pop_g3", "hu_g3"]  # NaN before origin 2012/2015 (ACS starts 2010)
    context = CONTEXT  # political lean (time-correct) + static transport and land value
    return {"own": own, "own+spatial": own + spatial, "own+spatial+econ": own + spatial + econ, "own+spatial+econ+demo": own + spatial + econ + demo,
            "own+spatial+econ+context": own + spatial + econ + context}


CONTEXT = ["county_gop_margin", "state_gop_margin", "state_red", "interstate_dist_km", "rail_dist_km", "primary_km_25km", "land_value_log",
           "prop_tax_rate", "no_wage_income_tax", "risk_score", "eal_log", "resl_score", "risk_hrcn", "risk_wfir", "risk_trnd", "risk_erqk",
           "pipe_cum", "pipe_recent"]
# States that do not tax wage income (hand-coded from state law; stable over 2008-2024): AK FL NV NH SD TN TX WA WY
NO_WAGE_TAX_STATES = {"02", "12", "32", "33", "46", "47", "48", "53", "56"}
TOP_FEATURES = {}  # (target, horizon) -> feature set chosen on validation for the "top" target
TOP_SHARE = 0.10  # "top" target: county is in the top 10% of absolute increases for that target year


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)) / len(y))


def precision_at_k(df, score, label, k=TOP_K):
    """Mean over target years of the share of the top-k scored counties that actually boomed."""
    d = df.assign(_s=score, _l=label)
    return float(np.mean([g.nlargest(k, "_s")._l.mean() for _, g in d.groupby("origin")]))


def reg_metrics(y, p):
    return {"mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}


def clf_metrics(df, y, p, raw=None):
    m = {"pr_auc": float(average_precision_score(y, p)), "base_rate": float(y.mean()), "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
         "ece": ece(y, p), "p_at_k": precision_at_k(df, p, y)}
    thr = np.sort(p)[::-1][min(TOP_K, len(p)) - 1]
    sel = p >= thr
    m["precision_topk"], m["recall_topk"] = float(y[sel].mean()), float(y[sel].sum() / max(y.sum(), 1))
    if raw is not None:
        m["ece_raw"] = ece(y, raw)
    return m


# ----------------------------------------------------------------------------- models
def make_model(name, task, seed=0):
    if task == "reg":
        return {
            "xgboost": lambda: xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, n_jobs=-1, random_state=seed),
            "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, n_jobs=-1, verbose=-1, random_state=seed),
            "random_forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=seed),
        }[name]()
    return {
        "xgboost": lambda: xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, n_jobs=-1, random_state=seed),
        "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, n_jobs=-1, verbose=-1, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=seed),
    }[name]()


def fit_predict(name, task, X_tr, y_tr, X_ev: dict):
    m = make_model(name, task)
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    out, t0 = {}, time.perf_counter()
    n = 0
    for k, X in X_ev.items():
        out[k] = m.predict(X) if task == "reg" else m.predict_proba(X)[:, 1]
        n += len(X)
    return out, fit_s, (time.perf_counter() - t0) / n * 1000 * 1000, m  # ms per 1k rows


def nnls_weights(P: np.ndarray, y: np.ndarray):
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])


# ----------------------------------------------------------------------------- experiment
def run(targets=TARGETS, ml=("xgboost", "lightgbm", "random_forest")):
    OUT.mkdir(parents=True, exist_ok=True)
    records, preds, preds_top = [], [], []
    for t in targets:
        data = build_dataset(t)
        groups = feature_groups(t)
        for h in HORIZONS:
            d = data.dropna(subset=[f"y{h}", f"boom{h}"]).copy()
            d["target_year"] = d.origin + h
            split = np.where(d.target_year <= t.train_end, "train", np.where(d.target_year <= t.val_end, "val", "test"))
            d["split"] = split
            tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
            if min(len(tr), len(va), len(te)) < 200:
                print(f"skip {t.key} h={h}: train/val/test = {len(tr)}/{len(va)}/{len(te)}")
                continue
            print(f"{t.key} h={h}y  train/val/test = {len(tr)}/{len(va)}/{len(te)}  test boom rate {te[f'boom{h}'].mean():.2f}")
            y, b = f"y{h}", f"boom{h}"
            base = dict(target=t.key, horizon_years=h, n_train=len(tr), n_val=len(va), n_test=len(te),
                        train_years=f"{tr.origin.min() + h}-{t.train_end}", val_years=f"{t.train_end + 1}-{t.val_end}",
                        test_years=f"{t.val_end + 1}-{te.target_year.max()}")

            def log_preds(model, split_name, frame, reg=None, prob=None):
                preds.append(pd.DataFrame({"target": t.key, "horizon_years": h, "model": model, "split": split_name, "fips": frame.fips.values,
                                           "origin": frame.origin.values, "y": frame[y].values, "boom": frame[b].values,
                                           "pred": reg if reg is not None else np.nan, "prob": prob if prob is not None else np.nan}))

            # ---- baselines (no fitting): persistence = hold the latest level; mean3 = hold the 3-year average
            gap = lambda f: (f["lu_0"] - f["avg3"]).values
            reg_pred, clf_pred, timing = {}, {}, {}
            for nm, fn in {"baseline_mean3": lambda f: np.zeros(len(f)), "baseline_persistence": gap}.items():
                reg_pred[nm] = {"val": fn(va), "test": fn(te)}
                timing[nm] = (0.0, 0.0)
            for nm in ("baseline_persistence",):  # score -> probability via isotonic fit on validation
                iso = IsotonicRegression(out_of_bounds="clip").fit(gap(va), va[b].values)
                clf_pred[nm] = {"val": iso.predict(gap(va)), "test": iso.predict(gap(te)), "raw_test": gap(te)}
            prior = tr[b].mean()
            clf_pred["baseline_base_rate"] = {"val": np.full(len(va), prior), "test": np.full(len(te), prior), "raw_test": np.full(len(te), prior)}

            # ---- ML models, all feature groups (ablation); the full set is the headline model
            full = "own+spatial+econ"
            for fs, cols in groups.items():
                for name in ml:
                    (r, fit_s, inf_ms, _m) = fit_predict(name, "reg", tr[cols], tr[y], {"val": va[cols], "test": te[cols]})
                    (c, cfit_s, cinf_ms, _c) = fit_predict(name, "clf", tr[cols], tr[b], {"val": va[cols], "test": te[cols]})
                    if fs == full:
                        reg_pred[name], timing[name] = r, (fit_s, inf_ms)
                        iso = IsotonicRegression(out_of_bounds="clip").fit(c["val"], va[b].values)
                        clf_pred[name] = {"val": iso.predict(c["val"]), "test": iso.predict(c["test"]), "raw_test": c["test"], "fit_s": cfit_s, "inf_ms": cinf_ms}
                    elif name == "xgboost" or fs == "own":  # ablation rows for every family on the smaller sets
                        isoa = IsotonicRegression(out_of_bounds="clip").fit(c["val"], va[b].values)
                        records.append({**base, "task": "ablation", "model": name, "features": fs, **{f"reg_{k}": v for k, v in reg_metrics(te[y].values, r["test"]).items()},
                                        **{f"clf_{k}": v for k, v in clf_metrics(te, te[b].values, isoa.predict(c["test"])).items()}})
                    if fs == full:  # ablation row for the full set too
                        records.append({**base, "task": "ablation", "model": name, "features": fs, **{f"reg_{k}": v for k, v in reg_metrics(te[y].values, r["test"]).items()},
                                        **{f"clf_{k}": v for k, v in clf_metrics(te, te[b].values, clf_pred[name]["test"]).items()}})

            # ---- ensemble: weights learned on validation only
            members = list(ml)
            w_reg = nnls_weights(np.column_stack([reg_pred[m]["val"] for m in members]), va[y].values)
            w_clf = nnls_weights(np.column_stack([clf_pred[m]["val"] for m in members]), va[b].values)
            reg_pred["ensemble"] = {s: np.column_stack([reg_pred[m][s] for m in members]) @ w_reg for s in ("val", "test")}
            clf_pred["ensemble"] = {s: np.column_stack([clf_pred[m][s] for m in members]) @ w_clf for s in ("val", "test")}
            clf_pred["ensemble"]["raw_test"] = clf_pred["ensemble"]["test"]
            timing["ensemble"] = (sum(timing[m][0] for m in members), sum(timing[m][1] for m in members))
            ens_w = {"reg": dict(zip(members, w_reg.round(3))), "clf": dict(zip(members, w_clf.round(3)))}

            # ---- records
            for name in reg_pred:
                rv, rt = reg_metrics(va[y].values, reg_pred[name]["val"]), reg_metrics(te[y].values, reg_pred[name]["test"])
                fit_s, inf_ms = timing[name]
                records.append({**base, "task": "reg", "model": name, "features": full if name in ml or name == "ensemble" else "-",
                                **{f"val_{k}": v for k, v in rv.items()}, **{f"test_{k}": v for k, v in rt.items()},
                                "test_p_at_k": precision_at_k(te, reg_pred[name]["test"], te[b].values), "test_base_rate": float(te[b].mean()),
                                "train_s": fit_s, "infer_ms_per_1k": inf_ms, "ensemble_weights": ens_w["reg"] if name == "ensemble" else None})
                log_preds(name, "val", va, reg=reg_pred[name]["val"]); log_preds(name, "test", te, reg=reg_pred[name]["test"])
            for name in clf_pred:
                cm_v = clf_metrics(va, va[b].values, clf_pred[name]["val"])
                cm_t = clf_metrics(te, te[b].values, clf_pred[name]["test"], clf_pred[name]["raw_test"])
                records.append({**base, "task": "clf", "model": name, "features": full if name in ml or name == "ensemble" else "-",
                                **{f"val_{k}": v for k, v in cm_v.items()}, **{f"test_{k}": v for k, v in cm_t.items()},
                                "train_s": clf_pred[name].get("fit_s", 0.0), "infer_ms_per_1k": clf_pred[name].get("inf_ms", 0.0),
                                "ensemble_weights": ens_w["clf"] if name == "ensemble" else None})
                log_preds(name, "val", va, prob=clf_pred[name]["val"]); log_preds(name, "test", te, prob=clf_pred[name]["test"])

            # ---- "top" target: is this county among the TOP_SHARE largest absolute increases in its target year?
            tp = f"top{h}"
            for fr in (tr, va, te):
                fr[tp] = fr.groupby("origin")[f"inc{h}"].transform(lambda s: (s >= s.quantile(1 - TOP_SHARE)).astype(float))
            top_pred = {}
            score_fns = {"baseline_size": lambda f: f["size"].values,  # big counties add the most
                         "baseline_recent_increase": lambda f: (np.expm1(f["lu_0"]) - f["size"]).values}
            for nm, fn in score_fns.items():
                iso = IsotonicRegression(out_of_bounds="clip").fit(fn(va), va[tp].values)
                top_pred[nm] = {"val": iso.predict(fn(va)), "test": iso.predict(fn(te)), "raw_test": fn(te)}
            # pick the feature set on VALIDATION (XGBoost), then train every model family on it
            val_score = {}
            for fs in (full, "own+spatial+econ+context"):
                cols = groups[fs]
                (c, _, _, _) = fit_predict("xgboost", "clf", tr[cols], tr[tp], {"val": va[cols], "test": te[cols]})
                val_score[fs] = average_precision_score(va[tp].values, c["val"])
                records.append({**base, "task": "ablation_top", "model": "xgboost", "features": fs, "val_pr_auc": val_score[fs],
                                **{f"top_{k}": v for k, v in clf_metrics(te, te[tp].values, c["test"]).items()}})
            chosen = max(val_score, key=val_score.get)
            TOP_FEATURES[(t.key, h)] = chosen
            for name in ml:
                cols = groups[chosen]
                (c, cfit_s, cinf_ms, _c) = fit_predict(name, "clf", tr[cols], tr[tp], {"val": va[cols], "test": te[cols]})
                iso = IsotonicRegression(out_of_bounds="clip").fit(c["val"], va[tp].values)
                top_pred[name] = {"val": iso.predict(c["val"]), "test": iso.predict(c["test"]), "raw_test": c["test"], "fit_s": cfit_s, "inf_ms": cinf_ms}
            w_top = nnls_weights(np.column_stack([top_pred[m]["val"] for m in members]), va[tp].values)
            top_pred["ensemble"] = {s: np.column_stack([top_pred[m][s] for m in members]) @ w_top for s in ("val", "test")}
            top_pred["ensemble"]["raw_test"] = top_pred["ensemble"]["test"]
            for name, pr in top_pred.items():
                records.append({**base, "task": "top", "model": name, "features": chosen if name in ml or name == "ensemble" else "-",
                                **{f"val_{k}": v for k, v in clf_metrics(va, va[tp].values, pr["val"]).items()},
                                **{f"test_{k}": v for k, v in clf_metrics(te, te[tp].values, pr["test"], pr["raw_test"]).items()},
                                "train_s": pr.get("fit_s", 0.0), "ensemble_weights": dict(zip(members, w_top.round(3))) if name == "ensemble" else None})
                for sp, fr in (("val", va), ("test", te)):
                    preds_top.append(pd.DataFrame({"target": t.key, "horizon_years": h, "model": name, "split": sp, "fips": fr.fips.values, "origin": fr.origin.values,
                                                   "top": fr[tp].values, "inc": fr[f"inc{h}"].values, "prob": pr[sp]}))

            # ---- geographic generalization (h=1 only): train on other Census regions, test on the held-out one
            if h == 1:
                cols = groups[full]
                region = d.fips.str[:2].map(STATE_REGION)
                for reg_name in REGIONS:
                    tr_o = d[(d.split == "train") & (region != reg_name)]
                    te_r = d[(d.split == "test") & (region == reg_name)]
                    if len(te_r) < 50:
                        continue
                    for name in ml:
                        out, _, _, _ = fit_predict(name, "reg", tr_o[cols], tr_o[y], {"test": te_r[cols]})
                        std = te[(region.loc[te.index] == reg_name)]
                        std_pred = pd.Series(reg_pred[name]["test"], index=te.index).loc[std.index].values
                        records.append({**base, "task": "geo_holdout", "model": name, "region": reg_name, "n_test_region": len(te_r),
                                        "mae_heldout_region": reg_metrics(te_r[y].values, out["test"])["mae"],
                                        "mae_standard": reg_metrics(std[y].values, std_pred)["mae"],
                                        "mae_persistence": reg_metrics(te_r[y].values, (te_r.lu_0 - te_r.avg3).values)["mae"]})
    res = pd.DataFrame(records)
    res.to_json(OUT / "metrics.json", orient="records", indent=1)
    res.to_csv(OUT / "metrics.csv", index=False)
    pd.concat(preds).to_parquet(OUT / "predictions.parquet")
    pd.concat(preds_top).to_parquet(OUT / "predictions_top.parquet")
    # best model per target by VALIDATION score (never test)
    best = {}
    for (tg, h, task), g in res[res.task.isin(["reg", "clf", "top"])].groupby(["target", "horizon_years", "task"]):
        row = g.loc[g.val_mae.idxmin()] if task == "reg" else g.loc[g.val_pr_auc.idxmax()]
        best[f"{tg}|{h}|{task}"] = row.model
    (OUT / "best_models.json").write_text(json.dumps(best, indent=1))
    (OUT / "top_features.json").write_text(json.dumps({f"{k}|{h}": v for (k, h), v in TOP_FEATURES.items()}, indent=1))
    return res, best


if __name__ == "__main__":
    res, best = run()
    print("done;", len(res), "records")
