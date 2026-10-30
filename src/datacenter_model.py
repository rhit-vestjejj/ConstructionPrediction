"""Data center siting propensity by county.

NOT a timed forecast: there are no dated historical construction starts, so the label is "county hosts >= 1 data center
today" (OpenStreetMap). Scores say how much a county's profile resembles places that host data centers. Counties with a
high score and no known site are candidate emerging markets.

Evaluation
  1. Spatial cross-validation: leave one Census region out (train on 3 regions, score the 4th).
  2. Emerging-location test: counties of Epoch AI's dated frontier campuses are removed from training and from every
     neighbour-based feature, then ranked against all other non-hosting counties.
"""
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_context import lean_asof
from forecasting import NO_WAGE_TAX_STATES, REGIONS, STATE_REGION
from geo import ROOT, centroids, load_geojson

K = 8
OUT = ROOT / "outputs" / "results"
GROUPS = {
    "economic": ["jobs518", "estabs518", "pay518", "jobs493", "jobs2362", "jobs23", "pay23", "g23", "g2362", "units23", "g_units"],
    "+geography&power": ["lat", "lon", "log_area", "elec_price", "nbr_jobs23", "nbr_units"],
    "+demographics": ["pop_lu", "inc_lu", "home_lu", "hu_lu"],
    "+power&grid": ["mw_total", "mw_gas", "mw_coal", "mw_nuc", "mw_hydro", "mw_windsolar", "mw_storage", "n_plants", "max_grid_kv", "mw_50km", "mw_150km",
                    "ind_price", "ind_price_min", "n_utilities", "kv230_dist_km", "kv345_dist_km", "line_km_50km"],
    "+climate&hazard": ["cdd", "hdd", "tmax_summer", "risk_score", "eal_log", "resl_score", "risk_hrcn", "risk_trnd", "risk_erqk", "risk_wfir", "risk_cfld", "risk_hwav", "risk_drgt"],
    "+context&tax": ["county_gop_margin", "state_gop_margin", "state_red", "interstate_dist_km", "rail_dist_km", "primary_km_25km", "land_value_log",
                     "prop_tax_rate", "no_wage_income_tax"],
    "+existing cluster": ["nbr_dc", "dist_dc_km"],
}
ML = ["logistic", "xgboost", "lightgbm", "random_forest"]


def county_area_km2() -> pd.Series:
    out = {}
    for f in load_geojson()["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        a = 0.0
        for poly in polys:
            r = np.array(poly[0])
            x, y = np.radians(r[:, 0]) * np.cos(np.radians(r[:, 1].mean())) * 6371, np.radians(r[:, 1]) * 6371
            a += abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
        out[f["id"]] = a
    return pd.Series(out)


def haversine(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def base_table() -> pd.DataFrame:
    q = pd.read_parquet(ROOT / "data" / "qcew.parquet")
    p = pd.read_parquet(ROOT / "data" / "panel.parquet")
    c = centroids().set_index("fips")

    def at(code, year, col="annual_avg_emplvl"):
        s = q[(q.industry_code == code) & (q.year == year)].set_index("area_fips")[col]
        return s.groupby(level=0).sum(min_count=1)

    df = pd.DataFrame(index=c.index)
    df["jobs518"], df["estabs518"], df["pay518"] = np.log1p(at("518", 2023)), np.log1p(at("518", 2023, "annual_avg_estabs")), np.log(at("518", 2023, "avg_annual_pay"))
    df["jobs493"], df["jobs2362"], df["jobs23"], df["pay23"] = np.log1p(at("493", 2023)), np.log1p(at("2362", 2023)), np.log1p(at("23", 2023)), np.log(at("23", 2023, "avg_annual_pay"))
    df["g23"] = np.log1p(at("23", 2023)) - np.log1p(at("23", 2019))
    df["g2362"] = np.log1p(at("2362", 2023)) - np.log1p(at("2362", 2019))
    u = p.pivot_table(index="fips", columns="year", values="units", aggfunc="sum")
    df["units23"], df["g_units"] = np.log1p(u[2023]), np.log1p(u[2023]) - np.log1p(u[2019])
    df["lat"], df["lon"], df["log_area"] = c.lat, c.lon, np.log(county_area_km2().reindex(df.index))
    e = pd.read_excel(ROOT / "data" / "raw" / "eia_avgprice.xlsx", header=1)
    e = e[(e.Year == 2020) & (e["Industry Sector Category"] == "Total Electric Industry")].set_index("State").Industrial  # 2020: latest in the free file
    ABBR = dict(zip("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split(),
                    "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split()))
    df["elec_price"] = df.index.str[:2].map({v: e.get(k) for k, v in ABBR.items()}).astype(float)
    acs = pd.read_parquet(ROOT / "data" / "acs.parquet")
    a = acs[acs.vintage == 2023].set_index("fips").reindex(df.index)
    df["pop_lu"], df["inc_lu"], df["home_lu"], df["hu_lu"] = np.log(a["pop"]), np.log(a.med_income), np.log(a.med_home_value), np.log(a.housing_units)
    df = df.join(pd.read_parquet(ROOT / "data" / "infra_features.parquet"))  # see data_infra.py
    df = df.join(pd.read_parquet(ROOT / "data" / "context_static.parquet"))  # roads, rail, land value (data_context.py)
    df = df.join(lean_asof(pd.read_parquet(ROOT / "data" / "elections.parquet"), 2023))  # 2020 election for a 2023 snapshot
    a22 = acs[acs.vintage == 2022].set_index("fips").reindex(df.index)
    df["prop_tax_rate"] = a22.med_re_taxes / a22.med_home_value
    df["no_wage_income_tax"] = df.index.str[:2].isin(NO_WAGE_TAX_STATES).astype(float)
    df["region"] = df.index.str[:2].map(STATE_REGION)
    return df[df.region.notna()]


def neighbour_features(df: pd.DataFrame, n_dc: pd.Series) -> pd.DataFrame:
    """Neighbour size and existing-cluster features; n_dc may have counties zeroed out to prevent leakage."""
    xy = df[["lon", "lat"]].values
    idx = NearestNeighbors(n_neighbors=K + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    out = df.copy()
    out["nbr_jobs23"], out["nbr_units"] = np.nanmean(df.jobs23.values[idx], axis=1), np.nanmean(df.units23.values[idx], axis=1)
    d = n_dc.reindex(df.index).fillna(0).values
    out["nbr_dc"] = d[idx].sum(axis=1)
    hosts = np.where(d > 0)[0]
    dist = np.array([haversine(xy[i, 0], xy[i, 1], xy[hosts, 0], xy[hosts, 1]) for i in range(len(df))])  # n x hosts
    dist[hosts, np.arange(len(hosts))] = np.inf  # exclude self
    out["dist_dc_km"] = dist.min(axis=1)
    return out


def make(name):
    return {
        "logistic": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
        "xgboost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, n_jobs=-1, random_state=0),
        "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=12, learning_rate=0.04, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=8, n_jobs=-1, verbose=-1, random_state=0),
        "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0)),
    }[name]()


def metrics(y, s):
    k = max(1, int(0.1 * len(y)))
    top = np.argsort(-s)[:k]
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)), "base_rate": float(y.mean()), "recall_top10pct": float(y[top].sum() / y.sum())}


def region_cv(X, y, region, name):
    oof = np.zeros(len(y))
    t0 = time.perf_counter()
    for r in REGIONS:
        te = (region == r).values
        m = make(name).fit(X[~te], y[~te])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, time.perf_counter() - t0


def main():
    df = base_table()
    dc = pd.read_parquet(ROOT / "data" / "datacenters.parquet")
    n_dc = dc.fips.value_counts().reindex(df.index).fillna(0)
    ep = pd.read_parquet(ROOT / "data" / "epoch_sites.parquet").dropna(subset=["fips"])
    ep = ep[ep.fips.isin(df.index)]
    y = (n_dc > 0).astype(int).values
    full = neighbour_features(df, n_dc)
    region = df.region
    print(f"{len(df)} counties; {y.sum()} host a data center (OSM); {ep.fips.nunique()} counties host Epoch frontier sites")
    res = {"n_counties": len(df), "n_hosts": int(y.sum()), "epoch_counties": int(ep.fips.nunique()), "cv": [], "emerging": [], "importance": {}}

    # ---- 1. spatial CV across feature groups and models
    cols_by_group = {}
    acc = []
    for g in GROUPS:
        acc += GROUPS[g]
        cols_by_group[g] = list(acc)
    oof_store = {}
    for g, cols in cols_by_group.items():
        X = full[cols].values
        for name in ML:
            oof, secs = region_cv(X, y, region, name)
            m = metrics(y, oof)
            res["cv"].append({"features": g, "model": name, "train_s_total": round(secs, 2), **{k: round(v, 4) for k, v in m.items()}})
            oof_store[(g, name)] = oof
            print(f"CV {g:20s} {name:14s} AUC {m['roc_auc']:.3f} PR {m['pr_auc']:.3f} rec@10% {m['recall_top10pct']:.2f} (base {m['base_rate']:.3f})")
    for base in ("jobs518", "jobs23", "nbr_dc"):
        m = metrics(y, np.nan_to_num(full[base].values, nan=-1))
        res["cv"].append({"features": "single-feature baseline", "model": base, **{k: round(v, 4) for k, v in m.items()}})
        print(f"baseline {base:10s} AUC {m['roc_auc']:.3f} PR {m['pr_auc']:.3f}")

    # ---- 2. emerging-location test: hold out Epoch counties everywhere (labels AND neighbour/cluster features)
    E = set(ep.fips)
    held = df.index.isin(E)
    n_dc_masked = n_dc.copy()
    n_dc_masked[list(E)] = 0
    masked = neighbour_features(df, n_dc_masked)
    since = {f: ep[ep.fips == f].first_dated_milestone.min() for f in E}
    late = df.index.isin([f for f, d in since.items() if pd.notna(d) and d.year >= 2022])
    neg = (y == 0) & ~held
    for g in GROUPS:
        cols = cols_by_group[g]
        for name in ML:
            m = make(name).fit(masked.loc[~held, cols].values, y[~held])
            s = m.predict_proba(masked[cols].values)[:, 1]
            for label, pos in (("all Epoch counties", held), ("first milestone 2022+", late)):
                keep = pos | neg
                mm = metrics(pos[keep].astype(int), s[keep])
                pct = float(np.median(100 * (s[neg][None, :] < s[pos][:, None]).mean(axis=1)))
                res["emerging"].append({"features": g, "model": name, "positives": label, "n_positive": int(pos.sum()), "median_percentile": round(pct, 1), **{k: round(v, 4) for k, v in mm.items()}})
            print(f"EMERGING {g:20s} {name:14s} AUC {res['emerging'][-2]['roc_auc']:.3f}  recall@10% {res['emerging'][-2]['recall_top10pct']:.2f}  median pct {res['emerging'][-2]['median_percentile']}")
    for base in ("jobs518", "jobs23", "nbr_jobs23"):
        for label, pos in (("all Epoch counties", held), ("first milestone 2022+", late)):
            keep = pos | neg
            mm = metrics(pos[keep].astype(int), np.nan_to_num(masked[base].values[keep], nan=-1))
            res["emerging"].append({"features": "single-feature baseline", "model": base, "positives": label, "n_positive": int(pos.sum()), **{k: round(v, 4) for k, v in mm.items()}})

    # ---- 3. final per-county scores: out-of-fold (region CV) with the full feature set, ensemble of tree/linear models
    g = "+existing cluster"
    oofs = np.column_stack([oof_store[(g, m)] for m in ML])
    ens = oofs.mean(axis=1)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(ens, y)
    m_ens = metrics(y, ens)
    res["cv"].append({"features": g, "model": "ensemble", **{k: round(v, 4) for k, v in m_ens.items()}})
    xm = make("xgboost").fit(full[cols_by_group[g]].values, y)
    res["importance"] = dict(sorted(zip(cols_by_group[g], map(float, xm.feature_importances_.round(4))), key=lambda kv: -kv[1]))
    contrib = xm.get_booster().predict(xgb.DMatrix(full[cols_by_group[g]].values), pred_contribs=True)[:, :-1]
    top = np.argsort(-np.abs(contrib), axis=1)[:, :3]
    cols = cols_by_group[g]
    out = pd.DataFrame({"fips": df.index, "n_dc_osm": n_dc.values.astype(int), "score_ensemble": ens, "likelihood": iso.predict(ens)})
    for i, name in enumerate(ML):
        out[f"score_{name}"] = oofs[:, i]
    out["percentile"] = 100 * out.score_ensemble.rank(pct=True)
    out["epoch_sites"] = out.fips.map(ep.groupby("fips").size()).fillna(0).astype(int).values
    out["drivers"] = [json.dumps([[cols[j], round(float(contrib[i, j]), 2)] for j in top[i]]) for i in range(len(df))]
    out["white_space"] = ((out.n_dc_osm == 0) & (out.epoch_sites == 0) & (out.percentile >= 95)).values
    out.to_parquet(ROOT / "data" / "datacenter_scores.parquet")
    (OUT / "datacenter_metrics.json").write_text(json.dumps(res, indent=1))
    print("ensemble CV", {k: round(v, 3) for k, v in m_ens.items()}, "| white-space counties:", int(out.white_space.sum()))
    names = pd.read_parquet(ROOT / "data" / "panel.parquet").drop_duplicates("fips", keep="last").set_index("fips").name
    ws = out[out.white_space].sort_values("score_ensemble", ascending=False).head(12)
    print(ws.assign(name=ws.fips.map(names))[["fips", "name", "percentile"]].to_string(index=False))


if __name__ == "__main__":
    main()
