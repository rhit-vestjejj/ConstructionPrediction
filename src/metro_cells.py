"""Metro-level, cell-level forecast: will an H3 cell get a new >= $1M non-residential permit next year?

Only for metros whose portals publish dated, geolocated permits (Fort Worth, Charlotte/Mecklenburg, Columbus).
Cell universe at origin t = cells within 2 rings of any cell that had a permit in or before t (no look-ahead).
Split by target year: train <= 2020, validation 2021-2022, test 2023-2024 (2025 is incomplete in Mecklenburg).
"""
import json
import time

import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from geo import ROOT

RAW = ROOT / "data" / "raw" / "permits"
OUT = ROOT / "outputs" / "results"
RES = 7  # ~5 km2 cells (edge ~1.2 km); 6 would be ~36 km2
START = {"fortworth": 2016, "mecklenburg": 2010, "columbus": 2012}
LAST_COMPLETE = {"fortworth": 2025, "mecklenburg": 2024, "columbus": 2025}
TRAIN_END, VAL_END, TEST_END = 2020, 2022, 2024
METRO_ID = {"fortworth": 0, "mecklenburg": 1, "columbus": 2}
FEATS = ["n0", "n1", "n2", "n_cum", "yrs_since", "val0", "indwh3", "ring1_n0", "ring1_n3", "ring2_n3", "dist_center_km", "metro"]
SITE = ["interstate_dist_km", "primary_dist_km", "rail_dist_km", "primary_km_25km", "land_value_log", "risk_score", "risk_cfld", "prop_tax_rate"]


def events() -> pd.DataFrame:
    fw = pd.read_parquet(RAW / "fortworth_geocoded.parquet")
    fw = fw[(fw.Use_Type != "Residential") & fw.Latitude.notna()]
    fw = pd.DataFrame({"metro": "fortworth", "date": pd.to_datetime(fw.File_Date, unit="ms"), "value": fw.JobValue, "lat": fw.Latitude, "lon": fw.Longitude,
                       "cls": np.where(fw.Use_Type == "Industrial", "industrial", np.where((fw.Use_Type == "Storage") & fw.Specific_Use.isin(["Warehouse", "Self Storage or Mini Warehouse"]), "warehouse", "other"))})
    mk = pd.read_parquet(RAW / "mecklenburg.parquet")
    mk = mk[~mk.usdcdesc.str.strip().str.startswith("329") & mk._lat.notna()]
    mk = pd.DataFrame({"metro": "mecklenburg", "date": pd.to_datetime(mk.issuedate, unit="ms"), "value": mk.bldgcost, "lat": mk._lat, "lon": mk._lon,
                       "cls": np.where(mk.usdcdesc.str.strip().str.startswith("320"), "industrial", np.where(mk.occupancy.astype(str).str.strip().str.startswith("S"), "warehouse", "other"))})
    co = pd.read_parquet(RAW / "columbus.parquet")
    co = co[~co.VALUE_DESC.isin(["Five or More Family Buildings", "Structures Other Than Buildings"]) & ~co.VALUE_DESC.str.startswith("Additions") & co._lat.notna()]
    co = pd.DataFrame({"metro": "columbus", "date": pd.to_datetime(co.ISSUED_DT, unit="ms"), "value": co.G3_VALUE_TTL, "lat": co._lat, "lon": co._lon,
                       "cls": np.where(co.VALUE_DESC == "Industrial Buildings", "industrial", "other")})
    e = pd.concat([fw, mk, co], ignore_index=True)
    e["year"] = e.date.dt.year
    e["cell"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(e.lat, e.lon)]
    return e[e.year >= e.metro.map(START)]


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def build(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metro, g in e.groupby("metro"):
        cnt = g.groupby(["cell", "year"]).size().unstack(fill_value=0)
        indwh = g[g.cls != "other"].groupby(["cell", "year"]).size().unstack(fill_value=0).reindex(cnt.index, fill_value=0)
        val = g.groupby(["cell", "year"]).value.sum().unstack(fill_value=0).reindex(cnt.index, fill_value=0)
        years = list(range(START[metro], LAST_COMPLETE[metro] + 1))
        for y in years:
            for d in (cnt, indwh, val):
                if y not in d.columns:
                    d[y] = 0
        disk1, disk2 = {}, {}
        def ring(c, k, store):
            if (c, k) not in store:
                store[(c, k)] = [x for x in h3.grid_disk(c, k) if x != c] if k == 1 else [x for x in h3.grid_disk(c, 2) if x not in h3.grid_disk(c, 1)]
            return store[(c, k)]
        for t in years[2:]:  # need t-2
            active = cnt.index[(cnt[[y for y in years if y <= t]].sum(axis=1) > 0)]
            universe = sorted({x for c in active for x in h3.grid_disk(c, 2)})
            idx = pd.Index(universe)
            C = cnt.reindex(idx, fill_value=0)
            I = indwh.reindex(idx, fill_value=0)
            V = val.reindex(idx, fill_value=0)
            past = [y for y in years if y <= t]
            n0, n1, n2 = C[t], C[t - 1], C[t - 2]
            cum = C[past].sum(axis=1)
            last = C[past].apply(lambda r: max([y for y in past if r[y] > 0], default=None), axis=1)
            cen = g[g.year <= t]
            clat, clon = cen.lat.median(), cen.lon.median()
            ll = np.array([h3.cell_to_latlng(c) for c in idx])
            lookup3 = (C[t] + C[t - 1] + C[t - 2]).to_dict()
            lookup0 = C[t].to_dict()
            f = pd.DataFrame({"cell": idx, "n0": n0.values, "n1": n1.values, "n2": n2.values, "n_cum": cum.values,
                              "yrs_since": (t - last).fillna(99).values, "val0": np.log1p(V[t]).values,
                              "indwh3": (I[t] + I[t - 1] + I[t - 2]).values,
                              "ring1_n0": [sum(lookup0.get(x, 0) for x in ring(c, 1, disk1)) for c in idx],
                              "ring1_n3": [sum(lookup3.get(x, 0) for x in ring(c, 1, disk1)) for c in idx],
                              "ring2_n3": [sum(lookup3.get(x, 0) for x in ring(c, 2, disk2)) for c in idx],
                              "dist_center_km": haversine(ll[:, 0], ll[:, 1], clat, clon), "metro": METRO_ID[metro],
                              "lat": ll[:, 0], "lon": ll[:, 1]})
            f["metro_name"], f["origin"] = metro, t
            for h in (1, 2):
                if t + h <= LAST_COMPLETE[metro]:
                    f[f"y{h}"] = (cnt.reindex(idx, fill_value=0)[t + h] > 0).astype(int).values
                    f[f"yiw{h}"] = (indwh.reindex(idx, fill_value=0)[t + h] > 0).astype(int).values
            rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    return out.merge(site_features(out.cell.unique()), on="cell", how="left")


def site_features(cells) -> pd.DataFrame:
    """Per-cell access (cell centre) and county-level land value, hazard and property tax. Static, 2022-2023 vintages."""
    from data_context import transport
    from geo import assign_counties
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    f = transport(ll[:, 0], ll[:, 1])
    f["cell"] = cells
    f["fips"] = assign_counties(ll[:, 1], ll[:, 0])
    ctx = pd.read_parquet(ROOT / "data" / "context_static.parquet")[["land_value_log"]].join(pd.read_parquet(ROOT / "data" / "infra_features.parquet")[["risk_score", "risk_cfld"]])
    acs = pd.read_parquet(ROOT / "data" / "acs.parquet")
    a = acs[acs.vintage == 2022].set_index("fips")
    ctx["prop_tax_rate"] = a.med_re_taxes / a.med_home_value
    return f.merge(ctx, left_on="fips", right_index=True, how="left").drop(columns="fips")


def ece(y, p, bins=10):
    chunks = np.array_split(np.argsort(p), bins)
    return float(sum(len(c) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)) / len(y))


def topk(df, score, y, frac=0.05):
    """Per metro-origin: share of the top 5% scored cells that were positive, and share of all positives found."""
    d = df.assign(_s=score, _y=y)
    prec, rec = [], []
    for _, g in d.groupby(["metro_name", "origin"]):
        k = max(1, int(frac * len(g)))
        top = g.nlargest(k, "_s")
        prec.append(top._y.mean())
        rec.append(top._y.sum() / max(g._y.sum(), 1))
    return float(np.mean(prec)), float(np.mean(rec))


def metrics(df, y, p, raw=None):
    pk, rk = topk(df, p, y)
    m = {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)), "base_rate": float(y.mean()), "precision_top5pct": pk, "recall_top5pct": rk}
    if raw is not None:
        m["ece_raw"], m["ece"] = ece(y, raw), ece(y, p)
    return m


def make(name):
    return {"xgboost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, n_jobs=-1, random_state=0),
            "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.04, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=10, n_jobs=-1, verbose=-1, random_state=0),
            "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=0))}[name]()


def baselines(d):
    return {"baseline_recent_history": (d.n0 * 1.0 + d.n1 * 0.5 + d.n2 * 0.25 + 0.05 * d.ring1_n3).values,
            "baseline_last_year": (d.n0 > 0).astype(float).values + 0.01 * d.n_cum.values,
            "baseline_neighbors": (d.ring1_n3 + 0.5 * d.ring2_n3).values}


def run():
    t0 = time.time()
    e = events()
    print("events", len(e), e.groupby("metro").size().to_dict(), "| classes", e.cls.value_counts().to_dict())
    df = build(e)
    print(f"cell-years {len(df)} built in {time.time() - t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / "data" / "metro_cells.parquet")
    recs, preds = [], []
    for h in (1, 2):
        d = df.dropna(subset=[f"y{h}"]).copy()
        y = d[f"y{h}"].values
        ty = d.origin + h
        tr, va, te = (d[ty <= TRAIN_END], d[(ty > TRAIN_END) & (ty <= VAL_END)], d[(ty > VAL_END) & (ty <= TEST_END)])
        print(f"\nh={h}y  train/val/test = {len(tr)}/{len(va)}/{len(te)}  positive rate {tr[f'y{h}'].mean():.3f}/{va[f'y{h}'].mean():.3f}/{te[f'y{h}'].mean():.3f}")
        scores = {"val": {}, "test": {}}
        fit = {}
        for nm, fn in baselines(va).items():
            scores["val"][nm] = fn
        for nm, fn in baselines(te).items():
            scores["test"][nm] = fn
        # calibrate baseline scores to probabilities (isotonic on validation)
        raw_test = dict(scores["test"])
        for nm in list(scores["val"]):
            iso = IsotonicRegression(out_of_bounds="clip").fit(scores["val"][nm], va[f"y{h}"].values)
            scores["val"][nm], scores["test"][nm] = iso.predict(scores["val"][nm]), iso.predict(scores["test"][nm])
        vs = {}
        for name, cols in (("history", FEATS), ("history+site", FEATS + SITE)):
            mx = make("xgboost").fit(tr[cols], tr[f"y{h}"])
            vs[name] = average_precision_score(va[f"y{h}"], mx.predict_proba(va[cols])[:, 1])
            recs.append({"horizon_years": h, "model": "xgboost", "task": "ablation", "features": name, "val_pr_auc": vs[name],
                         "test_pr_auc": float(average_precision_score(te[f"y{h}"], mx.predict_proba(te[cols])[:, 1]))})
        chosen = max(vs, key=vs.get)
        COLS = FEATS + SITE if chosen == "history+site" else FEATS
        CHOSEN[h] = chosen
        for nm in ("xgboost", "lightgbm", "random_forest"):
            t1 = time.perf_counter()
            m = make(nm).fit(tr[COLS], tr[f"y{h}"])
            fit_s = time.perf_counter() - t1
            pv, pt = m.predict_proba(va[COLS])[:, 1], m.predict_proba(te[COLS])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(pv, va[f"y{h}"].values)
            raw_test[nm] = pt
            scores["val"][nm], scores["test"][nm] = iso.predict(pv), iso.predict(pt)
            fit[nm] = fit_s
        mem = ["xgboost", "lightgbm", "random_forest"]
        w, _ = nnls(np.column_stack([scores["val"][m] for m in mem]), va[f"y{h}"].values)
        w = w / w.sum() if w.sum() else np.full(3, 1 / 3)
        for s in ("val", "test"):
            scores[s]["ensemble"] = np.column_stack([scores[s][m] for m in mem]) @ w
        raw_test["ensemble"] = scores["test"]["ensemble"]
        quiet = ((te.n0 == 0) & (te.n1 == 0) & (te.n2 == 0)).values
        for nm in scores["test"]:
            mv, mt = metrics(va, va[f"y{h}"].values, scores["val"][nm]), metrics(te, te[f"y{h}"].values, scores["test"][nm], raw_test[nm])
            rec = {"horizon_years": h, "model": nm, "features": chosen if nm in ("xgboost", "lightgbm", "random_forest", "ensemble") else "-", "n_train": len(tr), "n_val": len(va), "n_test": len(te), "train_years": f"{START_MIN(tr, h)}-{TRAIN_END}", "val_years": f"{TRAIN_END + 1}-{VAL_END}", "test_years": f"{VAL_END + 1}-{TEST_END}",
                   **{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}, "train_s": fit.get(nm, 0.0)}
            tq = te[quiet]
            if len(tq) and tq[f"y{h}"].sum() > 5:  # previously quiet cells: no permit in the last 3 years
                mq = metrics(tq, tq[f"y{h}"].values, scores["test"][nm][quiet])
                rec.update({f"quiet_{k}": v for k, v in mq.items()})
            if nm in ("xgboost", "lightgbm", "random_forest", "ensemble"):
                rec["ensemble_weights"] = dict(zip(mem, w.round(3))) if nm == "ensemble" else None
            recs.append(rec)
            for sp, fr in (("val", va), ("test", te)):
                preds.append(pd.DataFrame({"horizon_years": h, "model": nm, "split": sp, "cell": fr.cell.values, "metro": fr.metro_name.values, "origin": fr.origin.values,
                                           "y": fr[f"y{h}"].values, "prob": scores[sp][nm]}))
        # per-metro test breakdown and leave-one-metro-out generalisation (ensemble members only, h=1)
        if h == 1:
            for mname, mid in METRO_ID.items():
                trn, tst = tr[tr.metro != mid], te[te.metro == mid]
                if len(tst) < 200 or tst.y1.sum() < 10:
                    continue
                mm = make("xgboost").fit(trn[COLS], trn.y1)
                p = mm.predict_proba(tst[COLS])[:, 1]
                own = scores["test"]["xgboost"][(te.metro == mid).values]
                recs.append({"horizon_years": 1, "model": "xgboost", "task": "metro_holdout", "metro": mname, "n_test": len(tst), "pr_auc_heldout_metro": float(average_precision_score(tst.y1, p)),
                             "pr_auc_trained_incl_metro": float(average_precision_score(tst.y1, own)), "pr_auc_baseline_recent_history": float(average_precision_score(tst.y1, baselines(tst)["baseline_recent_history"])),
                             "base_rate": float(tst.y1.mean())})
    res = pd.DataFrame(recs)
    (OUT / "metro_features.json").write_text(json.dumps(CHOSEN))
    res.to_json(OUT / "metro_metrics.json", orient="records", indent=1)
    pd.concat(preds).to_parquet(OUT / "metro_predictions.parquet")
    return res, df


CHOSEN = {}


def START_MIN(tr, h):
    return int((tr.origin + h).min())


if __name__ == "__main__":
    res, df = run()
    pd.set_option("display.width", 250, "display.max_columns", 30)
    r = res[res.task.isna()] if "task" in res else res
    print("\nTEST, h=1y"); print(r[r.horizon_years == 1][["model", "test_pr_auc", "test_roc_auc", "test_precision_top5pct", "test_recall_top5pct", "test_ece", "quiet_pr_auc", "quiet_recall_top5pct", "train_s"]].round(3).to_string(index=False))
    print("\nTEST, h=2y"); print(r[r.horizon_years == 2][["model", "test_pr_auc", "test_roc_auc", "test_precision_top5pct", "test_recall_top5pct", "quiet_pr_auc", "quiet_recall_top5pct"]].round(3).to_string(index=False))
    print("\nmetro holdout"); print(res[res.get("task") == "metro_holdout"][["metro", "n_test", "base_rate", "pr_auc_heldout_metro", "pr_auc_trained_incl_metro", "pr_auc_baseline_recent_history"]].round(3).to_string(index=False))
