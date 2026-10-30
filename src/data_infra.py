"""County siting-factor features from free public data. Snapshot as of 2023 (vintages noted per source).

  EIA-860 (2023)      generating capacity by county (via plant lat/lon) and within 50/150 km
  EIA-861 (2023)      industrial electricity price and utility count by county (via service territories)
  NOAA nClimDiv       cooling/heating degree days, summer max temperature (2013-2023 mean)
  FEMA NRI            natural hazard risk scores
  HIFLD lines >=230kV distance to the nearest high-voltage line (layer is ~2014 vintage: stale)
"""
import glob
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from geo import ROOT, assign_counties, centroids

RAW = ROOT / "data" / "raw" / "infra"
FUEL = {"NG": "gas", "BIT": "coal", "SUB": "coal", "LIG": "coal", "RC": "coal", "WC": "coal", "NUC": "nuc", "WAT": "hydro", "WND": "windsolar", "SUN": "windsolar", "MWH": "storage"}
NOAA_STATES = "AL AZ AR CA CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
ABBR_FIPS = dict(zip("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split(),
                     "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split()))
NOAA_TO_FIPS = {f"{i + 1:02d}": ABBR_FIPS[s] for i, s in enumerate(NOAA_STATES)}


def power(cent: pd.DataFrame) -> pd.DataFrame:
    z = zipfile.ZipFile(RAW / "eia8602023.zip")
    g = pd.read_excel(io.BytesIO(z.read("3_1_Generator_Y2023.xlsx")), "Operable", header=1)
    p = pd.read_excel(io.BytesIO(z.read("2___Plant_Y2023.xlsx")), header=1)[["Plant Code", "Latitude", "Longitude", "Grid Voltage (kV)"]]
    p[["Latitude", "Longitude"]] = p[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce")
    g = g[(g.Status == "OP") & (g["Operating Year"] <= 2023)].merge(p, on="Plant Code").dropna(subset=["Latitude", "Longitude"])
    g["mw"] = pd.to_numeric(g["Nameplate Capacity (MW)"], errors="coerce").fillna(0)
    g["fuel"] = g["Energy Source 1"].map(FUEL).fillna("other")
    g["fips"] = assign_counties(g.Longitude.values, g.Latitude.values)
    print(f"EIA-860: {len(g)} operating generators, {int((g.fips == '').sum())} outside county polygons")
    g = g[g.fips != ""]
    out = g.pivot_table(index="fips", columns="fuel", values="mw", aggfunc="sum", fill_value=0).add_prefix("mw_")
    out["mw_total"] = g.groupby("fips").mw.sum()
    out["n_plants"] = g.groupby("fips")["Plant Code"].nunique()
    out["max_grid_kv"] = pd.to_numeric(g["Grid Voltage (kV)"], errors="coerce").groupby(g.fips).max()
    out = out.reindex(cent.index).fillna({c: 0 for c in out.columns if c != "max_grid_kv"})
    # capacity reachable around the county centre (haversine, radians)
    pl = g.groupby("Plant Code").agg(lat=("Latitude", "first"), lon=("Longitude", "first"), mw=("mw", "sum"))
    tree = BallTree(np.radians(pl[["lat", "lon"]].values), metric="haversine")
    xy = np.radians(cent[["lat", "lon"]].values)
    for km in (50, 150):
        idx = tree.query_radius(xy, r=km / 6371.0)
        out[f"mw_{km}km"] = [pl.mw.values[i].sum() for i in idx]
    return out


def norm_county(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.replace(r"\b(county|parish|borough|census area|municipality)\b", "", regex=True)
    return s.str.replace(r"\bst\.? ", "saint ", regex=True).str.replace(r"[^a-z]", "", regex=True)


def utilities(cent: pd.DataFrame) -> pd.DataFrame:
    z = zipfile.ZipFile(RAW / "f8612023.zip")
    terr = pd.read_excel(io.BytesIO(z.read("Service_Territory_2023.xlsx")))
    sales = pd.read_excel(io.BytesIO(z.read("Sales_Ult_Cust_2023.xlsx")), header=[0, 1, 2])
    sales.columns = ["_".join(str(x) for x in c if "Unnamed" not in str(x)) for c in sales.columns]
    num = lambda c: pd.to_numeric(sales[c], errors="coerce")
    sales = sales.assign(util=pd.to_numeric(sales["Utility Characteristics_Utility Number"], errors="coerce").astype("Int64"), part=sales["Utility Characteristics_Service Type"].astype(str).str.strip(),
                         rev=num("INDUSTRIAL_Revenues_Thousand Dollars") * 1000, mwh=num("INDUSTRIAL_Sales_Megawatthours"))
    b = sales[(sales.part == "Bundled") & (sales.mwh > 1000)].groupby("util").agg(rev=("rev", "sum"), mwh=("mwh", "sum"))
    b["price"] = b.rev / b.mwh  # $ per MWh, industrial, bundled service
    # county names -> FIPS via the permit panel's names (state FIPS + normalised county name)
    names = pd.read_parquet(ROOT / "data" / "panel.parquet").drop_duplicates("fips", keep="last")[["fips", "name"]]
    names["key"] = names.fips.str[:2] + "|" + norm_county(names.name)
    key2fips = dict(zip(names.key, names.fips))
    terr["fips"] = (terr.State.map(ABBR_FIPS).fillna("") + "|" + norm_county(terr.County)).map(key2fips)
    print(f"EIA-861: {terr.fips.notna().mean():.1%} of {len(terr)} service-territory rows matched to a county FIPS")
    t = terr.dropna(subset=["fips"]).assign(**{"Utility Number": lambda d: pd.to_numeric(d["Utility Number"], errors="coerce").astype("Int64")}).merge(b, left_on="Utility Number", right_index=True, how="left")
    t = t.drop_duplicates(["fips", "Utility Number"])
    w = t.dropna(subset=["price"]).assign(pw=lambda d: d.price * d.mwh)
    out = pd.DataFrame({
        "ind_price": w.groupby("fips").pw.sum() / w.groupby("fips").mwh.sum(),  # weighted by each utility's industrial MWh
        "ind_price_min": w.groupby("fips").price.min(),
        "n_utilities": t.groupby("fips")["Utility Number"].nunique()})
    return out.reindex(cent.index)


def climate(cent: pd.DataFrame) -> pd.DataFrame:
    res = {}
    for name, f in (("cdd", "climdiv-cdd.txt"), ("hdd", "climdiv-hdd.txt"), ("tmax", "climdiv-tmax.txt")):
        rows = []
        for line in (RAW / f).read_text().splitlines():
            code, year = line[:5], int(line[7:11])
            if 2013 <= year <= 2023:
                v = np.array(line[11:].split(), dtype=float)
                v[v <= -9990] = np.nan
                rows.append((NOAA_TO_FIPS.get(code[:2], "?") + code[2:], v))
        d = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
        g = d.groupby(level=0)
        res[name] = g.apply(lambda x: x.sum(axis=1, skipna=False).mean()) if name != "tmax" else g.apply(lambda x: x.iloc[:, 5:8].mean(axis=1).mean())
    out = pd.DataFrame({"cdd": res["cdd"], "hdd": res["hdd"], "tmax_summer": res["tmax"]})
    return out.reindex(cent.index)


def hazards(cent: pd.DataFrame) -> pd.DataFrame:
    r = pd.read_csv(RAW / "nri_counties_selected.csv", dtype={"STCOFIPS": str}).set_index("STCOFIPS")
    out = pd.DataFrame({"risk_score": r.RISK_SCORE, "eal_log": np.log1p(r.EAL_VALT), "resl_score": r.RESL_SCORE})
    for k in ("HRCN", "TRND", "ERQK", "WFIR", "CFLD", "HWAV", "DRGT"):
        out[f"risk_{k.lower()}"] = r[f"{k}_RISKS"]
    return out.reindex(cent.index)


def grid(cent: pd.DataFrame) -> pd.DataFrame:
    pts = {230: [], 345: []}
    for f in sorted(glob.glob(str(RAW / "lines230_*.geojson"))):
        for ft in json.loads(open(f).read())["features"]:
            kv = float((ft.get("properties") or {}).get("VOLTAGE") or 0)
            geom = ft.get("geometry") or {}
            lines = [geom["coordinates"]] if geom.get("type") == "LineString" else geom.get("coordinates", []) if geom.get("type") == "MultiLineString" else []
            for ln in lines:
                a = np.array(ln)[:, :2]
                for p0, p1 in zip(a[:-1], a[1:]):  # densify to ~5 km so distance isn't biased by sparse vertices
                    n = max(1, int(np.hypot((p1[0] - p0[0]) * 85, (p1[1] - p0[1]) * 111) // 5))
                    seg = p0 + (p1 - p0) * (np.arange(n)[:, None] / n)
                    pts[230].append(seg)
                    if kv >= 345:
                        pts[345].append(seg)
    xy = np.radians(cent[["lat", "lon"]].values)
    out = pd.DataFrame(index=cent.index)
    for kv, segs in pts.items():
        P = np.radians(np.vstack(segs)[:, ::-1])
        tree = BallTree(P, metric="haversine")
        out[f"kv{kv}_dist_km"] = tree.query(xy, k=1)[0][:, 0] * 6371.0
        if kv == 230:
            out["line_km_50km"] = np.array([len(i) for i in tree.query_radius(xy, r=50 / 6371.0)]) * 5.0
    print(f"HIFLD: {sum(len(s) for s in pts[230])} densified points >=230kV, {sum(len(s) for s in pts[345])} >=345kV")
    return out


if __name__ == "__main__":
    cent = centroids().set_index("fips")
    parts = {"power": power(cent), "utilities": utilities(cent), "climate": climate(cent), "hazards": hazards(cent), "grid": grid(cent)}
    df = pd.concat(parts.values(), axis=1)
    df.to_parquet(ROOT / "data" / "infra_features.parquet")
    print(df.shape, "counties;\nnon-missing share by column:")
    print(df.notna().mean().round(2).to_string())
    print("\nsanity (median over counties):"); print(df.median().round(1).to_string())
    print("\ntop counties by mw_total:"); print(df.mw_total.sort_values(ascending=False).head(5).round(0).to_string())
