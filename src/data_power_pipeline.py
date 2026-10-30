"""Time-varying power features from yearly EIA-860 files (2015-2025): operating and PLANNED generating capacity near
each county. Planned (proposed) generators are an early sign of where large new loads are expected.

Vintage y describes the grid at the end of year y and is published in the second half of y+1, so a forecast made
with data through year t uses vintage t-1 (see dc_timed.py).  Output: data/power_pipeline.parquet (fips x vintage).
"""
import glob
import io
import re
import zipfile

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from geo import ROOT, centroids

RAW = ROOT / "data" / "raw" / "eia860"


def read_sheet(z, name_part, sheet=0):
    name = next(n for n in z.namelist() if name_part in n)
    raw = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=sheet, header=None, nrows=5)
    hdr = int(next(i for i in range(5) if raw.iloc[i].astype(str).str.contains("Plant Code").any()))
    return pd.read_excel(io.BytesIO(z.read(name)), sheet_name=sheet, header=hdr)


def vintage(path):
    y = int(re.search(r"eia860(\d{4})", path.split("/")[-1]).group(1))
    z = zipfile.ZipFile(path)
    plants = read_sheet(z, "2___Plant")[["Plant Code", "Latitude", "Longitude"]]
    plants[["Latitude", "Longitude"]] = plants[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce")
    out = {}
    for sheet in ("Operable", "Proposed"):
        g = read_sheet(z, "3_1_Generator", sheet)
        g = g.merge(plants, on="Plant Code").dropna(subset=["Latitude", "Longitude"])
        g["mw"] = pd.to_numeric(g["Nameplate Capacity (MW)"], errors="coerce").fillna(0)
        g["gas"] = g["Energy Source 1"].astype(str).eq("NG")
        out[sheet] = g.groupby("Plant Code").agg(lat=("Latitude", "first"), lon=("Longitude", "first"), mw=("mw", "sum"),
                                                gas_mw=("mw", lambda s: s[g.loc[s.index, "gas"]].sum()))
    return y, out


def near(plants: pd.DataFrame, cen: pd.DataFrame, km: float, col: str) -> np.ndarray:
    if not len(plants):
        return np.zeros(len(cen))
    tree = BallTree(np.radians(plants[["lat", "lon"]].values), metric="haversine")
    idx = tree.query_radius(np.radians(cen[["lat", "lon"]].values), r=km / 6371.0)
    v = plants[col].values
    return np.array([v[i].sum() for i in idx])


if __name__ == "__main__":
    cen = centroids()
    rows = []
    for path in sorted(glob.glob(str(RAW / "eia860*.zip"))):
        y, g = vintage(path)
        f = pd.DataFrame({"fips": cen.fips, "vintage": y,
                          "op_mw_100km": near(g["Operable"], cen, 100, "mw"),
                          "prop_mw_100km": near(g["Proposed"], cen, 100, "mw"),
                          "prop_gas_mw_100km": near(g["Proposed"], cen, 100, "gas_mw"),
                          "prop_mw_25km": near(g["Proposed"], cen, 25, "mw")})
        rows.append(f)
        print(y, f"proposed MW total {g['Proposed'].mw.sum():,.0f} ({g['Proposed'].gas_mw.sum():,.0f} gas) at {len(g['Proposed'])} plants", flush=True)
    pd.concat(rows).to_parquet(ROOT / "data" / "power_pipeline.parquet")
    print("saved", sum(len(r) for r in rows), "rows")
