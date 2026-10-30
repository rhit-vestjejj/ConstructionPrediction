"""New-construction commercial/industrial/warehouse permits >= $1M for metros whose open-data portals publish them.
Dated and geolocated, so they allow cell-level (H3) analysis inside those metros. Metro-only: NOT a national dataset."""
import time

import numpy as np
import pandas as pd
import requests

from geo import ROOT

RAW = ROOT / "data" / "raw" / "permits"
UA = {"User-Agent": "Mozilla/5.0 (compatible; ConstructionPrediction research)"}
SERVICES = {
    "fortworth": dict(url="https://mapit.fortworthtexas.gov/ags/rest/services/CIVIC/Permits/MapServer/0/query",
                      where="Permit_Type='Commercial Building Permit' AND Permit_SubType='New' AND JobValue>1000000",
                      fields="Permit_No,File_Date,JobValue,Use_Type,Specific_Use,SqFt,Latitude,Longitude,B1_WORK_DESC,Address", order="File_Date"),
    "mecklenburg": dict(url="https://meckgis.mecklenburgcountync.gov/server/rest/services/BuildingPermits/FeatureServer/0/query",
                        where="bldgcost>1000000 AND usdcdesc LIKE '3%'", fields="permitnum,issuedate,bldgcost,usdcdesc,occupancy,totalsqft,workdesc", order="issuedate", geometry=True),
    "columbus": dict(url="https://services1.arcgis.com/9yy6msODkIBzkUXU/arcgis/rest/services/Building_Permits/FeatureServer/0/query",
                     where="GENERAL_TYPE='Commercial - New Structure' AND G3_VALUE_TTL>1000000",
                     fields="B1_ALT_ID,ISSUED_DT,G3_VALUE_TTL,SQFT,VALUE_DESC,CONST_TYPE_CODE", order="ISSUED_DT", geometry=True),
}


def pull(name: str, cfg: dict) -> pd.DataFrame:
    path = RAW / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows, off = [], 0
    while True:
        params = {"where": cfg["where"], "outFields": cfg["fields"], "returnGeometry": "true" if cfg.get("geometry") else "false", "outSR": 4326,
                  "resultOffset": off, "resultRecordCount": 1000, "orderByFields": cfg["order"], "f": "json"}
        j = requests.get(cfg["url"], params=params, headers=UA, timeout=120).json()
        if "error" in j:
            raise RuntimeError(f"{name}: {j['error']}")
        feats = j.get("features", [])
        for f in feats:
            r = dict(f["attributes"])
            g = f.get("geometry") or {}
            if "x" in g:
                r["_lon"], r["_lat"] = g["x"], g["y"]
            rows.append(r)
        print(f"{name}: {len(rows)} rows", end="\r")
        if len(feats) < 1000 and not j.get("exceededTransferLimit"):
            break
        off += len(feats)
        time.sleep(0.3)
    RAW.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    print(f"{name}: {len(df)} rows saved          ")
    return df


def geocode_missing_fortworth(df: pd.DataFrame) -> pd.DataFrame:
    """Recover coordinates for rows without them via the Census batch geocoder (free, <=10k rows per call)."""
    path = RAW / "fortworth_geocoded.parquet"
    if path.exists():
        return pd.read_parquet(path)
    miss = df[df.Latitude.isna() & (df.Use_Type != "Residential") & df.Address.notna()]
    csv = "\n".join(f'{i},"{a.strip().replace(chr(34), "")}",Fort Worth,TX,' for i, a in zip(miss.index, miss.Address))
    r = requests.post("https://geocoding.geo.census.gov/geocoder/locations/addressbatch", files={"addressFile": ("a.csv", csv)},
                      data={"benchmark": "Public_AR_Current"}, timeout=600)
    out = pd.read_csv(pd.io.common.StringIO(r.text), header=None, names=["id", "in", "match", "mtype", "matched", "coord", "tiger", "side"], on_bad_lines="skip")
    ok = out[out.match == "Match"].dropna(subset=["coord"])
    xy = ok.coord.str.split(",", expand=True).astype(float)
    df = df.copy()
    df.loc[ok.id.values, "Longitude"], df.loc[ok.id.values, "Latitude"] = xy[0].values, xy[1].values
    print(f"fortworth: geocoded {len(ok)} of {len(miss)} rows without coordinates")
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    for n, c in SERVICES.items():
        d = pull(n, c)
        if n == "fortworth":
            d = geocode_missing_fortworth(d)
        print(n, d.shape, list(d.columns))
