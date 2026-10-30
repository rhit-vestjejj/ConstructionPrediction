"""Political lean, transport access and land value.

  Elections: county presidential results 2008-2024 (github.com/tonmcg/US_County_Level_Election_Results_08-24).
             lean_asof(year) only uses elections held on or before that year (no look-ahead).
  Transport: Census TIGER 2023 primary roads (interstates = RTTYP 'I') and rail lines -> distance from any point.
             Static 2023 network: mild look-ahead for historical origins (road networks change slowly).
  Land:      USDA NASS Census of Agriculture 2022, farmland incl. buildings, $/acre (rough land-cost proxy).
"""
import numpy as np
import pandas as pd
import shapefile
from sklearn.neighbors import BallTree

from geo import ROOT, centroids

RAW = ROOT / "data" / "raw" / "context"
ELECTION_YEARS = (2008, 2012, 2016, 2020, 2024)


def elections() -> pd.DataFrame:
    old = pd.read_csv(RAW / "US_County_Level_Presidential_Results_08-16.csv")
    rows = []
    for y in (2008, 2012, 2016):
        rows.append(pd.DataFrame({"fips": old.fips_code.astype(int).astype(str).str.zfill(5), "year": y, "gop": old[f"gop_{y}"], "dem": old[f"dem_{y}"], "total": old[f"total_{y}"]}))
    for y in (2020, 2024):
        d = pd.read_csv(RAW / f"{y}_US_County_Level_Presidential_Results.csv", dtype={"county_fips": str})
        rows.append(pd.DataFrame({"fips": d.county_fips.str.zfill(5), "year": y, "gop": d.votes_gop, "dem": d.votes_dem, "total": d.total_votes}))
    e = pd.concat(rows, ignore_index=True)
    e["state"] = e.fips.str[:2]
    return e


def lean_asof(e: pd.DataFrame, year: int) -> pd.DataFrame:
    """County and state lean from the latest presidential election held on or before `year` (NaN before 2008)."""
    past = [y for y in ELECTION_YEARS if y <= year]
    if not past:
        return pd.DataFrame(columns=["county_gop_margin", "state_gop_margin", "state_red"])
    d = e[e.year == past[-1]]
    st = d.groupby("state")[["gop", "dem", "total"]].sum()
    st_margin = (st.gop - st.dem) / st.total
    out = pd.DataFrame({"county_gop_margin": ((d.gop - d.dem) / d.total).values}, index=d.fips.values)
    out["state_gop_margin"] = out.index.str[:2].map(st_margin)
    out["state_red"] = (out.state_gop_margin > 0).astype(float)
    return out[~out.index.duplicated()]


def _line_points(zip_path, keep=lambda rec: True, step_km=2.0):
    r = shapefile.Reader(str(zip_path))
    pts = []
    for sr in r.iterShapeRecords():
        if not keep(sr.record):
            continue
        p = np.array(sr.shape.points)
        parts = list(sr.shape.parts) + [len(p)]
        for a, b in zip(parts[:-1], parts[1:]):
            seg = p[a:b]
            for p0, p1 in zip(seg[:-1], seg[1:]):
                n = max(1, int(np.hypot((p1[0] - p0[0]) * 85, (p1[1] - p0[1]) * 111) // step_km))
                pts.append(p0 + (p1 - p0) * (np.arange(n)[:, None] / n))
    return np.vstack(pts)


_TREES = {}


def _tree(name):
    if name not in _TREES:
        if name == "interstate":
            P = _line_points(RAW / "tl_2023_us_primaryroads.zip", keep=lambda rec: rec["RTTYP"] == "I")
        elif name == "primary":
            P = _line_points(RAW / "tl_2023_us_primaryroads.zip")
        else:
            P = _line_points(RAW / "tl_2023_us_rails.zip", keep=lambda rec: rec["MTFCC"] == "R1011")  # main-line rail
        _TREES[name] = BallTree(np.radians(P[:, ::-1]), metric="haversine")
    return _TREES[name]


def transport(lat, lon) -> pd.DataFrame:
    xy = np.radians(np.c_[lat, lon])
    out = {}
    for name in ("interstate", "primary", "rail"):
        out[f"{name}_dist_km"] = _tree(name).query(xy, k=1)[0][:, 0] * 6371.0
    out["primary_km_25km"] = np.array([len(i) for i in _tree("primary").query_radius(xy, r=25 / 6371.0)]) * 2.0
    return pd.DataFrame(out)


def land_value() -> pd.Series:
    q = pd.read_csv(RAW / "qs_census2022_county_land_value.tsv", sep="\t", dtype=str)
    q = q[(q.SHORT_DESC == "AG LAND, INCL BUILDINGS - ASSET VALUE, MEASURED IN $ / ACRE") & (q.DOMAIN_DESC == "TOTAL") & (q.AGG_LEVEL_DESC == "COUNTY")]
    v = pd.to_numeric(q.VALUE.str.replace(",", ""), errors="coerce")  # "(D)" = suppressed -> NaN
    return pd.Series(np.log(v.values), index=(q.STATE_FIPS_CODE.str.zfill(2) + q.COUNTY_CODE.str.zfill(3)).values, name="land_value_log").groupby(level=0).first()


if __name__ == "__main__":
    e = elections()
    e.to_parquet(ROOT / "data" / "elections.parquet")
    c = centroids().set_index("fips")
    st = transport(c.lat.values, c.lon.values).set_index(c.index)
    st["land_value_log"] = land_value().reindex(c.index)
    st.to_parquet(ROOT / "data" / "context_static.parquet")
    print("elections:", e.groupby("year").fips.nunique().to_dict())
    l = lean_asof(e, 2024)
    print("2024 red states:", int(l.groupby(l.index.str[:2]).state_red.first().sum()), "of", l.index.str[:2].nunique())
    print(st.notna().mean().round(2).to_dict())
    print(st.median().round(1).to_dict())
    for f, n in (("48439", "Tarrant TX"), ("51107", "Loudoun VA"), ("17031", "Cook IL"), ("30091", "Sheridan MT")):
        print(n, st.loc[f].round(1).to_dict(), "| lean:", l.loc[f].round(2).to_dict())
