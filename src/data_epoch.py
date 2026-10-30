"""Epoch AI frontier data center sites (CC-BY 4.0): geocode US addresses to counties, derive dated milestones."""
import json
import re
import time

import numpy as np
import pandas as pd
import requests

from geo import ROOT

RAW = ROOT / "data" / "raw" / "epoch"
GEOCODER = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"


def geocode(addr: str) -> dict | None:
    r = requests.get(GEOCODER, params={"address": addr, "benchmark": "Public_AR_Current", "vintage": "Current_Current", "format": "json"}, timeout=60)
    m = r.json()["result"]["addressMatches"]
    if not m:
        return None
    c = m[0]["geographies"]["Counties"][0]
    return {"fips": c["GEOID"], "county": c["NAME"], "lon": m[0]["coordinates"]["x"], "lat": m[0]["coordinates"]["y"]}


def zip_to_county() -> dict:
    """ZIP (ZCTA) -> county with the largest land overlap. Approximate: used only when the street is not in Census range files."""
    z = pd.read_csv(ROOT / "data" / "raw" / "zcta_county.txt", sep="|", dtype=str, encoding="utf-8-sig", usecols=["GEOID_ZCTA5_20", "GEOID_COUNTY_20", "AREALAND_PART"]).dropna()
    z["a"] = z.AREALAND_PART.astype(float)
    return z.sort_values("a").drop_duplicates("GEOID_ZCTA5_20", keep="last").set_index("GEOID_ZCTA5_20").GEOID_COUNTY_20.to_dict()


def fallback(addr: str) -> dict | None:
    m = re.findall(r"\b(\d{5})\b", addr)
    f = zip_to_county().get(m[-1]) if m else None
    return {"fips": f, "lon": None, "lat": None, "approx": True} if f else None


# Sites with no usable address in the Epoch file. County from the project's publicly reported location; flagged "manual".
# Left out on purpose (location unclear to me): Google Kansas City East, Anthropic Barber Lake.
MANUAL = {"Google Mesa": "04013", "Amazon Madison Mega Site": "28089", "Google The Dalles": "41065", "Meta Huntsville": "01089",
          "Google Storey County": "32029", "OpenAI Stargate Michigan": "26161", "OpenAI Stargate Milam": "48331",
          "OpenAI Stargate New Mexico": "35013", "OpenAI Stargate Wisconsin": "55089"}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = pd.read_csv(RAW / "data_centers.csv")
    sites = sites[sites.Country == "United States"].copy()
    cache_path = RAW / "geocode_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for a in sites.Address.dropna().unique():
        if cache.get(a) is None:  # retry earlier misses with the fallback
            for attempt in range(3):
                try:
                    cache[a] = geocode(a)
                    break
                except Exception:
                    time.sleep(3)
            else:
                cache[a] = None
            if cache[a] is None:
                cache[a] = fallback(a)
            cache_path.write_text(json.dumps(cache))
    geo = sites.Address.map(cache)
    sites["fips"] = geo.map(lambda g: g["fips"] if isinstance(g, dict) else None)
    sites["lat"] = geo.map(lambda g: g["lat"] if isinstance(g, dict) else None)
    sites["lon"] = geo.map(lambda g: g["lon"] if isinstance(g, dict) else None)
    manual = sites.Name.map(MANUAL)
    sites["location_source"] = np.where(sites.fips.notna(), np.where(geo.map(lambda g: bool(g.get("approx", False)) if isinstance(g, dict) else False), "zip", "address"), np.where(manual.notna(), "manual", "none"))
    sites["fips"] = sites.fips.fillna(manual)
    sites["approx_location"] = geo.map(lambda g: bool(g.get("approx", False)) if isinstance(g, dict) else None)
    t = pd.read_csv(RAW / "data_center_timelines.csv", parse_dates=["Date"])
    t = t[t["Data center"].isin(sites.Name)].sort_values("Date")
    first = t.groupby("Data center").Date.min().rename("first_dated_milestone")
    peak_mw = t.groupby("Data center")["Power (MW)"].max().rename("planned_peak_mw")
    sites = sites.merge(first, left_on="Name", right_index=True, how="left").merge(peak_mw, left_on="Name", right_index=True, how="left")
    return sites, t


if __name__ == "__main__":
    sites, t = build()
    out = sites[["Name", "Owner", "Address", "fips", "lat", "lon", "Current power (MW)", "location_source", "first_dated_milestone", "planned_peak_mw"]]
    out.to_parquet(ROOT / "data" / "epoch_sites.parquet")
    t.to_parquet(ROOT / "data" / "epoch_timelines.parquet")
    print(len(out), "US sites;", out.fips.notna().sum(), "geocoded to", out.fips.nunique(), "counties")
    print(out.first_dated_milestone.dt.year.value_counts().sort_index().to_dict())
    print(out[out.fips.isna()].Name.tolist())
