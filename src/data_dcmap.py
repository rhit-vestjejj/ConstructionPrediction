"""County pipeline snapshot from US Data Center Map (https://dcmap.us) via its public agent API: data/dcmap_counties.parquet.

Aggregates only (operating / under construction / planned counts and reported MW for its ~90 busiest counties). Their
usage terms welcome API use with attribution but not bulk scraping or republishing the dataset, so we take only this
file and always show the source and as-of date. It is a single current snapshot with no history, so it is NOT a model
input (a walk-forward backtest would leak); the site uses it as context per county and as an outside check on the
forecast. Cite as: US Data Center Map, data as of <data_as_of>. https://dcmap.us"""
import json
import sys

import pandas as pd
import requests

from data_dc_projects import county_fips
from geo import ROOT

URL = "https://dcmap.us/agent-api/v2/counties.json"
RAW = ROOT / "data" / "raw" / "dcmap" / "counties.json"


def fetch(refresh=False) -> dict:
    if refresh or not RAW.exists():
        r = requests.get(URL, timeout=60, headers={"User-Agent": "ConstructionPrediction/0.1 (vestjejj@rose-hulman.edu)"})
        r.raise_for_status()
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(r.text)
    return json.loads(RAW.read_text())


def main(refresh=False):
    d = fetch(refresh)
    rows = pd.DataFrame([{"name": c["name"], "state": c["state"], "url": c["source_url"], **{k: c["status_counts"].get(k, 0) for k in ("operational", "construction", "planned")},
                          "power_mw": c.get("reported_power_mw") or 0.0, "pipeline_mw": c.get("pipeline_mw") or 0.0} for c in d["data"] if c.get("kind") == "county"])
    rows["fips"] = county_fips(rows.state, rows.name)
    miss = rows[rows.fips.isna()]
    if len(miss):
        print("unmatched:", ", ".join(miss.name + " " + miss.state))
    rows = rows.dropna(subset=["fips"])
    rows.assign(as_of=d["data_as_of"]).to_parquet(ROOT / "data" / "dcmap_counties.parquet", index=False)
    print(f"dcmap: {len(rows)} counties, data as of {d['data_as_of']}")


if __name__ == "__main__":
    main("--refresh" in sys.argv)
