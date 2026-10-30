"""US data center locations from OpenStreetMap (Overpass), assigned to counties."""
import json
import time

import pandas as pd
import requests

from geo import ROOT, assign_counties

MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
QUERY = """[out:json][timeout:120];
(nwr["telecom"="data_center"](24,-125,50,-66);nwr["building"="data_center"](24,-125,50,-66););out center tags;"""


def fetch() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "osm_data_centers.json"
    if not path.exists():
        for attempt in range(6):
            try:
                r = requests.post(MIRRORS[attempt % 2], data={"data": QUERY}, timeout=150,
                                  headers={"User-Agent": "ConstructionPrediction/0.1 (personal research)", "Accept": "*/*"})
                if r.ok:
                    break
                print(f"overpass HTTP {r.status_code}, retrying")
            except requests.exceptions.RequestException as e:
                print(f"overpass {type(e).__name__}, retrying")
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    rows = []
    for e in json.loads(path.read_text())["elements"]:
        lat, lon = e.get("lat"), e.get("lon")
        if lat is None:
            lat, lon = e["center"]["lat"], e["center"]["lon"]
        t = e.get("tags", {})
        rows.append((e["id"], t.get("name", "Data center"), t.get("operator", ""), lat, lon))
    return pd.DataFrame(rows, columns=["osm_id", "name", "operator", "lat", "lon"]).drop_duplicates("osm_id")


if __name__ == "__main__":
    d = fetch()
    d = d[(d.lat.between(24, 50)) & (d.lon.between(-125, -66))]  # contiguous US (the map's scope for now)
    d["fips"] = assign_counties(d.lon.values, d.lat.values)
    d = d[d.fips != ""]
    d.to_parquet(ROOT / "data" / "datacenters.parquet")
    print(len(d), "data centers;", d.fips.nunique(), "counties")
    print(d.fips.value_counts().head(5))
