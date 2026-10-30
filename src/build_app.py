"""Build outputs/app.html: a self-contained, filterable US county map of construction activity."""
import json

import numpy as np
import pandas as pd

from data_qcew import SECTORS
from geo import ROOT, load_geojson

q = pd.read_parquet(ROOT / "data" / "qcew.parquet")
dc = pd.read_parquet(ROOT / "data" / "datacenters.parquet")
panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
geo = load_geojson()

fips = [f["id"] for f in geo["features"]]
names = panel.drop_duplicates("fips", keep="last").set_index("fips").name.to_dict()
years = sorted(q.year.unique().tolist())

emp = {}
for code in SECTORS:
    wide = q[q.industry_code == code].pivot_table(index="area_fips", columns="year", values="annual_avg_emplvl").reindex(fips)
    emp[code] = {int(y): [None if np.isnan(v) else int(v) for v in wide[y]] for y in years}

dc_counts = dc.fips.value_counts().reindex(fips).fillna(0).astype(int).tolist()
payload = dict(
    fips=fips, names=[names.get(f, f) for f in fips], years=[int(y) for y in years],
    sectors=SECTORS, emp=emp, dc_counts=dc_counts,
    dc=dict(lat=dc.lat.round(4).tolist(), lon=dc.lon.round(4).tolist(),
            text=(dc.name + np.where(dc.operator != "", " (" + dc.operator + ")", "")).tolist()),
)

html = (ROOT / "src" / "app_template.html").read_text()
html = html.replace("__DATA__", json.dumps(payload, separators=(",", ":"))).replace("__GEO__", json.dumps(geo, separators=(",", ":")))
out = ROOT / "outputs" / "app.html"
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
