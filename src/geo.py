"""County geometry helpers (plotly's public county GeoJSON; no geopandas needed)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
GEO_PATH = ROOT / "data" / "counties.geojson"
GEO_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


def load_geojson() -> dict:
    if not GEO_PATH.exists():
        GEO_PATH.write_bytes(requests.get(GEO_URL, timeout=120).content)
    return json.loads(GEO_PATH.read_text())


def _coords(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return np.concatenate([np.array(ring) for poly in polys for ring in poly[:1]])


def centroids() -> pd.DataFrame:
    rows = []
    for f in load_geojson()["features"]:
        c = _coords(f["geometry"])
        rows.append((f["id"], c[:, 0].mean(), c[:, 1].mean()))
    return pd.DataFrame(rows, columns=["fips", "lon", "lat"])


def assign_counties(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Point-in-polygon county FIPS for each point ('' if outside every county)."""
    from matplotlib.path import Path as MplPath

    shapes = []
    for f in load_geojson()["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            ring = np.array(poly[0])
            shapes.append((f["id"], ring[:, 0].min(), ring[:, 0].max(), ring[:, 1].min(), ring[:, 1].max(), MplPath(ring)))
    out = np.full(len(lon), "", dtype=object)
    for i, (x, y) in enumerate(zip(lon, lat)):
        for fips, x0, x1, y0, y1, path in shapes:
            if x0 <= x <= x1 and y0 <= y <= y1 and path.contains_point((x, y)):
                out[i] = fips
                break
    return out
