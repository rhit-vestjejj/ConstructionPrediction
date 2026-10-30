"""Download Census Building Permits Survey county files and build a county-year panel."""
import io
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
URL = "https://www2.census.gov/econ/bps/County/co{year}a.txt"
import datetime
THIS_YEAR = datetime.date.today().year
YEARS = range(1990, THIS_YEAR)  # annual county files; the current year is never complete
YTD_URL = "https://www2.census.gov/econ/bps/County/co{yy}08y.txt"  # January-August year-to-date (exists from 2000)

COLS = ["year", "state", "county", "region", "division", "name"]
for kind in ["1u", "2u", "34u", "5u"]:
    COLS += [f"{kind}_bldgs", f"{kind}_units", f"{kind}_value"]
COLS += [f"rep_{i}" for i in range(12)]  # "reported" columns, unused


def fetch(year: int, ytd: bool = False) -> pd.DataFrame | None:
    path = RAW / (f"co{year % 100:02d}08y.txt" if ytd else f"co{year}a.txt")
    if not path.exists():
        r = requests.get(YTD_URL.format(yy=f"{year % 100:02d}") if ytd else URL.format(year=year), timeout=60)
        if r.status_code != 200:
            print(f"{year}: HTTP {r.status_code}, skipping")
            return None
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    df = pd.read_csv(path, skiprows=3, header=None, names=COLS, dtype={"state": str, "county": str},
                     encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["state", "county"]).drop(columns=[c for c in df if c.startswith("rep_")])
    df["year"] = year  # early files have a garbled date column (e.g. "9099"); trust the filename
    for c in ["state", "county", "name"]:
        df[c] = df[c].astype(str).str.strip()
    return df


def build_panel() -> pd.DataFrame:
    frames = [f for y in YEARS if (f := fetch(y)) is not None]
    df = pd.concat(frames, ignore_index=True)
    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df["name"] = df["name"].str.strip()
    df["units"] = df[["1u_units", "2u_units", "34u_units", "5u_units"]].sum(axis=1)
    df["value"] = df[["1u_value", "2u_value", "34u_value", "5u_value"]].sum(axis=1)
    df["multifamily_units"] = df[["2u_units", "34u_units", "5u_units"]].sum(axis=1)
    keep = ["fips", "year", "name", "state", "units", "value", "multifamily_units", "1u_units"]
    return df[keep].rename(columns={"1u_units": "sf_units"})


def build_ytd() -> pd.DataFrame:
    """Permitted units January-August of each year (2000-latest): an early read on the year in progress."""
    frames = [f.assign(year=y) for y in range(2000, THIS_YEAR + 1) if (f := fetch(y, ytd=True)) is not None]
    df = pd.concat(frames, ignore_index=True)
    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    df["units_ytd"] = df[["1u_units", "2u_units", "34u_units", "5u_units"]].sum(axis=1)
    return df[["fips", "year", "units_ytd"]]


if __name__ == "__main__":
    ytd = build_ytd()
    ytd.to_parquet(ROOT / "data" / "permits_ytd.parquet")
    print("YTD (Jan-Aug):", ytd.year.min(), "-", ytd.year.max(), ytd.groupby("year").units_ytd.sum().tail(3).to_dict())
    panel = build_panel()
    panel.to_parquet(ROOT / "data" / "panel.parquet")
    print(panel.shape, panel.year.min(), panel.year.max(), panel.fips.nunique(), "counties")
    print(panel.groupby("year").units.sum().tail(8))
