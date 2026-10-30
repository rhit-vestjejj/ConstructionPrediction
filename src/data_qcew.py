"""BLS QCEW county construction employment by sector (private ownership), 2010-latest."""
import io

import numpy as np
import pandas as pd
import requests

from geo import ROOT

URL = "https://data.bls.gov/cew/data/api/{year}/a/industry/{ind}.csv"
UA = {"User-Agent": "ConstructionPrediction/0.1 (personal research)"}
SECTORS = {"23": "All construction", "236": "Buildings (res + commercial)", "237": "Heavy & civil", "238": "Specialty trades"}
# Extra series used by the models: 2361 residential building, 2362 nonresidential building (commercial/industrial/
# institutional), 493 warehousing & storage and 518 data processing/hosting (descriptive only, not forecast targets).
EXTRA = {"2361": "Residential building", "2362": "Nonresidential building", "493": "Warehousing & storage", "518": "Data processing & hosting"}
KEEP = ["disclosure_code", "area_fips", "industry_code", "year", "annual_avg_estabs", "annual_avg_emplvl", "total_annual_wages", "avg_annual_pay"]


def fetch(year: int, ind: str) -> pd.DataFrame | None:
    path = ROOT / "data" / "raw" / "qcew" / f"{year}_{ind}.csv"
    if not path.exists():
        r = requests.get(URL.format(year=year, ind=ind), headers=UA, timeout=120)
        if r.status_code != 200:
            print(f"{year} {ind}: HTTP {r.status_code}, skipping")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    df = pd.read_csv(path, dtype={"area_fips": str, "industry_code": str})
    df = df[(df.own_code == 5) & df.area_fips.str.fullmatch(r"\d{5}") & ~df.area_fips.str.endswith("000")]
    df = df[KEEP].copy()
    # BLS suppresses small cells (disclosure_code "N") and stores 0; that is missing, not zero.
    suppressed = df.disclosure_code.eq("N")
    df.loc[suppressed, ["annual_avg_estabs", "annual_avg_emplvl", "total_annual_wages", "avg_annual_pay"]] = np.nan
    return df.drop(columns="disclosure_code")


def last_complete_year() -> int:
    """Latest year whose ANNUAL file can be trusted. BLS posts a preliminary annual file well before it is complete
    (Oct 2026: the 2025 file summed to 7.45M construction jobs vs a 7.77M Q1-Q4 average), so a year is only used
    once the following year's Q4 is published. Until then forecasting.py estimates it from the quarters."""
    q = pd.read_parquet(ROOT / "data" / "qcew_quarterly.parquet")
    return int(q[q.qtr == 4].year.max()) - 1


if __name__ == "__main__":
    frames = [f for y in range(2014, last_complete_year() + 1) for i in {**SECTORS, **EXTRA} if (f := fetch(y, i)) is not None]
    q = pd.concat(frames, ignore_index=True)
    q.to_parquet(ROOT / "data" / "qcew.parquet")
    print(q.shape, q.year.min(), q.year.max(), q.area_fips.nunique(), "areas")
    print(q[q.industry_code == "23"].groupby("year").annual_avg_emplvl.sum().tail(5))
