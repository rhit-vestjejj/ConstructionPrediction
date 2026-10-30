"""BLS QCEW quarterly county construction employment (private), 2014Q1 onward, for the 'latest quarter' signal.
Quarterly employment = mean of the quarter's three monthly employment levels. Suppressed cells -> missing."""
import numpy as np
import pandas as pd
import requests

from geo import ROOT

URL = "https://data.bls.gov/cew/data/api/{year}/{q}/industry/{ind}.csv"
UA = {"User-Agent": "ConstructionPrediction/0.1 (personal research)"}
INDUSTRIES = ("23", "2362", "237")


def fetch(year, q, ind):
    path = ROOT / "data" / "raw" / "qcew_q" / f"{year}q{q}_{ind}.csv"
    if not path.exists():
        r = requests.get(URL.format(year=year, q=q, ind=ind), headers=UA, timeout=120)
        if r.status_code != 200:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    d = pd.read_csv(path, dtype={"area_fips": str, "industry_code": str},
                    usecols=["area_fips", "own_code", "industry_code", "year", "qtr", "disclosure_code", "month1_emplvl", "month2_emplvl", "month3_emplvl"])
    d = d[(d.own_code == 5) & d.area_fips.str.fullmatch(r"\d{5}") & ~d.area_fips.str.endswith("000")]
    emp = d[["month1_emplvl", "month2_emplvl", "month3_emplvl"]].mean(axis=1)
    return pd.DataFrame({"fips": d.area_fips, "industry_code": d.industry_code, "year": d.year, "qtr": d.qtr,
                         "emp": np.where(d.disclosure_code.eq("N"), np.nan, emp)})


if __name__ == "__main__":
    frames = []
    import datetime
    for y in range(2014, datetime.date.today().year + 1):
        for q in (1, 2, 3, 4):
            for ind in INDUSTRIES:
                f = fetch(y, q, ind)
                if f is not None:
                    frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(ROOT / "data" / "qcew_quarterly.parquet")
    last = out.groupby(["year", "qtr"]).size().index.max()
    print(out.shape, "| quarters:", out.groupby(["year", "qtr"]).ngroups, "| latest:", last)
