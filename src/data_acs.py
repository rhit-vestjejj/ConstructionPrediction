"""ACS 5-year county estimates (population, income, housing) for vintages 2010-2023. Needs CENSUS_API_KEY (env var or .env)."""
import json
import os

import pandas as pd
import requests

from geo import ROOT

VARS = {"B01003_001E": "pop", "B19013_001E": "med_income", "B25001_001E": "housing_units", "B25077_001E": "med_home_value",
        "B25103_001E": "med_re_taxes"}  # median real estate taxes paid (owner-occupied)


def key() -> str:
    k = os.environ.get("CENSUS_API_KEY")
    if not k and (ROOT / ".env").exists():
        k = dict(l.strip().split("=", 1) for l in (ROOT / ".env").read_text().splitlines() if "=" in l).get("CENSUS_API_KEY")
    if not k:
        raise SystemExit("Set CENSUS_API_KEY (env var or .env)")
    return k


def fetch(year: int) -> pd.DataFrame | None:
    path = ROOT / "data" / "raw" / "acs" / f"{year}.json"
    if not path.exists():
        r = requests.get(f"https://api.census.gov/data/{year}/acs/acs5", params={"get": ",".join(VARS), "for": "county:*", "key": key()}, timeout=120)
        if r.status_code != 200:
            print(f"{year}: HTTP {r.status_code} {r.text[:80]!r}")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    rows = json.loads(path.read_text())
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["fips"] = df.state + df.county
    out = df[["fips"] + list(VARS)].rename(columns=VARS)
    for c in VARS.values():
        out[c] = pd.to_numeric(out[c], errors="coerce").where(lambda s: s > 0)  # negative sentinels (-666666666) are missing
    out["vintage"] = year
    return out


if __name__ == "__main__":
    import datetime
    # 5-year vintage Y is released in December of Y+1; unreleased years simply fail to fetch
    frames = [f for y in range(2010, datetime.date.today().year) if (f := fetch(y)) is not None]
    acs = pd.concat(frames, ignore_index=True)
    acs.to_parquet(ROOT / "data" / "acs.parquet")
    print(acs.shape, sorted(acs.vintage.unique()), acs.fips.nunique(), "counties")
    print(acs[acs.vintage == 2023].describe().loc[["count", "50%"]].round(0).to_string())
