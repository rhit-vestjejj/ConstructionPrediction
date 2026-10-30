"""Merge agent-compiled data center project lists (data/raw/dc_news/<sweep or region>/projects.csv) with the existing
Epoch + first-pass list into one dated, county-coded table: data/dc_projects.parquet.

Each project gets first_year = earliest of its announce_date / event_date (the first public signal).
Duplicates across sources are dropped on (county, normalised operator, first_year) and on (county, normalised name)."""
import glob
import re
import sys

import pandas as pd

from data_infra import ABBR_FIPS, norm_county
from geo import ROOT

NEWS_DIR = ROOT / "data" / "raw" / "dc_news"  # one folder per research sweep/region, each with projects.csv (+ verification.md)


def year_of(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"((?:19|20)\d\d)")[0], errors="coerce")


def county_fips(state: pd.Series, county: pd.Series) -> pd.Series:
    names = pd.read_parquet(ROOT / "data" / "panel.parquet").drop_duplicates("fips", keep="last")[["fips", "name"]]
    key2fips = dict(zip(names.fips.str[:2] + "|" + norm_county(names.name), names.fips))
    st = state.astype(str).str.strip().str.upper().map(ABBR_FIPS).fillna("")
    c = county.fillna("").astype(str).str.replace(r"\(.*?\)", "", regex=True).str.split(r"[/;,]| and ").str[0].str.strip()
    out = (st + "|" + norm_county(c)).map(key2fips)
    alt = (st + "|" + norm_county(c + " city")).map(key2fips)  # Virginia independent cities
    return out.fillna(alt)


def norm_op(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"[^a-z ]", "", regex=True).str.split().str[0].fillna("")


def main(src_dir=NEWS_DIR):
    frames = []
    for f in sorted(glob.glob(f"{src_dir}/**/projects.csv", recursive=True)):
        d = pd.read_csv(f, dtype=str, on_bad_lines="skip")
        d["region"] = "/".join(f.split("/")[len(str(src_dir).split("/")):-1])
        frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    a["first_year"] = pd.concat([year_of(a.get("announce_date", pd.Series(dtype=str))), year_of(a.get("event_date", pd.Series(dtype=str)))], axis=1).min(axis=1)
    a["fips"] = county_fips(a.state, a.county)
    print(f"agent rows {len(a)} from {a.region.nunique()} regions | with county FIPS {a.fips.notna().mean():.0%} | dated {a.first_year.notna().mean():.0%}")
    new = pd.DataFrame({"name": a.project_name, "operator": a.operator, "fips": a.fips, "year": a.first_year, "status": a.status, "event_type": a.event_type,
                        "mw": pd.to_numeric(a.mw, errors="coerce"), "source": a.source_url, "evidence": a.evidence, "region": a.region})
    old = pd.read_parquet(ROOT / "data" / "projects.parquet").rename(columns={"source": "source"})
    old = pd.DataFrame({"name": old.name, "operator": old.operator, "fips": old.fips, "year": old.year, "status": None, "event_type": "announced",
                        "mw": old.mw, "source": old.source, "evidence": "page", "region": "earlier"})
    p = pd.concat([old, new], ignore_index=True).dropna(subset=["fips", "year"])
    p["year"] = p.year.astype(int)
    p["op"], p["nm"] = norm_op(p.operator), p.name.fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    p = p.sort_values("year")  # keep the earliest record of a project
    p = p.drop_duplicates(["fips", "nm"]).drop_duplicates(["fips", "op", "year"])
    p = p.drop(columns=["op", "nm"])
    p.to_parquet(ROOT / "data" / "dc_projects.parquet")
    print(f"{len(p)} unique dated, county-coded projects in {p.fips.nunique()} counties")
    print("by first year:", p.year.value_counts().sort_index().to_dict())
    return p


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
