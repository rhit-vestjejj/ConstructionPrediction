"""Publicly announced construction projects per county (data centers so far): Epoch AI frontier campuses plus the
agent-compiled list (scratch research, 122 rows, each with a source URL). Used as a 'known pipeline' layer and as a
model input that only counts projects announced on or before the forecast year."""
import pandas as pd

from data_infra import ABBR_FIPS, norm_county
from geo import ROOT

AGENT_CSV = ROOT / "data" / "raw" / "projects_agent.csv"

if __name__ == "__main__":
    ep = pd.read_parquet(ROOT / "data" / "epoch_sites.parquet").dropna(subset=["fips"])
    a = pd.DataFrame({"name": ep.Name, "operator": ep.Owner.astype(str).str.split(" #").str[0], "fips": ep.fips, "year": ep.first_dated_milestone.dt.year,
                      "mw": ep.planned_peak_mw, "type": "data center", "source": "Epoch AI (CC-BY)"})
    g = pd.read_csv(AGENT_CSV)
    names = pd.read_parquet(ROOT / "data" / "panel.parquet").drop_duplicates("fips", keep="last")[["fips", "name"]]
    key2fips = dict(zip(names.fips.str[:2] + "|" + norm_county(names.name), names.fips))
    g["fips"] = (g.state.map(ABBR_FIPS).fillna("") + "|" + norm_county(g.county.fillna(""))).map(key2fips)
    yr = pd.to_datetime(g.announce_date.fillna(g.construction_start_date).astype(str).str[:10], errors="coerce", format="mixed").dt.year
    b = pd.DataFrame({"name": g.project_name, "operator": g.operator, "fips": g.fips, "year": yr, "mw": g.mw, "type": "data center", "source": g.source_url})
    print(f"agent list: {len(g)} rows, {b.fips.notna().sum()} matched to a county, {b.year.notna().sum()} dated")
    p = pd.concat([a, b], ignore_index=True).dropna(subset=["fips", "year"])
    p = p.drop_duplicates(subset=["fips", "operator", "year"])  # crude de-dup between the two sources
    p["year"] = p.year.astype(int)
    p.to_parquet(ROOT / "data" / "projects.parquet")
    print(len(p), "dated projects in", p.fips.nunique(), "counties; by year:", p.year.value_counts().sort_index().to_dict())
