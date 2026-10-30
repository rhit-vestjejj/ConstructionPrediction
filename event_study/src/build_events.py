"""Merge the three research sweeps (data/research/events_*.csv) into data/dc_events.csv.

Each row is a press release that was opened and checked; duplicates across sweeps are collapsed.
Rows that are not construction/engineering awards are dropped here, with the reason recorded.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DROP = {  # (ticker, date): reason
    ("J", "2026-09-30"): "software-as-a-service agreement (digital twin), not a construction award",
    ("WLDN", "2024-12-12"): "utility energy-efficiency program, not a data center construction award",
    ("AMRC", "2025-09-09"): "development partnership with final agreements pending, not a contract award",
}
BUNDLED = {"ORN", "CTRI"}  # multi-project award releases where data center is one item
BORDERLINE = {  # kept in the main sample, flagged for robustness
    ("EFX.TO", "2026-10-01"): "equipment maker (packaged power units), not a site contractor",
    ("EFX.TO", "2026-10-13"): "equipment maker (packaged power units), not a site contractor",
    ("STN.TO", "2025-05-15"): "design/engineering selection; newsroom only, time unknown",
    ("ACM", "2025-12-02"): "non-US project; AECOM website only, time unknown",
    ("CTRI", "2025-10-01"): "data center item is minor turbine refurbishment",
}
# Deal values not in USD: (value, currency). Aecon reports in CAD.
NON_USD_DEALS = {("ARE.TO", "2026-07-02"): (1.7e9, "CAD")}


def main():
    frames = []
    for f in sorted((DATA / "research").glob("events_*.csv")):
        d = pd.read_csv(f, dtype=str)
        d["sweep"] = f.stem[-1]
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["n_text"] = raw.notna().sum(axis=1)

    # keep the most complete row per release, remember which sweeps found it
    sweeps = raw.groupby(["ticker", "release_date"]).sweep.agg("".join)
    ev = (raw.sort_values("n_text", ascending=False)
             .drop_duplicates(["ticker", "release_date"])
             .drop(columns=["sweep", "n_text"])
             .join(sweeps, on=["ticker", "release_date"])
             .rename(columns={"sweep": "found_by_sweeps"}))

    key = list(zip(ev.ticker, ev.release_date))
    ev["dropped_reason"] = [DROP.get(k, "") for k in key]
    ev["release_type"] = ["bundled" if t in BUNDLED or t == "SLND" else "standalone" for t in ev.ticker]
    ev["borderline_reason"] = [BORDERLINE.get(k, "") for k in key]
    ev["deal_value"] = pd.to_numeric(ev.deal_value_usd, errors="coerce")
    ev["deal_currency"] = ["USD" if pd.notna(v) else "" for v in ev.deal_value]
    for k, (v, cur) in NON_USD_DEALS.items():
        m = (ev.ticker == k[0]) & (ev.release_date == k[1])
        ev.loc[m, ["deal_value", "deal_currency"]] = [v, cur]
    ev = ev.drop(columns=["deal_value_usd"]).sort_values(["release_date", "ticker"])

    dropped = ev[ev.dropped_reason != ""]
    ev = ev[ev.dropped_reason == ""].drop(columns=["dropped_reason"])
    cols = ["ticker", "company", "release_date", "release_time_et", "timing", "headline", "deal_value",
            "deal_currency", "deal_value_text", "release_type", "borderline_reason", "other_same_day_news",
            "source_url", "notes", "found_by_sweeps"]
    ev[cols].to_csv(DATA / "dc_events.csv", index=False)
    dropped.to_csv(DATA / "dc_events_dropped.csv", index=False)
    print(f"{len(ev)} events kept, {len(dropped)} dropped")


if __name__ == "__main__":
    main()
