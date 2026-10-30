"""Backtest: if you correctly predicted which public contractor builds a data center somewhere, did its stock pay?

Inputs: data/builders/builders_*.csv, one row per (project, named firm) from source-checked research.
Three tests on the eventual public builders:
  1. Project announced: CAR around the day the project itself was first announced (the location boom goes public).
  2. Builder named: CAR around the first date the firm was publicly named on the project.
  3. Perfect foresight: buy 60 / 120 trading days before the builder is named, sell 5 days after.

Run from event_study/:  .venv/bin/python src/builder_backtest.py
"""
import json

import matplotlib
import numpy as np
import pandas as pd

from dc_announcement_event_study import (BLUE, CHARTS, DATA, GRID, INK, INK2, ORANGE, RESULTS, SURFACE,
                                         abnormal_returns, car, load_prices, tstat)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EU_SUFFIXES = (".DE", ".MC", ".ST", ".PA", ".AS", ".L", ".SW", ".MI", ".CO", ".HE", ".OL", ".VI", ".BR")
NAMED_WIN, NAMED_EST = (-120, 20), (-370, -130)  # estimation ends before the foresight holding period
ANNOUNCE_WIN = (-10, 20)


def bench_for(ticker):
    if ticker.endswith(".TO"):
        return "XIU.TO"
    if ticker.endswith(EU_SUFFIXES):
        return "EXV6.DE"  # iShares STOXX Europe 600 Construction & Materials
    if ticker.endswith(".KS"):
        return "^KS11"
    if ticker.endswith(".CA"):
        return "^CASE30"
    return "PKB"


def exact_date(s):
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return d if pd.notna(d) else pd.NaT


def load_builders():
    frames = [pd.read_csv(f, dtype=str) for f in sorted((DATA / "builders").glob("builders_*.csv"))]
    b = pd.concat(frames, ignore_index=True)
    b["public_ticker"] = b.public_ticker.fillna("").str.strip().str.upper().replace({"NAN": ""})
    # listed parent not confirmed by the research (flagged "verify"): keep for coverage, not as a stock event
    unverified = b.notes.fillna("").str.contains("verify", case=False) & b.public_parent.fillna("").str.contains(
        "verify|uncertain", case=False)
    b.loc[unverified | (b.firm.fillna("").str.contains("Linxon")), "public_ticker"] = ""
    b["announce_day"] = b.announce_date.map(exact_date)
    b["named_day"] = b.first_named_date.map(exact_date)
    return b


def coverage(b):
    found = b[~b.firm.fillna("").str.lower().str.startswith("not found")]
    by_proj = b.groupby("project")
    proj_found = found.project.nunique()
    proj_public = found[found.public_ticker != ""].project.nunique()
    gc = found[found.role.fillna("").str.contains("GC|design|EPC", case=False)]
    return {
        "projects_researched": int(by_proj.ngroups),
        "projects_with_any_builder_named": int(proj_found),
        "projects_with_public_builder": int(proj_public),
        "projects_where_GC_is_public": int(gc[gc.public_ticker != ""].project.nunique()),
        "projects_where_GC_named": int(gc.project.nunique()),
        "public_firm_counts": found[found.public_ticker != ""].public_ticker.value_counts().to_dict(),
        "private_firm_counts": found[found.public_ticker == ""].firm.value_counts().head(25).to_dict(),
    }


def window_stats(df, cols):
    out = {}
    for c in cols:
        x = df[c].dropna()
        t, p = tstat(x)
        out[c] = {"n": int(len(x)), "n_firms": int(df.loc[x.index, "public_ticker"].nunique()),
                  "mean": float(x.mean()) if len(x) else np.nan,
                  "median": float(x.median()) if len(x) else np.nan,
                  "t": t, "p": p, "pct_positive": float((x > 0).mean()) if len(x) else np.nan}
    return out


def main():
    b = load_builders()
    cov = coverage(b)
    pub = b[b.public_ticker != ""].copy()
    tickers = sorted(pub.public_ticker.unique())
    px = load_prices(sorted(set(tickers) | {bench_for(t) for t in tickers}), cache_name="prices_builders.parquet")

    rows, named_paths, ann_paths = [], {}, {}
    # one event per (ticker, day): several projects can name the same firm on the same day
    for kind, col, win, est in [("named", "named_day", NAMED_WIN, NAMED_EST),
                                ("announced", "announce_day", ANNOUNCE_WIN, (-250, -30))]:
        ev = pub.dropna(subset=[col]).drop_duplicates(["public_ticker", col])
        for i, e in ev.iterrows():
            t = e.public_ticker
            if t not in px or px[t].notna().sum() == 0:
                continue
            ar, d0 = abnormal_returns(px[t], px[bench_for(t)], e[col], "unknown", win=win, est_win=est)
            if ar is None:
                continue
            r = {"kind": kind, "project": e.project, "firm": e.firm, "role": e.role, "public_ticker": t,
                 "event_date": e[col].date(), "day0": d0.date(), "source_url": e.source_url}
            for a, z in [(-10, -1), (0, 1), (0, 5), (0, 20), (-60, 5), (-120, 5)]:
                r[f"CAR[{a},{z}]"] = car(ar, a, z)
            rows.append(r)
            # zero at day -1 so a pre-event run-up shows as a rising line into day 0
            (named_paths if kind == "named" else ann_paths)[len(rows) - 1] = ar.cumsum() - ar.loc[:-1].sum()
    res = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "builder_events.csv", index=False)

    named = res[res.kind == "named"] if len(res) else res
    ann = res[res.kind == "announced"] if len(res) else res
    summary = {
        "coverage": cov,
        "builder_named": window_stats(named, ["CAR[-10,-1]", "CAR[0,1]", "CAR[0,5]", "CAR[0,20]"]) if len(named) else {},
        "perfect_foresight": window_stats(named, ["CAR[-60,5]", "CAR[-120,5]"]) if len(named) else {},
        "project_announced": window_stats(ann, ["CAR[-10,-1]", "CAR[0,1]", "CAR[0,5]", "CAR[0,20]"]) if len(ann) else {},
    }
    (RESULTS / "builder_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if named_paths:
        plot(pd.DataFrame(named_paths), pd.DataFrame(ann_paths), CHARTS / "builder_car_chart.png")
    return summary


def plot(named, ann, out):
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for paths, color, label, band in [(named, BLUE, f"Builder publicly named (n={named.shape[1]}, 95% CI shaded)", True),
                                      (ann, ORANGE, f"Project first announced (n={ann.shape[1]})", False)]:
        if paths.empty:
            continue
        m = paths.mean(axis=1) * 100
        if band:
            se = paths.std(axis=1, ddof=1) / np.sqrt(paths.count(axis=1)) * 100
            ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.12, linewidth=0)
        ax.plot(m.index, m, color=color, linewidth=2, label=label)
    ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("Trading days relative to event", color=INK2, fontsize=9)
    ax.set_ylabel("Mean CAR, zeroed at day -1 (%)", color=INK2, fontsize=9)
    ax.set_title("Public contractors on data center projects: abnormal return around each event",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
