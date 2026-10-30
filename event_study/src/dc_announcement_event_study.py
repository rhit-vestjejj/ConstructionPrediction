"""Event study: do public contractors earn abnormal returns when they announce data center awards?

Events live in data/dc_events.csv (one row per verified press release, with source URL).
Earnings release dates/times come from SEC 8-K item 2.02 acceptance timestamps (US) or
data/earnings_canada.csv (TSX names). Trailing revenue comes from SEC XBRL company facts
or data/revenue_manual.csv.

Run from the event_study/ folder:  .venv/bin/python src/dc_announcement_event_study.py
"""
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*inplace keyword.*")  # yfinance vs pandas 3

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, CHARTS = ROOT / "data", ROOT / "outputs" / "results", ROOT / "outputs" / "charts"
CACHE = DATA / "cache"
SEC_HEADERS = {"User-Agent": os.environ.get("SEC_USER_AGENT", "ConstructionPrediction research vestjejj@rose-hulman.edu")}

US_BENCH, CA_BENCH, ROBUST_BENCH = "PKB", "XIU.TO", "XLI"
EST_WIN = (-250, -30)        # market-model estimation window, trading days relative to day0
MIN_EST_OBS = 120
CAR_WINDOWS = [(-1, 1), (0, 1), (0, 5), (0, 20), (-10, -1)]
PLOT_WIN = (-10, 20)
EARN_EXCLUDE_DAYS = 3        # drop awards within this many trading days of an earnings release
LARGE_DEAL = 0.05            # "large" = deal value above 5% of trailing revenue
N_EARNINGS_Q = 8
PRICE_START, PRICE_END = "2021-06-01", "2026-10-31"


def is_canadian(ticker):
    return ticker.endswith(".TO")


# ---------- data ----------

def load_prices(tickers, cache_name="prices.parquet"):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / cache_name
    if path.exists():
        px = pd.read_parquet(path)
        if set(tickers) <= set(px.columns):
            return px[sorted(tickers)]
    raw = yf.download(sorted(tickers), start=PRICE_START, end=PRICE_END,
                      auto_adjust=True, progress=False, threads=True)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.to_parquet(path)
    return px


def sec_json(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "_".join(url.split("/")[-2:])  # submissions/ and companyfacts/ share filenames
    if path.exists():
        return json.loads(path.read_text())
    time.sleep(0.15)  # SEC fair-access limit
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    path.write_text(r.text)
    return r.json()


CIK_OVERRIDES = {"NVEE": 1532961}  # NV5 Global, delisted Aug 2025 so absent from the ticker map


def cik_map():
    m = sec_json("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"]: int(v["cik_str"]) for v in m.values()} | CIK_OVERRIDES


def earnings_releases(ticker, ciks):
    """Earnings release timestamps (ET) for one ticker."""
    if is_canadian(ticker):
        path = DATA / "earnings_canada.csv"
        if not path.exists():
            return pd.DataFrame(columns=["release_date", "timing"])
        df = pd.read_csv(path)
        df = df[df.ticker == ticker].copy()
        df["timing"] = df.release_time_et.map(timing_from_time)
        df["release_date"] = pd.to_datetime(df.release_date)
        return df[["release_date", "timing"]]
    sub = sec_json(f"https://data.sec.gov/submissions/CIK{ciks[ticker]:010d}.json")["filings"]["recent"]
    rows = []
    for form, items, acc in zip(sub["form"], sub["items"], sub["acceptanceDateTime"]):
        if form == "8-K" and "2.02" in items.split(","):
            ts = pd.Timestamp(acc).tz_convert("America/New_York")
            rows.append({"release_date": ts.tz_localize(None).normalize(),
                         "timing": timing_from_time(ts.strftime("%H:%M"))})
    return pd.DataFrame(rows).drop_duplicates("release_date")


REV_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "ContractsRevenue",
            "RevenueFromContractWithCustomerIncludingAssessedTax"]


def revenue_facts(ticker, ciks):
    facts = sec_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{ciks[ticker]:010d}.json").get("facts", {}).get("us-gaap", {})
    frames = []
    for tag in REV_TAGS:
        if tag in facts and "USD" in facts[tag]["units"]:
            df = pd.DataFrame(facts[tag]["units"]["USD"])
            if "start" in df:
                frames.append(df.dropna(subset=["start"]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    for c in ["start", "end", "filed"]:
        df[c] = pd.to_datetime(df[c])
    df["days"] = (df.end - df.start).dt.days
    # prefer the tag that is current for each period: keep the largest reported value per (start, end, filed)
    return df.sort_values("val").drop_duplicates(["start", "end", "filed"], keep="last")


def ttm_revenue(facts, asof):
    """Trailing-12-month revenue known before `asof`: last FY + current YTD - prior-year YTD."""
    known = facts[facts.filed < asof]
    fy = known[known.days.between(350, 380)].sort_values(["end", "filed"])
    if fy.empty:
        return np.nan
    fy_row = fy.iloc[-1]
    ytd = known[(known.end > fy_row.end) & known.days.between(80, 290)].sort_values(["end", "filed"])
    if ytd.empty:
        return float(fy_row.val)
    cur = ytd.iloc[-1]
    prior = known[(abs((known.end - (cur.end - pd.DateOffset(years=1))).dt.days) <= 10)
                  & (abs(known.days - cur.days) <= 10)].sort_values("filed")
    if prior.empty:
        return float(fy_row.val)
    return float(fy_row.val + cur.val - prior.iloc[-1].val)


# ---------- event mechanics ----------

def timing_from_time(t):
    if not isinstance(t, str) or not t.strip():
        return "unknown"
    h, m = map(int, t.strip()[:5].split(":"))
    mins = h * 60 + m
    return "pre-market" if mins < 570 else ("after-close" if mins >= 960 else "intraday")


def day0_index(dates, release_date, timing):
    """Position of the first session that could trade on the news."""
    pos = dates.searchsorted(release_date)  # first session on/after release date
    if pos < len(dates) and dates[pos] == release_date and timing == "after-close":
        pos += 1
    return pos if pos < len(dates) else None


def abnormal_returns(stock_px, bench_px, release_date, timing, win=PLOT_WIN, est_win=EST_WIN):
    """Market-model abnormal returns indexed by event day, or None if the window is incomplete."""
    df = pd.concat([stock_px, bench_px], axis=1, keys=["s", "m"]).dropna().pct_change().dropna()
    p0 = day0_index(df.index, pd.Timestamp(release_date), timing)
    if p0 is None or p0 + 1 >= len(df) or p0 + win[0] < 0:  # recent events keep the short windows
        return None, None
    # recent listings: start estimation at the first available return, still requiring MIN_EST_OBS days
    est = df.iloc[max(0, p0 + est_win[0]): max(0, p0 + est_win[1] + 1)]
    if len(est) < MIN_EST_OBS:
        return None, None
    beta, alpha = np.polyfit(est.m, est.s, 1)
    w = df.iloc[p0 + win[0]: p0 + win[1] + 1]
    ar = w.s - (alpha + beta * w.m)
    ar.index = range(win[0], win[0] + len(ar))
    return ar, df.index[p0]


def car(ar, a, b):
    if ar.index.min() > a or ar.index.max() < b:
        return np.nan
    return float(ar.loc[a:b].sum())


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0.0)
    return float(t), float(p)


# ---------- analysis ----------

def run_study(events, px, bench_for):
    rows, paths = [], {}
    for i, e in events.iterrows():
        b = bench_for(e.ticker)
        if e.ticker not in px or px[e.ticker].notna().sum() == 0:
            continue
        ar, d0 = abnormal_returns(px[e.ticker], px[b], e.release_date, e.timing)
        if ar is None:
            continue
        r = {"event_id": i, "day0": d0.date(), "benchmark": b}
        for a, z in CAR_WINDOWS:
            r[f"CAR[{a},{z}]"] = car(ar, a, z)
        rows.append(r)
        paths[i] = ar.cumsum()
    return pd.DataFrame(rows).set_index("event_id"), pd.DataFrame(paths)


def summarize(cars):
    out = {}
    for a, z in CAR_WINDOWS:
        col = f"CAR[{a},{z}]"
        t, p = tstat(cars[col])
        x = cars[col].dropna()
        out[col] = {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
                    "t": t, "p": p, "pct_positive": float((x > 0).mean())}
    return out


def mark_earnings_overlap(events, px, earnings):
    """Trading-day distance from each event's day0 to the nearest earnings day0."""
    dist = []
    for _, e in events.iterrows():
        dates = px[e.ticker].dropna().index
        p0 = day0_index(dates, pd.Timestamp(e.release_date), e.timing)
        er = earnings.get(e.ticker)
        if p0 is None or er is None or er.empty:
            dist.append(np.nan)
            continue
        eps = [day0_index(dates, r.release_date, r.timing) for r in er.itertuples()]
        eps = [q for q in eps if q is not None]
        dist.append(min(abs(q - p0) for q in eps) if eps else np.nan)
    return pd.Series(dist, index=events.index)


def earnings_reactions(tickers, px, earnings, bench_for):
    rows = []
    for t in tickers:
        er = earnings.get(t)
        if er is None or er.empty:
            continue
        er = er.sort_values("release_date")
        er = er[er.release_date <= px.index.max() - pd.Timedelta(days=5)].tail(N_EARNINGS_Q)
        for r in er.itertuples():
            ar, d0 = abnormal_returns(px[t], px[bench_for(t)], r.release_date, r.timing)
            if ar is not None:
                rows.append({"ticker": t, "release_date": r.release_date.date(), "day0": d0.date(),
                             "CAR[0,1]": car(ar, 0, 1)})
    return pd.DataFrame(rows)


def deal_size_tests(df):
    d = df.dropna(subset=["deal_to_revenue", "CAR[0,1]"])
    d = d[d.deal_to_revenue > 0]
    res = {"n_with_deal_value": int(len(d))}
    if len(d) >= 5:
        X = sm.add_constant(np.log(d.deal_to_revenue))
        fit = sm.OLS(d["CAR[0,1]"], X).fit()
        hc1 = sm.OLS(d["CAR[0,1]"], X).fit(cov_type="HC1")  # unreliable at small n; classical p is the headline
        res["regression"] = {"n": int(len(d)), "slope": float(fit.params.iloc[1]), "slope_t": float(fit.tvalues.iloc[1]),
                             "slope_p": float(fit.pvalues.iloc[1]), "slope_p_hc1": float(hc1.pvalues.iloc[1]),
                             "intercept": float(fit.params.iloc[0]), "r2": float(fit.rsquared)}
    groups = {"large (>5% rev)": df[df.deal_to_revenue > LARGE_DEAL],
              "small (<=5% rev)": df[df.deal_to_revenue <= LARGE_DEAL],
              "undisclosed": df[df.deal_to_revenue.isna()]}
    for k, g in groups.items():
        t, p = tstat(g["CAR[0,1]"])
        res[k] = {"n": int(len(g)), "mean_CAR[0,1]": float(g["CAR[0,1]"].mean()) if len(g) else np.nan, "t": t, "p": p}
    lg, sm_ = groups["large (>5% rev)"]["CAR[0,1]"], groups["small (<=5% rev)"]["CAR[0,1]"]
    if len(lg) >= 2 and len(sm_) >= 2:
        t, p = stats.ttest_ind(lg, sm_, equal_var=False)
        res["large_minus_small"] = {"diff": float(lg.mean() - sm_.mean()), "welch_t": float(t), "p": float(p)}
    return res


# ---------- chart ----------

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(paths_main, paths_xli, award_abs, earn_abs, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.7, 1]},
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    for paths, color, label, band in [(paths_main, BLUE, "PKB / XIU.TO benchmark (95% CI shaded)", True),
                                      (paths_xli, ORANGE, "XLI benchmark", False)]:
        m = paths.mean(axis=1) * 100
        if band:
            se = paths.std(axis=1, ddof=1) / np.sqrt(paths.count(axis=1)) * 100
            ax1.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.12, linewidth=0)
        ax1.plot(m.index, m, color=color, linewidth=2, label=label)
    ax1.axvline(0, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax1.axhline(0, color=INK2, linewidth=0.8)
    ax1.set_xlabel("Trading days relative to first tradeable session (day 0)", color=INK2, fontsize=9)
    ax1.set_ylabel("Mean CAR from day -10 (%)", color=INK2, fontsize=9)
    ax1.set_title(f"Data center award announcements (n={paths_main.shape[1]})",
                  loc="left", fontsize=11, color=INK)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax1.set_xlim(PLOT_WIN[0], PLOT_WIN[1] + 2)

    rng = np.random.default_rng(0)
    for k, (vals, color, name) in enumerate([(award_abs, BLUE, "Award\nannouncements"),
                                             (earn_abs, ORANGE, "Earnings\nreleases")]):
        v = np.asarray(vals) * 100
        ax2.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=22, color=color, alpha=0.55,
                    edgecolors=SURFACE, linewidths=1)
        med = np.median(v)
        ax2.hlines(med, k - 0.3, k + 0.3, color=INK, linewidth=2)
        ax2.annotate(f"median {med:.1f}%\nn={len(v)}", (k + 0.32, med), va="center", fontsize=9, color=INK)
    ax2.set_xticks([0, 1], ["Award\nannouncements", "Earnings\nreleases"])
    ax2.set_xlim(-0.5, 1.9)
    ax2.set_ylabel("|CAR[0,+1]| (%)", color=INK2, fontsize=9)
    ax2.set_title("Size of 2-day reaction, same tickers", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------- main ----------

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(DATA / "dc_events.csv", dtype={"release_time_et": str})
    events["release_date"] = pd.to_datetime(events.release_date)
    events["timing"] = [t if t in ("pre-market", "intraday", "after-close") else timing_from_time(tm)
                        for t, tm in zip(events.timing, events.release_time_et)]
    tickers = sorted(events.ticker.unique())
    px = load_prices(tickers + [US_BENCH, CA_BENCH, ROBUST_BENCH])

    ciks = cik_map()
    earnings = {t: earnings_releases(t, ciks) for t in tickers}

    # trailing revenue at announcement (USD from SEC XBRL; manual file for TSX names, in the deal's currency)
    manual = pd.read_csv(DATA / "revenue_manual.csv", parse_dates=["release_date"])
    manual = manual.set_index(["ticker", "release_date"]).ttm_revenue
    rev = []
    for i, e in events.iterrows():
        if pd.isna(e.deal_value):
            rev.append(np.nan)
        elif (e.ticker, e.release_date) in manual.index:
            rev.append(float(manual.loc[(e.ticker, e.release_date)]))
        elif e.ticker in ciks:
            rev.append(ttm_revenue(revenue_facts(e.ticker, ciks), e.release_date))
        else:
            rev.append(np.nan)
    events["ttm_revenue"] = rev
    events["deal_to_revenue"] = events.deal_value / events.ttm_revenue

    events["earnings_gap_days"] = mark_earnings_overlap(events, px, earnings)
    events["excluded_earnings"] = events.earnings_gap_days <= EARN_EXCLUDE_DAYS
    news = events.other_same_day_news.fillna("").str.strip()
    events["same_day_news_flag"] = news.ne("") & ~news.str.lower().str.startswith("none")

    main_bench = lambda t: CA_BENCH if is_canadian(t) else US_BENCH
    clean = events[~events.excluded_earnings]
    cars_main, paths_main = run_study(clean, px, main_bench)
    cars_xli, paths_xli = run_study(clean, px, lambda t: ROBUST_BENCH)

    full = events.join(cars_main.add_suffix("_main")).join(cars_xli.drop(columns=["day0"]).add_suffix("_xli"))
    full.rename(columns={"day0_main": "day0"}, inplace=True)
    full.to_csv(RESULTS / "dc_events_with_cars.csv", index_label="event_id")

    analysis = clean.join(cars_main, how="inner")
    car_cols = [f"CAR[{a},{z}]" for a, z in CAR_WINDOWS]
    no_news = analysis[~analysis.same_day_news_flag]
    core = analysis[(analysis.release_type == "standalone") & analysis.borderline_reason.isna()]

    earn = earnings_reactions(sorted(analysis.ticker.unique()), px, earnings, main_bench)
    earn.to_csv(RESULTS / "earnings_reactions.csv", index=False)
    award_abs, earn_abs = analysis["CAR[0,1]"].abs(), earn["CAR[0,1]"].abs()
    mw = stats.mannwhitneyu(earn_abs, award_abs, alternative="greater") if len(earn_abs) and len(award_abs) else None

    summary = {
        "n_events_listed": int(len(events)),
        "n_excluded_earnings_overlap": int(events.excluded_earnings.sum()),
        "n_analyzed": int(len(analysis)),
        "main_benchmark": summarize(cars_main),
        "xli_benchmark": summarize(cars_xli),
        "main_excluding_same_day_news": summarize(no_news[car_cols]),
        "main_standalone_non_borderline": summarize(core[car_cols]),
        "deal_size": deal_size_tests(analysis),
        "earnings_comparison": {
            "n_earnings": int(len(earn_abs)), "n_awards": int(len(award_abs)),
            "mean_abs_CAR01_awards": float(award_abs.mean()), "median_abs_CAR01_awards": float(award_abs.median()),
            "mean_abs_CAR01_earnings": float(earn_abs.mean()), "median_abs_CAR01_earnings": float(earn_abs.median()),
            "ratio_median_earnings_to_awards": float(earn_abs.median() / award_abs.median()),
            "mannwhitney_earnings_gt_awards_p": float(mw.pvalue) if mw else np.nan,
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    plot(paths_main[analysis.index], paths_xli[paths_xli.columns.intersection(analysis.index)],
         award_abs, earn_abs, CHARTS / "car_chart.png")
    return summary


if __name__ == "__main__":
    main()
