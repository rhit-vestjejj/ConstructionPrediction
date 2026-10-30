# Data center event study

Part 2 of ConstructionPrediction: is there a tradeable stock signal when public contractors win data center
construction work?

**Short answer: no.** See [`outputs/REPORT.md`](outputs/REPORT.md) for the numbers.

## Run

```bash
cd event_study
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py                 # whole pipeline, about 1 min with a warm cache
.venv/bin/python run.py study           # one step: events | study | builders | report
.venv/bin/python run.py --refresh       # re-download prices (Yahoo) and filings (SEC)
```

Set `SEC_USER_AGENT="your-name your@email"` to use your own contact for SEC requests.

## What it does

| Step | Script | Question |
|---|---|---|
| `events` | `src/build_events.py` | Merges the researched contractor award releases into `data/dc_events.csv` |
| `study` | `src/dc_announcement_event_study.py` | Do stocks move when a contractor announces a data center award? How does that compare with earnings-day moves? |
| `builders` | `src/builder_backtest.py` | If you predicted where a data center goes and who builds it, would the builder's stock have paid? |
| `report` | `src/report.py` | Writes `outputs/REPORT.md` and prints a summary |

## Files

| Path | What |
|---|---|
| `data/research/` | Raw award-release research, one row per verified press release with its source URL |
| `data/dc_events.csv` | Merged award events (32), with release time, deal value, flags and source |
| `data/dc_events_dropped.csv` | Releases found but excluded, with the reason |
| `data/builders/` | 90 large projects from part 1 and the builders named for each, with sources |
| `data/earnings_canada.csv`, `data/revenue_manual.csv` | Hand-checked inputs for the TSX-listed companies |
| `outputs/results/` | Per-event CARs, earnings reactions and summary JSONs |
| `outputs/charts/` | `car_chart.png` and `builder_car_chart.png` |

## Method

- **Model:** a market model estimated over days -250 to -30. Recent listings start at their first available
  price, with at least 120 days required.
- **Benchmarks:** PKB for US stocks, XIU.TO for Canadian stocks and XLI as a robustness check. Builders listed
  elsewhere are measured against their local market.
- **Day 0:** the first session that could trade on the news. Releases at 16:00 ET or later roll to the next
  session.
- **Earnings:** awards within 3 trading days of an earnings release are excluded. US earnings release times come
  from SEC 8-K item 2.02 acceptance timestamps.
- **Trailing revenue:** from SEC XBRL company facts.

Every event, date and deal value was checked against a source. Anything that couldn't be verified was left out
or left blank.

## Limits

- **Small samples:** 21 award releases are usable, and only 6 builder events.
- **Mirror timestamps:** most release times come from wire mirrors, because Business Wire and GlobeNewswire block
  automated fetches.
- **No price history:** NV5 (5 events) was delisted and Yahoo has no prices for it.
- **Undisclosed builders:** builder research is limited to what is public, and most data center general
  contractors are private or never named.
