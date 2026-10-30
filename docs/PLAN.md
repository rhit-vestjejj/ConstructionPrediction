# ConstructionPrediction: status and plan

Goal: an interactive US map that forecasts where construction will happen, compares model families honestly, and
shows its own track record. This file records what exists, what the data supports, and what is still open.

## What the models predict

| View | Target | Labels | Evaluation |
|---|---|---|---|
| County construction (default "surge") | county in the top 10% of absolute increase, or a 25%+ jump vs its own 3-year average | BLS QCEW jobs (2014-2024, NAICS 23 / 2362 / 237), Census permits (1990-2025) | fixed chronological split + yearly-retrained walk-forward; 12 / 24 / 36 months |
| Data centers (default) | county gets a NEW data center project next year (first public signal) | 915 dated projects compiled from news and public records by research agents + Epoch AI | walk-forward 2021-2025; model chosen on 2021-22, honest test 2023-25 |
| Data centers (propensity) | county hosts a data center today | OpenStreetMap sites | leave-one-region-out CV; emerging-location test on Epoch campuses |
| Metro permits (H3 cells, ~5 km2) | new $1M+ non-residential permit in a cell next year | Fort Worth, Charlotte/Mecklenburg, Columbus open permit data | train <= 2020, validate 2021-22, test 2023-24 |

## What was learned
- Retraining every year (walk-forward) beat a one-time fit in 11 of 12 target/horizon cases.
- Fresh signals (Q1 jobs, Jan-Aug permits of the forecast year) lift 12-month accuracy a lot; that part is a nowcast.
- Tree ensembles beat GraphSAGE and GraphSAGE+GRU everywhere; a plain neural net roughly ties at best.
- Learning-to-rank and tuning added little. Demographics, power, grid, climate, hazards, tax and political lean
  add small gains at most; county size and recent activity dominate.
- Data centers: the timed model is ~10x random but only slightly better than "counties with earlier projects";
  its edge is new markets (~5x random).

## Data caveats
- Jobs and permits are activity proxies, not project starts or dollars.
- Agent-compiled project data: ~80-90% right on county and year in spot checks; thin before 2021.
- Some features are static recent snapshots (power plants, roads, land, hazards): mild look-ahead in early years.
- Industrial/warehouse labels exist only in a few metros' permit portals.

## Open items
1. Scenario simulation: only for variables the models are trained on; national macro series have too few years to
   learn county-level effects, so it stays disabled.
2. More metros for the H3 model (Chicago, Austin, Seattle, Denver portals are known to work).
3. Industrial/manufacturing labels nationally (Clean Investment Monitor needs a free account).
4. Move the main environment to native arm64 Python (the conda one runs under Rosetta; XGBoost is pinned to 2.1.4).
5. Bootstrap uncertainty intervals.
