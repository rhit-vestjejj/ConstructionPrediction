# ConstructionPrediction

Interactive map of US construction forecasts: county-level models for construction jobs and housing permits, a data
center siting-propensity model, and cell-level (H3) permit forecasts for metros that publish permit data.
See `docs/PLAN.md` for the roadmap and what the data can and cannot support.

## Run the app

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python serve.py          # opens http://127.0.0.1:8000/forecast_app.html
```

The site is `outputs/forecast_app.html` plus `outputs/forecast_backtest.json`. The JSON file holds past-year
predictions and loads only when someone opens "Check past years". Opening the HTML file directly works for everything
except that view, which needs the files served (`serve.py` or any static host). URL presets work, e.g.
`forecast_app.html#cat=metro&metro=columbus` or `#cat=res&mode=backtest&origin=2019&sel=48201`.

The map is drawn on a canvas from county outlines pre-projected at build time (`src/albers_usa.py`, d3's AlbersUsa
frame), so a recolor, pan or zoom is a single repaint.

## Hosting (GitHub Pages)

`.github/workflows/pages.yml` publishes the site on every push to `main` that changes the built app: it copies
`outputs/forecast_app.html` (as `index.html`) and `outputs/forecast_backtest.json` to Pages. One-time setup: repo
Settings → Pages → Source: **GitHub Actions**. The site is then at https://rhit-vestjejj.github.io/ConstructionPrediction/.
Rebuild the app (`build_forecast_app.py`) and commit `outputs/` before pushing.

## Rebuild from scratch

A Census API key goes in `.env` as `CENSUS_API_KEY=...` (git-ignored). Then, from `src/`:

| Step | Script | Output |
|---|---|---|
| Housing permits (Census BPS) | `data.py` | `data/panel.parquet` |
| Construction jobs (BLS QCEW) | `data_qcew.py` | `data/qcew.parquet` |
| Data centers (OpenStreetMap) | `data_dc.py` | `data/datacenters.parquet` |
| Frontier AI campuses (Epoch AI) | `data_epoch.py` | `data/epoch_sites.parquet` |
| Demographics (ACS) | `data_acs.py` | `data/acs.parquet` |
| Power, grid, climate, hazards | `data_infra.py` | `data/infra_features.parquet` (expects files in `data/raw/infra/`) |
| Metro permits | `data_permits_metro.py` | `data/raw/permits/*.parquet` |
| Quarterly jobs (latest-quarter signal) | `data_qcew_quarterly.py` | `data/qcew_quarterly.parquet` |
| Elections, roads/rail, land value | `data_context.py` | `data/elections.parquet`, `data/context_static.parquet` |
| Data center projects (agent-compiled) | `data_dc_projects.py` | `data/dc_projects.parquet` |
| Timed data center model | `dc_timed.py` | `outputs/results/dc_timed_*` |
| Walk-forward / ranking / tuning | `walkforward.py [--reuse-configs]` | `outputs/results/walkforward_*` |
| Graph models (separate arm64 env) | `.venv-gnn/bin/python src/gnn.py` | `outputs/results/gnn_*` |
| County model comparison | `forecasting.py` | `outputs/results/metrics.*`, `predictions.parquet` |
| Data center propensity | `datacenter_model.py` | `data/datacenter_scores.parquet`, `outputs/results/datacenter_metrics.json` |
| Metro H3 cell model | `metro_cells.py` | `outputs/results/metro_metrics.json`, `metro_predictions.parquet` |
| App | `build_forecast_app.py` (`--template-only` re-renders from the cached payload in seconds) | `outputs/forecast_app.html`, `outputs/forecast_backtest.json` |

## Test

```bash
.venv/bin/python tests/e2e_app.py                                    # serves outputs/ and drives the app in Chrome
.venv/bin/python tests/e2e_app.py http://127.0.0.1:8000/forecast_app.html
```

## Notes
- XGBoost is pinned to 2.1.4: 3.x needs a newer OpenMP runtime than the conda install on the dev machine.
- All metrics shown in the app come from saved runs; model selection and ensemble weights use validation years only.
