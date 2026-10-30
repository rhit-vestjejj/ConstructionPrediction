"""Refresh everything: pull the latest public data, retrain all models, rebuild the app, run the browser tests.

    .venv/bin/python refresh.py                 # full refresh
    .venv/bin/python refresh.py --from models   # skip downloads, start at the first model step

New data center news sweeps go in data/raw/dc_news/<sweep-name>/projects.csv (see docs/news_sweep.md) and are
merged automatically. Stops at the first failing step. Logs: logs/refresh_<timestamp>.log
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "bin" / "python")
STEPS = [  # (name, group, command relative to src/)
    ("Census building permits (annual + Jan-Aug)", "data", ["data.py"]),
    ("BLS quarterly construction jobs", "data", ["data_qcew_quarterly.py"]),
    ("BLS annual construction jobs", "data", ["data_qcew.py"]),
    ("ACS demographics", "data", ["data_acs.py"]),
    ("Data center projects (merge news sweeps)", "data", ["data_dc_projects.py"]),
    ("County models (fixed split)", "models", ["forecasting.py"]),
    ("County models (yearly retrained)", "models", ["walkforward.py", "--reuse-configs"]),
    ("Data center siting propensity", "models", ["datacenter_model.py"]),
    ("Data center new-project forecast", "models", ["dc_timed.py"]),
    ("Data center scenario grids", "models", ["scenarios.py"]),
    ("Metro H3 permit model", "models", ["metro_cells.py"]),
    ("Build app", "app", ["build_forecast_app.py"]),
    ("Browser tests", "app", [str(ROOT / "tests" / "e2e_app.py")]),
]


def freshness() -> dict:
    d = ROOT / "data"
    out = {}
    try:
        out["permits_annual_through"] = int(pd.read_parquet(d / "panel.parquet").year.max())
        out["permits_ytd_through"] = f"Jan-Aug {int(pd.read_parquet(d / 'permits_ytd.parquet').year.max())}"
        q = pd.read_parquet(d / "qcew_quarterly.parquet")
        y = int(q.year.max())
        out["jobs_quarterly_through"] = f"{y} Q{int(q[q.year == y].qtr.max())}"
        out["jobs_annual_through"] = int(pd.read_parquet(d / "qcew.parquet").year.max())
        p = pd.read_parquet(d / "dc_projects.parquet")
        out["dc_projects"] = f"{len(p)} projects, latest year {int(p.year.max())}"
    except Exception as e:  # freshness is informational only
        out["error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=["data", "models", "app"], default="data")
    a = ap.parse_args()
    order = ["data", "models", "app"]
    steps = [s for s in STEPS if order.index(s[1]) >= order.index(a.start)]
    (ROOT / "logs").mkdir(exist_ok=True)
    log = ROOT / "logs" / f"refresh_{datetime.datetime.now():%Y%m%d_%H%M}.log"
    t_all = time.time()
    with open(log, "w") as fh:
        for name, _, cmd in steps:
            t0 = time.time()
            fh.write(f"\n===== {name}: {' '.join(cmd)}\n")
            fh.flush()
            r = subprocess.run([PY, "-u", *cmd], cwd=ROOT / "src", stdout=fh, stderr=subprocess.STDOUT)
            status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
            print(f"{name:45s} {status:18s} {time.time() - t0:6.0f}s", flush=True)
            if r.returncode != 0:
                print(f"\nStopped. See {log}")
                sys.exit(1)
        summary = {"finished": f"{datetime.datetime.now():%Y-%m-%d %H:%M}", "minutes": round((time.time() - t_all) / 60, 1), **freshness()}
        fh.write("\n" + json.dumps(summary, indent=1))
    (ROOT / "outputs" / "results" / "last_refresh.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps(summary, indent=1) + f"\nLog: {log}")


if __name__ == "__main__":
    main()
