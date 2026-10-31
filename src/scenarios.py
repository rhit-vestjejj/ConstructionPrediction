"""Scenario grids for the data center new-project forecast (conditional model estimates, not causal forecasts).

County-level sliders change inputs the model was trained on and re-run it, so counties respond differently:
  elec   industrial electricity price  (EIA-861, static)          -20% .. +20%
  tax    effective property tax rate   (ACS)                      -20% .. +20%
  dp     data-processing & hosting jobs (BLS NAICS 518)           -20% .. +100%
Output: per slider combination, the change in each county's probability (mean of the three model families' raw
probabilities, scenario minus baseline). The national hyperscaler-capex slider is applied in the app from the
learned national elasticity (outputs/results/capex_link.json).
"""
import itertools
import json

import numpy as np
import pandas as pd

import dc_timed as dt
from geo import ROOT

GRID = {"elec": [-0.2, -0.1, 0.0, 0.1, 0.2], "tax": [-0.2, -0.1, 0.0, 0.1, 0.2], "dp": [-0.2, 0.0, 0.2, 0.5, 1.0]}
MODELS = ("xgboost", "lightgbm", "logistic")


def apply(cur: pd.DataFrame, elec: float, tax: float, dp: float) -> pd.DataFrame:
    x = cur.copy()
    x["ind_price"] = x.ind_price * (1 + elec)
    x["prop_tax_rate"] = x.prop_tax_rate * (1 + tax)
    x["jobs518"] = np.log1p(np.expm1(x.jobs518) * (1 + dp))  # stored as log1p(jobs)
    return x


def main():
    d = dt.panel()
    lab = d.dropna(subset=["y"])
    cur = d[d.origin == d.origin.max()].reset_index(drop=True)
    models = {n: dt.make(n).fit(lab[dt.FEATS], lab.y) for n in MODELS}
    pred = lambda x: np.mean([m.predict_proba(x[dt.FEATS])[:, 1] for m in models.values()], axis=0)
    base = pred(cur)
    out = {"fips": cur.fips.tolist(), "grid": GRID, "delta": {}}
    for e, t, j in itertools.product(GRID["elec"], GRID["tax"], GRID["dp"]):
        delta = pred(apply(cur, e, t, j)) - base
        out["delta"][f"{e}|{t}|{j}"] = [int(round(v * 1000)) for v in delta]  # per-mille probability change
    (ROOT / "outputs" / "results" / "dc_scenarios.json").write_text(json.dumps(out, separators=(",", ":")))
    for key in ("0.2|0.0|0.0", "0.0|0.2|0.0", "0.0|0.0|1.0", "-0.2|-0.2|1.0"):
        v = np.array(out["delta"][key]) / 10
        print(f"elec|tax|dp = {key:14s} mean change {v.mean():+.2f} pp | counties up >1pp: {(v > 1).sum()} | down >1pp: {(v < -1).sum()} | max {v.max():+.1f} min {v.min():+.1f}")
    capex_link()


def capex_link(first=2017, last=dt.LAST_LABEL):
    """National capex slider: log(counties with a new project) on log(hyperscaler capex, $bn), OLS over first..last.
    capex_bn is carried over from the existing file (company filings); the county counts are recomputed from the projects."""
    path = ROOT / "outputs" / "results" / "capex_link.json"
    c = json.loads(path.read_text())
    n = pd.read_parquet(ROOT / "data" / "dc_projects.parquet").groupby("year").fips.nunique()
    yrs = list(range(first, last + 1))
    x, y = np.log([c["capex_bn"][str(t)] for t in yrs]), np.log([n[t] for t in yrs])
    (b, a), cov = np.polyfit(x, y, 1, cov="unscaled")
    se = float(np.sqrt(cov[0, 0] * ((y - (a + b * x)) ** 2).sum() / (len(yrs) - 2)))
    c.update(elasticity=float(b), se=se, years=[first, last], n=len(yrs), counties_with_new_project={str(t): int(n[t]) for t in n.index if 2015 <= t <= last})
    path.write_text(json.dumps(c))
    print(f"capex elasticity {b:.3f} (se {se:.3f}) over {first}-{last}")


if __name__ == "__main__":
    main()
