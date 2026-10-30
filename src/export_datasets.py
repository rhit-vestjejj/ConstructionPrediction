"""Export the top-10% datasets (same features/labels/years as walkforward.py) for the separate GNN environment."""
import json

import forecasting as fc
from geo import ROOT

OUT = ROOT / "data" / "datasets"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    topf = json.loads((ROOT / "outputs" / "results" / "top_features.json").read_text())
    meta = {}
    for t in fc.TARGETS:
        data = fc.build_dataset(t)
        for h in fc.HORIZONS:
            cols = fc.feature_groups(t)[topf.get(f"{t.key}|{h}", "own+spatial+econ")]
            d = data.dropna(subset=[f"inc{h}"]).copy()
            d["ty"] = d.origin + h
            d["top"] = d.groupby("origin")[f"inc{h}"].transform(lambda s: (s >= s.quantile(1 - fc.TOP_SHARE)).astype(float))
            keep = list(dict.fromkeys(["fips", "origin", "ty", "top", "lat", "lon", "size"] + cols))
            d[keep].to_parquet(OUT / f"{t.key}_h{h}.parquet")
            meta[f"{t.key}|{h}"] = {"features": cols, "train_end": t.train_end, "val_end": t.val_end}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("exported", len(meta), "datasets")
