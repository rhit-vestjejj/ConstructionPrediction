"""Run the data center event study end to end.

    .venv/bin/python run.py              # everything: events -> award study -> builder backtest -> report
    .venv/bin/python run.py study        # just one step (events | study | builders | report)
    .venv/bin/python run.py --refresh    # re-download prices and SEC data first
"""
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # yfinance trips pandas 3 deprecation warnings on every download
sys.path.insert(0, str(Path(__file__).parent / "src"))

import build_events  # noqa: E402
import builder_backtest  # noqa: E402
import dc_announcement_event_study as study  # noqa: E402
import report  # noqa: E402

STEPS = ["events", "study", "builders", "report"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=STEPS, help="run only this step (default: all)")
    ap.add_argument("--refresh", action="store_true", help="clear cached prices and SEC data before running")
    args = ap.parse_args()

    if args.refresh:
        shutil.rmtree(study.CACHE, ignore_errors=True)
    steps = [args.step] if args.step else STEPS

    if "events" in steps:
        print("[1/4] Merging verified events")
        build_events.main()
    if "study" in steps:
        print("[2/4] Award announcement event study")
        study.main()
    if "builders" in steps:
        print("[3/4] Builder backtest")
        builder_backtest.main()
    if "report" in steps:
        print("[4/4] Writing report")
        s = json.loads((study.RESULTS / "summary.json").read_text())
        b = json.loads((study.RESULTS / "builder_summary.json").read_text())
        report.build(s, b)
        report.console(s, b)


if __name__ == "__main__":
    main()
