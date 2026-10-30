"""Turn the two summary JSONs into outputs/REPORT.md and a short console readout."""
from datetime import date

from dc_announcement_event_study import CAR_WINDOWS, RESULTS, ROOT


def pct(x):
    return f"{x * 100:+.1f}%"


def window_table(blocks, windows):
    """blocks: list of (label, stats-dict) columns."""
    head = "| Window | " + " | ".join(label for label, _ in blocks) + " |"
    rows = [head, "|---" * (len(blocks) + 1) + "|"]
    for w in windows:
        cells = []
        for _, s in blocks:
            v = s.get(w)
            cells.append(f"{pct(v['mean'])} (t={v['t']:.2f}, n={v['n']})" if v and v["n"] else "–")
        rows.append(f"| {w} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def build(study, builders):
    car_cols = [f"CAR[{a},{z}]" for a, z in CAR_WINDOWS]
    ec, ds, cov = study["earnings_comparison"], study["deal_size"], builders["coverage"]
    reg = ds.get("regression", {})
    md = f"""# Data center contract awards: is there a trading signal?

Generated {date.today()} by `run.py`. Abnormal returns use a market model: PKB as the benchmark for US
stocks, XIU.TO for Canadian stocks, and XLI as a robustness check. t-statistics are cross-sectional.

## Verdict

No usable signal from individual contract wins.

- **Contractor award releases:** across {study['n_analyzed']} verified releases, stocks rose about
  {pct(study['main_benchmark']['CAR[0,1]']['mean'])} over two days, and the gain was gone within a week.
  Clean standalone awards show about zero.
- **Who builds data centers:** of {cov['projects_researched']} large projects, only
  {cov['projects_with_public_builder']} had any public builder named. The main builders are private.
- **Earnings day:** earnings moves are about {ec['ratio_median_earnings_to_awards']:.1f}x larger than award
  moves. If data center activity reaches these stocks at all, it is through backlog reported on earnings day.

## 1. Contractor award announcements

**Sample:** {study['n_events_listed']} releases found, {study['n_excluded_earnings_overlap']} excluded for
falling within 3 trading days of earnings, {study['n_analyzed']} analysed.

{window_table([("Main benchmark", study["main_benchmark"]), ("XLI", study["xli_benchmark"]),
               ("No same-day news", study["main_excluding_same_day_news"]),
               ("Standalone only", study["main_standalone_non_borderline"])], car_cols)}

**Deal size:** {ds['n_with_deal_value']} events disclose a deal value.
- **Regression of CAR[0,+1] on log(deal value / revenue):** slope {reg.get('slope', float('nan')):+.3f},
  p={reg.get('slope_p', float('nan')):.2f}.
- **Deals above 5% of revenue:** {pct(ds['large (>5% rev)']['mean_CAR[0,1]'])} average
  (n={ds['large (>5% rev)']['n']}), against {pct(ds['small (<=5% rev)']['mean_CAR[0,1]'])} for smaller deals
  (n={ds['small (<=5% rev)']['n']}).

**Earnings comparison (same tickers, last 8 quarters):**

| Event | n | Median 2-day move, ignoring direction |
|---|---|---|
| Earnings releases | {ec['n_earnings']} | {ec['median_abs_CAR01_earnings'] * 100:.1f}% |
| Award releases | {ec['n_awards']} | {ec['median_abs_CAR01_awards'] * 100:.1f}% |

Earnings moves are larger than award moves (Mann-Whitney p={ec['mannwhitney_earnings_gt_awards_p']:.4f}).

![CAR chart](charts/car_chart.png)

## 2. Predict the location, then the builder

**Question:** if you knew where a data center would be built and which public contractor would build it,
would buying that contractor's stock have paid?

- **Builder named:** {cov['projects_with_any_builder_named']} of {cov['projects_researched']} projects have any
  builder publicly named.
- **Public builder:** {cov['projects_with_public_builder']} projects involve a public firm:
  {", ".join(f"{k} ×{v}" for k, v in cov['public_firm_counts'].items())}.
- **Private builders seen most often:**
  {", ".join(list(cov['private_firm_counts'])[:10])}.

{window_table([("Builder named", builders["builder_named"] | builders["perfect_foresight"]),
               ("Project announced", builders["project_announced"])],
              ["CAR[-10,-1]", "CAR[0,1]", "CAR[0,5]", "CAR[0,20]", "CAR[-60,5]", "CAR[-120,5]"])}

![Builder chart](charts/builder_car_chart.png)
"""
    (ROOT / "outputs" / "REPORT.md").write_text(md)
    return md


def console(study, builders):
    m, cov, ec = study["main_benchmark"], builders["coverage"], study["earnings_comparison"]
    lines = [
        "",
        f"Award announcements: {study['n_analyzed']} analysed of {study['n_events_listed']} found",
        *[f"  {w:12s} {pct(m[w]['mean']):>7s}  t={m[w]['t']:+.2f}  n={m[w]['n']}" for w in m],
        f"Median 2-day move: earnings {ec['median_abs_CAR01_earnings'] * 100:.1f}% vs awards "
        f"{ec['median_abs_CAR01_awards'] * 100:.1f}%",
        f"Builder research: {cov['projects_with_public_builder']} of {cov['projects_researched']} projects "
        f"have a public builder",
        f"Full report: {ROOT / 'outputs' / 'REPORT.md'}",
    ]
    print("\n".join(lines))
