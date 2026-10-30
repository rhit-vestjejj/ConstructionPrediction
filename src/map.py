"""Interactive US county map of building permits, with a year slider."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from geo import ROOT, load_geojson

panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
pop_proxy = panel.assign(u=panel.units)
years = list(range(2000, 2025, 4)) + [2024]
years = sorted(set(years))

geo = load_geojson()
frames, first = [], None
for y in years:
    d = panel[panel.year == y]
    trace = go.Choropleth(
        geojson=geo, locations=d.fips, z=np.log10(d.units.clip(lower=1)),
        customdata=np.c_[d.name, d.state, d.units, d.value / 1e6],
        hovertemplate="%{customdata[0]}<br>%{customdata[2]:,.0f} units permitted<br>$%{customdata[3]:,.1f}M<extra></extra>",
        colorscale="YlOrRd", zmin=0, zmax=4, marker_line_width=0.1,
        colorbar=dict(title="Units / yr", tickvals=[0, 1, 2, 3, 4], ticktext=["1", "10", "100", "1k", "10k"]),
    )
    frames.append(go.Frame(data=[trace], name=str(y)))
    first = first or trace

fig = go.Figure(data=[first], frames=frames)
fig.update_layout(
    title="New housing units permitted by county",
    geo=dict(scope="usa", projection=go.layout.geo.Projection(type="albers usa"), showlakes=False),
    margin=dict(l=0, r=0, t=50, b=0),
    sliders=[dict(steps=[dict(method="animate", label=str(y), args=[[str(y)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for y in years],
                  currentvalue=dict(prefix="Year: "))],
)
out = ROOT / "outputs" / "permits_map.html"
fig.write_html(out, include_plotlyjs="cdn")
print("wrote", out)
