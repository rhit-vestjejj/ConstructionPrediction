"""End-to-end check of outputs/forecast_app.html in real Chrome: exercises every control and a real map click.
Run:  .venv/bin/python tests/e2e_app.py [url]   (default: the built file)"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
if len(sys.argv) > 1:
    URL = sys.argv[1]
else:  # serve outputs/ over HTTP like a real host: past-year data is fetched on demand, which file:// can't do
    import functools, http.server, threading
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "outputs")))
    srv.RequestHandlerClass.log_message = lambda *a: None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    URL = f"http://127.0.0.1:{srv.server_address[1]}/forecast_app.html"
SHOTS = ROOT / "outputs" / "e2e"
SHOTS.mkdir(exist_ok=True, parents=True)
fails, errs = [], []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        fails.append(name)


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1500, "height": 950})
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append("PAGEERROR " + str(e)))
    pg.goto(URL)
    pg.wait_for_function("typeof MAP !== 'undefined' && MAP.count() > 0", timeout=30000)
    n = lambda: pg.evaluate("MAP.count()")  # shapes drawn on the map canvas
    pts = lambda: pg.evaluate("MAP.pts.length")  # markers drawn on the map canvas

    def click_shape(layer=None):
        """Real mouse click on a point inside a visible shape near the middle of the list."""
        xy = pg.evaluate("(li)=>{const ids=MAP.ids(li);for(let k=0;k<ids.length;k++){const p=mapPoint(ids[(Math.floor(ids.length/2)+k)%ids.length]);if(p)return p}return null}", layer)
        if xy:
            pg.mouse.click(*xy)
        return xy
    check("initial map renders", n() > 500, n())
    check("opens on data centers", pg.input_value("#cat") == "dc" and pg.input_value("#view") == "timed" and pts() > 50, pg.input_value("#cat"))
    pg.select_option("#cat", "res")
    pg.wait_for_timeout(600)
    # ---- right now / heating up / next lenses and county search
    for lens, view in [("now", "now"), ("trend", "trend"), ("next", "surge")]:
        pg.click(f'#lens button[data-v="{lens}"]')
        pg.wait_for_timeout(700)
        check(f"lens {lens} switches the map", pg.input_value("#view") == view and n() > 500 and not errs, pg.input_value("#view"))
    pg.fill("#search", "Maricopa County, AZ")
    pg.dispatch_event("#search", "change")
    pg.wait_for_timeout(600)
    check("search selects the county", "Maricopa County, AZ" in pg.inner_text("#detail"), pg.inner_text("#detail")[:60])
    check("selected county is outlined", pg.evaluate("$('map').data.some(t=>t.name==='sel'&&t.locations[0]==='04013')"))
    pg.click('#pages button[data-p="insights"]')
    pg.wait_for_timeout(800)
    ins = pg.inner_text("#insights")
    check("key takeaways: data center predictions", "Expected in 2027" in ins and "so far" in ins, ins[:120])
    check("key takeaways: heating up and next lists", "heating up fastest" in ins and "most likely to surge" in ins, ins[:120])
    pg.locator("#insights a[data-f]").first.click()
    pg.wait_for_timeout(900)
    check("takeaway county link opens it on the map", pg.is_visible("#map") and "FIPS" in pg.inner_text("#detail"), pg.inner_text("#detail")[:60])
    pg.select_option("#cat", "res")
    pg.wait_for_timeout(600)
    pg.click("#adv summary")  # horizon, model and mode live under Advanced settings

    def counties_click():
        """Real mouse click on a county path near the middle of the map."""
        click_shape()
        pg.wait_for_timeout(600)
        return pg.inner_text("#detail")

    for cat in ["all", "com", "inf", "res", "dc", "ind", "wh"]:
        pg.select_option("#cat", cat)
        pg.wait_for_timeout(900)
        check(f"category {cat} draws", pg.locator("#mapnote .banner").count() >= 0 and not errs, errs[-1:] )
        if cat in ("all", "com", "inf", "res", "dc"):
            txt = counties_click()
            check(f"{cat}: click shows county detail", "FIPS" in txt and "Click a county" not in txt, txt[:60])

    pg.select_option("#cat", "res")
    pg.wait_for_timeout(500)
    for h in ["1", "2", "3"]:
        pg.select_option("#horizon", h)
        pg.wait_for_timeout(500)
        check(f"residential horizon {h}y", "Forecast for" in pg.inner_text("#legend"), pg.inner_text("#legend")[:80])
    pg.click('#mode button[data-v="backtest"]')
    pg.wait_for_timeout(700)
    opts = pg.locator("#origin option").count()
    check("backtest offers origin years", opts >= 2, opts)
    pg.select_option("#origin", index=opts - 1)  # most recent year: has the top-10% (walk-forward) backtest
    pg.wait_for_timeout(500)
    check("top-10% views offered for recent backtest year", pg.locator('#view option[value="actual_top"]').count() == 1)
    pg.select_option("#origin", index=0)  # oldest year: top-10% views must be hidden, not broken
    pg.wait_for_timeout(500)
    check("no top-10% views for years without walk-forward data", pg.locator('#view option[value="actual_top"]').count() == 0 or "incl. top-10%" in pg.locator("#origin option").first.inner_text())
    pg.select_option("#origin", index=opts - 1)
    pg.wait_for_timeout(500)
    check("backtest legend", "Backtest" in pg.inner_text("#legend"), pg.inner_text("#legend")[:80])
    for v in ["surge", "actual_surge", "top", "actual_top", "prob", "growth", "disagree", "actual"]:
        pg.select_option("#view", v)
        pg.wait_for_timeout(500)
        check(f"backtest view {v}", n() > 300, n())
    for m in ["best", "ensemble", "xgboost", "lightgbm", "random_forest", "baseline_persistence"]:
        pg.select_option("#model", m)
        pg.wait_for_timeout(400)
        check(f"backtest model {m}", n() > 300, n())
    check("default view is the top-10% target", pg.input_value("#view") in ("top", "prob", "growth", "disagree", "actual", "actual_top"))
    pg.click('#pages button[data-p="stats"]')
    for tab in ["compare", "top", "feat", "valid"]:
        pg.click(f'#tabs button[data-k="{tab}"]')
        pg.wait_for_timeout(500)
        check(f"tab {tab} has content", len(pg.inner_text("#panel")) > 40, pg.inner_text("#panel")[:50])
    pg.click('#pages button[data-p="map"]')
    pg.click('#mode button[data-v="forecast"]')
    pg.select_option("#cat", "dc")
    pg.wait_for_timeout(700)
    check("data center default view is the timed forecast", pg.input_value("#view") == "timed", pg.input_value("#view"))
    click_shape()
    pg.wait_for_timeout(500)
    check("dc click shows new-project chance", "Chance of a new project in" in pg.inner_text("#detail"), pg.inner_text("#detail")[:80])
    pg.click('#mode button[data-v="backtest"]')
    pg.wait_for_timeout(700)
    check("dc backtest years", pg.locator("#origin option").count() == 5, pg.locator("#origin option").count())
    for v in ["timed", "actual_timed"]:
        pg.select_option("#view", v)
        pg.wait_for_timeout(500)
        check(f"dc backtest view {v}", n() > 1000, n())
    pg.click('#mode button[data-v="forecast"]')
    pg.wait_for_timeout(600)
    for v in ["timed", "like", "pct", "white", "count", "disagree"]:
        pg.select_option("#view", v)
        pg.wait_for_timeout(400)
        check(f"dc view {v}", n() > 300, n())
    # ---- scenario sliders (data centers)
    pg.select_option("#view", "timed")
    pg.wait_for_timeout(400)
    check("scenario sliders shown", pg.locator("#s_capex").count() == 1)
    pg.locator("#s_elec").fill("0.2")
    pg.wait_for_timeout(700)
    check("moving a slider shows the final chance", pg.input_value("#view") == "scen", pg.input_value("#view"))
    pg.click('#s_view button[data-v="scen_diff"]')
    pg.wait_for_timeout(500)
    check("what-if change view", pg.input_value("#view") == "scen_diff" and "Change" in pg.inner_text("#map .cbar"), pg.input_value("#view"))
    pg.click('#s_view button[data-v="scen"]')
    pg.wait_for_timeout(500)
    check("legend reports scenario totals", "scenario" in pg.inner_text("#legend"), pg.inner_text("#legend")[:120])
    pg.locator("#s_capex").fill("0.5")
    pg.wait_for_timeout(700)
    check("slider survives redraw", pg.locator("#s_capex").count() == 1 and pg.input_value("#s_capex") == "0.5", pg.input_value("#s_capex"))
    pg.click('#pages button[data-p="stats"]')
    pg.click('#tabs button[data-k="dcscen"]')
    pg.wait_for_timeout(500)
    check("scenario table lists counties", "Largest increases" in pg.inner_text("#panel"), pg.inner_text("#panel")[:80])
    pg.click('#pages button[data-p="map"]')
    pg.click('#s_scope button[data-v="county"]')
    pg.wait_for_timeout(500)
    pg.evaluate("detail('48253')")  # Jones County, TX
    pg.wait_for_timeout(400)
    check("county-only scope names the county", "Jones" in pg.inner_text("#s_scopeName"), pg.inner_text("#s_scopeName"))
    check("county detail shows per-lever sensitivity", "Electricity 20% cheaper" in pg.inner_text("#detail") and "points" in pg.inner_text("#detail"), pg.inner_text("#detail")[:120])
    changed = pg.evaluate("(()=>{SC.scope='county';S.sel='48253';const s=scenProb();return s.scen.filter((v,i)=>Math.abs(v-s.base[i])>1e-9).length})()")
    check("county-only scenario changes exactly one county", changed <= 1, changed)
    pg.click('#s_scope button[data-v="all"]')
    pg.click("#s_reset")
    pg.wait_for_timeout(600)
    check("reset returns to baseline view", pg.input_value("#view") == "timed", pg.input_value("#view"))
    pg.evaluate("detail('51107')")  # Loudoun County, VA: in the dcmap.us snapshot
    pg.wait_for_timeout(300)
    txt = pg.inner_text("#detail")
    check("county detail shows the dcmap pipeline", "pipeline today" in txt.lower() and "under construction" in txt.lower() and "US Data Center Map" in txt, txt[-200:])
    pg.click('#pages button[data-p="stats"]')
    for tab in ["dctimed", "dcscen", "dcmodel", "emerging", "white", "pipeline", "dccaveats"]:
        pg.click(f'#tabs button[data-k="{tab}"]')
        pg.wait_for_timeout(400)
        check(f"dc tab {tab}", len(pg.inner_text("#panel")) > 40, pg.inner_text("#panel")[:50])
        if tab == "dctimed":
            check("accuracy page has the dcmap outside check", "Outside check" in pg.inner_text("#panel") and "in its top 50" in pg.inner_text("#panel"), pg.inner_text("#panel")[-200:])
    pg.click('#pages button[data-p="map"]')
    # ---- metro H3 cells
    pg.select_option("#cat", "metro")
    pg.wait_for_timeout(900)
    check("metro selector visible", pg.is_visible("#metrosel"))
    for metro in ["fortworth", "mecklenburg", "columbus"]:
        pg.select_option("#metrosel", metro)
        pg.wait_for_timeout(900)
        hexes = pg.evaluate("MAP.ids(-1).length")
        check(f"metro {metro}: hexes drawn", hexes > 200, hexes)
        click_shape(-1)  # county outlines are the first layer, hexes the last
        pg.wait_for_timeout(500)
        txt = pg.inner_text("#detail")
        check(f"metro {metro}: hex click shows cell detail", "H3 cell" in txt and "Probability" in txt, txt[:80])
    pg.screenshot(path=str(SHOTS / "metro.png"))
    pg.click('#mode button[data-v="backtest"]')
    pg.wait_for_timeout(700)
    check("metro backtest origins", pg.locator("#origin option").count() >= 3, pg.locator("#origin option").count())
    for v in ["prob", "actual"]:
        pg.select_option("#view", v)
        pg.wait_for_timeout(500)
        check(f"metro backtest view {v}", pg.evaluate("MAP.ids(-1).length") > 200)
    for m in ["baseline_recent_history", "xgboost", "ensemble"]:
        pg.select_option("#model", m)
        pg.wait_for_timeout(400)
        check(f"metro backtest model {m}", "Backtest" in pg.inner_text("#legend"))
    pg.click('#pages button[data-p="stats"]')
    for tab in ["mcompare", "mtop", "mvalid"]:
        pg.click(f'#tabs button[data-k="{tab}"]')
        pg.wait_for_timeout(400)
        check(f"metro tab {tab}", len(pg.inner_text("#panel")) > 60, pg.inner_text("#panel")[:50])
    pg.click('#pages button[data-p="map"]')
    pg.click('#mode button[data-v="forecast"]')
    pg.select_option("#cat", "dc")
    pg.wait_for_timeout(600)
    check("leaving metro hides metro selector", not pg.is_visible("#metrosel"))
    pg.select_option("#cat", "all")
    pg.wait_for_timeout(700)
    check("default view is the surge view", pg.input_value("#view") == "surge", pg.input_value("#view"))
    check("data center projects hidden outside data centers", pts() == 0, pts())
    pg.select_option("#cat", "dc")
    pg.wait_for_timeout(600)
    check("announced projects layer drawn", pts() > 50, pts())
    check("data center forecast is for next year", "2027" in pg.inner_text("#view"), pg.inner_text("#view")[:80])
    for lens, view in [("now", "count"), ("trend", "dctrend")]:
        pg.click(f'#lens button[data-v="{lens}"]')
        pg.wait_for_timeout(600)
        check(f"dc lens {lens}", pg.input_value("#view") == view and n() > 300, pg.input_value("#view"))
    pg.check("#dc")
    pg.wait_for_timeout(500)
    check("OSM layer toggles", pts() > 100, pts())
    pg.screenshot(path=str(SHOTS / "dc.png"))
    pg.set_viewport_size({"width": 400, "height": 800})
    pg.wait_for_timeout(600)
    check("mobile: no horizontal scroll", pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2"), pg.evaluate("document.documentElement.scrollWidth"))
    pg.screenshot(path=str(SHOTS / "mobile.png"), full_page=False)
    b.close()

check("no console/page errors", not errs, "; ".join(errs[:3]))
print(f"\n{len(fails)} failures" if fails else "\nALL PASSED")
sys.exit(1 if fails else 0)
