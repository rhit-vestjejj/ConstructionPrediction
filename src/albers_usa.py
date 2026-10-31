"""Pre-projected county outlines for the forecast app.

A port of d3-geo's geoAlbersUsa (lower 48 Albers + Alaska and Hawaii insets) at its standard 960x500 frame
(scale 1070, translate [480, 250]), so the browser draws ready-made SVG paths instead of projecting ~100k points on
every page load. The page uses d3.geoAlbersUsa().scale(1070).translate([480, 250]) for markers, so both agree.
"""
import math

K, TX, TY = 1070.0, 480.0, 250.0


def _conic_equal_area(parallels, rotate, center, k, tx, ty):
    y0, y1 = (math.radians(p) for p in parallels)
    sy0 = math.sin(y0)
    n = (sy0 + math.sin(y1)) / 2
    c = 1 + sy0 * (2 * n - sy0)
    r0 = math.sqrt(c) / n

    def raw(lam, phi):
        r = math.sqrt(c - 2 * n * math.sin(phi)) / n
        return r * math.sin(lam * n), r0 - r * math.cos(lam * n)

    cx, cy = raw(*(math.radians(v) for v in center))  # d3 applies center in the rotated frame
    dlam = math.radians(rotate)

    def project(lon, lat):
        lam = math.radians(lon) + dlam
        lam = lam - 2 * math.pi if lam > math.pi else lam + 2 * math.pi if lam < -math.pi else lam
        x, y = raw(lam, math.radians(lat))
        return tx + k * (x - cx), ty - k * (y - cy)

    return project


LOWER48 = _conic_equal_area((29.5, 45.5), 96, (-0.6, 38.7), K, TX, TY)
ALASKA = _conic_equal_area((55, 65), 154, (-2, 58.5), K * 0.35, TX - 0.307 * K, TY + 0.201 * K)
HAWAII = _conic_equal_area((8, 18), 157, (-3, 19.9), K, TX - 0.205 * K, TY + 0.212 * K)


def county_paths(geo, nd=1) -> dict:
    """FIPS -> [SVG path string, x0, y0, x1, y1]. Puerto Rico and other areas outside AlbersUsa are left out, as d3 does."""
    out = {}
    for f in geo["features"]:
        fips = str(f["id"])
        proj = {"02": ALASKA, "15": HAWAII}.get(fips[:2], LOWER48)
        if fips[:2] in ("60", "66", "69", "72", "78"):
            continue
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        parts, xs, ys = [], [], []
        for poly in polys:
            for ring in poly:
                pts, last = [], None
                for lon, lat in ring:
                    x, y = proj(lon, lat)
                    p = (round(x, nd), round(y, nd))
                    if p != last:
                        pts.append(p)
                        last = p
                if len(pts) >= 3:
                    parts.append("M" + "L".join(f"{x:g},{y:g}" for x, y in pts) + "Z")
                    xs += [x for x, _ in pts]
                    ys += [y for _, y in pts]
        if parts:  # the bounding box saves the page from parsing every path to find it
            out[fips] = ["".join(parts), min(xs), min(ys), max(xs), max(ys)]
    return out
