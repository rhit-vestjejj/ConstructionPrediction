"""Serve the forecast app locally and open it in a browser:  .venv/bin/python serve.py  [--port 8000]"""
import argparse
import functools
import http.server
import threading
import webbrowser
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--no-open", action="store_true")
a = ap.parse_args()
root = Path(__file__).resolve().parent / "outputs"
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)
url = f"http://127.0.0.1:{a.port}/forecast_app.html"
print(f"Serving {root} at {url}  (Ctrl+C to stop)")
if not a.no_open:
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
