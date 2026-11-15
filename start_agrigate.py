#!/usr/bin/env python3
"""
AgriGate: starts the website and the LightGBM model API, then opens your browser.

Before the first run, install the model's libraries once:
    python -m pip install -r model/requirements.txt
Then start it with:
    python start_agrigate.py
"""

import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import webbrowser
from urllib.parse import urlparse, parse_qs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- trained LightGBM model (model/agrigate_model.py)
MODEL = None
MODEL_ERROR = None
try:
    sys.path.insert(0, os.path.join(BASE_DIR, "model"))
    import agrigate_model as MODEL
except Exception as e:  # site still runs, the page shows "model offline"
    MODEL_ERROR = f"{e.__class__.__name__}: {e}"


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class AgriGateHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def send_json(self, obj, code=200):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path.startswith("/api/model/"):
            if MODEL is None:
                return self.send_json({"ok": False, "error": "LightGBM model not loaded: " + str(MODEL_ERROR)}, 503)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/model/info":
                    return self.send_json({"ok": True, "model": MODEL.MODEL_INFO})
                lat, lon = float(q["lat"]), float(q["lon"])
                if url.path == "/api/model/predict":
                    return self.send_json(MODEL.predict(lat, lon))
                if url.path == "/api/model/climate":
                    return self.send_json(MODEL.climate_trend(lat, lon, int(q.get("until", 2050))))
                if url.path == "/api/model/future":
                    return self.send_json(MODEL.future(lat, lon, int(q.get("year", 2050))))
                return self.send_json({"ok": False, "error": "unknown endpoint"}, 404)
            except (KeyError, ValueError):
                return self.send_json({"ok": False, "error": "give lat and lon as numbers, e.g. ?lat=41.6&lon=-93.6"}, 400)
            except Exception as e:
                return self.send_json({"ok": False, "error": f"{e.__class__.__name__}: {e}"}, 500)
        if self.path == "/health":
            payload = json.dumps({"status": "ok", "app": "AgriGate", "lightgbm_model": MODEL is not None}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, format, *args):
        sys.stdout.write("[AgriGate] " + (format % args) + "\n")
        sys.stdout.flush()


def pick_port(preferred=8768):
    for port in (preferred, 8769, 8770, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
        except OSError:
            continue
    return 8768


def main():
    os.chdir(BASE_DIR)
    hosted = bool(os.environ.get("PORT"))          # set by hosting services such as Render
    if hosted:
        port = int(os.environ["PORT"])
        server = ThreadingHTTPServer(("0.0.0.0", port), AgriGateHandler)
        url = f"http://0.0.0.0:{port}/index.html"
    else:
        port = pick_port(8768)
        url = f"http://127.0.0.1:{port}/index.html"
        server = ThreadingHTTPServer(("127.0.0.1", port), AgriGateHandler)
    print(f"AgriGate running at {url}")
    print("LightGBM model: " + ("loaded" if MODEL is not None else f"NOT loaded ({MODEL_ERROR})\n  -> install the libraries first:  python -m pip install -r model/requirements.txt"))
    print("Keep this window open while you use AgriGate. Close it (or press Ctrl+C) to stop.")

    if not hosted and "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down AgriGate...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                            # keep a double-clicked window open so the error can be read
        print(f"\nAgriGate could not start: {e.__class__.__name__}: {e}")
        if not os.environ.get("PORT"):
            input("Press Enter to close...")
