"""Tiny dashboard server: serves index.html and /api/runs (all runs/*/log.jsonl as JSON).

python3 dashboard/server.py [--port 8765] [--host 127.0.0.1] [--runs runs] [--include "refit_*"] [--page refit.html]
"""
import argparse, fnmatch, json, os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--runs", default=str(HERE.parent / "runs"))
ap.add_argument("--exclude", default="smoke*,timing,engine_*", help="comma-separated run-dir glob patterns to hide")
ap.add_argument("--include", default="*", help="comma-separated run-dir glob patterns to show")
ap.add_argument("--page", default="index.html", help="page served at /")
args = ap.parse_args()
RUNS = Path(args.runs)
EXCLUDE = set(filter(None, args.exclude.split(",")))
INCLUDE = set(filter(None, args.include.split(",")))


def load_runs():
    out = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir() and any(fnmatch.fnmatch(p.name, i) for i in INCLUDE) and not any(fnmatch.fnmatch(p.name, e) for e in EXCLUDE) and (p / "log.jsonl").exists()):
        train, evals = [], []
        for line in (d / "log.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written line
            (evals if "eval_acc" in r else train).append(r)
        if not train and not evals:
            continue
        cfg = {}
        if (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text())
        out.append({"name": d.name, "config": cfg, "train": train, "eval": evals,
                    "mtime": (d / "log.jsonl").stat().st_mtime,
                    "finished": any(e["step"] == (cfg.get("max_steps") or cfg.get("steps", 200)) for e in evals)
                                and ((d / "best_mask.pt").exists() or bool(cfg.get("max_steps")))})
    return out


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def do_GET(self):
        if self.path in ("/", "/index.html") and args.page != "index.html":
            self.path = "/" + args.page
        if self.path.startswith("/api/runs"):
            body = json.dumps(load_runs()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, *a):
        pass


print(f"dashboard: http://{args.host}:{args.port}  (runs dir: {RUNS})", flush=True)
ThreadingHTTPServer((args.host, args.port), H).serve_forever()
