"""Bundle run logs + mask stats into site/data.js (stdlib only) so index.html works from file://.
  python3 site/build_data.py        # re-run whenever runs/ changes
"""
import json, os, glob, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = {}
for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "log.jsonl"))):
    name = os.path.basename(os.path.dirname(path))
    if name in ("smoke", "timing"):
        continue
    evals, train = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # partially written line
        if "eval_acc" in r:
            evals.append([r["step"], r["eval_acc"], r["frac_of_gate"], r["frac_of_model"], r["deleted"]])
        else:
            train.append([r["step"], r["reward"], r["frac_deleted_gate"], r["mean_len"], r["alpha"], r["n_active"], r["step_s"]])
    cfg_p = os.path.join(os.path.dirname(path), "config.json")
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    runs[name] = {"eval": evals, "train": train, "steps": cfg.get("steps", 200),
                  "mtime": os.path.getmtime(path)}
masks = {}
mp = os.path.join(ROOT, "site", "mask_stats.json")
if os.path.exists(mp):
    masks = json.load(open(mp))
    for r in masks.get("runs", {}).values():
        r.pop("col_counts", None)
data = {"built": time.time(), "runs": runs, "masks": masks}
with open(os.path.join(ROOT, "site", "data.js"), "w") as f:
    f.write("window.DT_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
print({k: (len(v["eval"]), len(v["train"])) for k, v in runs.items()}, "->", "site/data.js")
