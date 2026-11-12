"""Summarise saved best_mask.pt files (flat indices of deleted gate_proj weights per layer) -> site/mask_stats.json.
Run inside the deletion-tuning image (needs torch; CPU only):
  docker run --rm -v "$PWD:/w" -w /w deletion-tuning python site/analyze_masks.py
"""
import json, os, sys
import torch

H, I = 1024, 3584            # Qwen3.5-0.8B hidden / intermediate; gate_proj.weight is [I, H]
RUNS = ["dt_gsm8k", "dt_seed1", "dt_seed2", "dt_seed3"]
out = {"hidden": H, "intermediate": I, "runs": {}}
sets = {}
for r in RUNS:
    p = f"runs/{r}/best_mask.pt"
    if not os.path.exists(p):
        continue
    layers = torch.load(p)
    per_layer, row_max, col_hist, rows_touched = [], [], [], []
    s = set()
    col_tot = torch.zeros(H)
    for li, idx in enumerate(layers):
        idx = idx.long()
        rows, cols = idx // H, idx % H
        per_layer.append(int(idx.numel()))
        rc = torch.bincount(rows, minlength=I)
        cc = torch.bincount(cols, minlength=H).float()
        col_tot += cc
        row_max.append(int(rc.max()))
        rows_touched.append(int((rc > 0).sum()))
        col_hist.append(torch.histc(cc, bins=32, min=0, max=float(cc.max()) + 1).int().tolist())
        s.update((li << 32 | i) for i in idx.tolist())
    sets[r] = s
    # concentration of deletions across hidden (input) dims: share captured by top 1% / 10% of columns
    srt = col_tot.sort(descending=True).values
    tot = float(srt.sum())
    out["runs"][r] = {
        "per_layer": per_layer, "total": sum(per_layer),
        "rows_touched": rows_touched, "row_max": row_max,
        "top1pct_cols_share": float(srt[: H // 100].sum()) / tot,
        "top10pct_cols_share": float(srt[: H // 10].sum()) / tot,
        "col_counts_sorted": srt[:64].int().tolist(),
        "col_counts": col_tot.int().tolist(),
    }
names = list(sets)
out["overlap"] = {f"{a}|{b}": len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                  for i, a in enumerate(names) for b in names[i + 1:]}
out["intersect"] = {f"{a}|{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
json.dump(out, open("site/mask_stats.json", "w"))
print(json.dumps({k: (v if k != "runs" else {r: {kk: vv for kk, vv in d.items() if kk in ("per_layer", "total", "top1pct_cols_share", "top10pct_cols_share", "row_max")} for r, d in v.items()}) for k, v in out.items() if k not in ("hidden", "intermediate")}, indent=0))
