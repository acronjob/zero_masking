"""Bake a saved Deletion Tuning mask into a copy of the base checkpoint (deleted gate_proj weights set to 0).

python3 tools/apply_mask.py runs/paper_seed3/mask_step50.pt models/qwen35-0.8b-seed3-step50
The mask file is a list of 24 int32 tensors (flat indices of deleted weights) for model.layers.{0..23}.mlp.gate_proj,
in layer order, as saved by train_grpo_paper.py. A refit checkpoint (refit_step{N}.pt: per layer {"idx", "delta"})
sets those coordinates to the trained values instead of zero.
"""
import json, shutil, sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

mask_path, out = Path(sys.argv[1]), Path(sys.argv[2])
src = Path(snapshot_download("Qwen/Qwen3.5-0.8B"))
mask = torch.load(mask_path)
assert len(mask) == 24, len(mask)
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if not f.name.endswith(".safetensors"):
        shutil.copy(f, out / f.name)  # config, tokenizer, chat template, index, ...
deleted = newly_zero = 0
for f in src.glob("*.safetensors"):
    sd = load_file(f)
    for i, idx in enumerate(mask):
        k = f"model.language_model.layers.{i}.mlp.gate_proj.weight"
        if k in sd:
            w = sd[k].clone().view(-1)
            if isinstance(idx, dict):  # refit: trained values
                idx, val = idx["idx"], idx["delta"].to(w.dtype)
            else:
                val = 0
            newly_zero += int((w[idx.long()] != 0).sum())
            w[idx.long()] = val
            sd[k] = w.view_as(sd[k])
            deleted += idx.numel()
    save_file(sd, out / f.name, metadata={"format": "pt"})
json.dump({"source_mask": str(mask_path), "deleted": deleted, "newly_zeroed": newly_zero},
          open(out / "deletion_mask_info.json", "w"), indent=1)
print(f"set {deleted} gate_proj weights ({newly_zero} were nonzero) -> {out}")
