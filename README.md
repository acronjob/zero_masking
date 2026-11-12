# zero_masking

**Deletion Tuning on Qwen3.5-0.8B (GRPO / GSM8K)**

Minimal reproduction of *The Surprising Effectiveness of Deleting Weights in LLM Post-Training*
(arXiv:2611.10866): train **only a binary mask over `gate_proj` weights** (no weight values change) with GRPO on GSM8K.

| file | what |
|---|---|
| `deletion_tuning.py` | `MaskedLinear` (frozen W, learnable logits G), soft mask σ(α(t)G) w/ α annealed 1e3→1e4, hard mask `G>0`, Wanda init (η=1/2048), −λ‖W̃‖² regularizer |
| `train_grpo.py` | single-GPU GRPO loop (HF `generate`, hard-mask rollouts, soft-mask updates), greedy GSM8K eval, logs to `runs/<name>/log.jsonl`, saves `best_mask.pt` (indices of deleted weights) |
| `run.sh` | builds the image and runs on `GPU` (default 3), mounting HF cache + `runs/` |

```bash
./run.sh                                  # default: 200 steps, 32 prompts x 4 rollouts, ~80 min on one RTX PRO 6000
GPU=5 ./run.sh --steps 400 --lr 3e-4      # any train_grpo.py flag
```

## Faithful replication (`train_grpo_paper.py`, vLLM rollouts)

Ports the authors' released GSM8K recipe ([deleting-weights](https://github.com/agentic-learning-ai-lab/deleting-weights):
`verl/verl/trainer/config/gsm8k.yaml`, `src/verl_adapters.py`, `src/wanda_calibration.py`, `src/learnable_sparsity/`,
verl's `dp_actor.py`/`core_algos.py`) to one GPU. The script docstring lists every setting. Key points that differ
from `train_grpo.py`: 348 steps (3 epochs × 116), 64 prompts × 4, 4096-token answers, **4 PPO mini-batch updates per
step** (clip 0.2, dual-clip 3.0, old log-probs from the hard mask), sharpness schedule over all 1392 optimizer steps,
λ decaying to 0, AdamW wd 0.01 + grad-clip 1.0, verl's strict GSM8K reward, full 1,319-problem test eval.

```bash
GPU=2 NAME=paper_seed0 IMAGE=deletion-tuning-vllm SCRIPT=train_grpo_paper.py ./run.sh --seed 0 --out /workspace/runs/paper_seed0
docker stop paper_seed0                                       # stop a run
```
- Rollouts/eval run in vLLM 0.33 in the same process (`Dockerfile.vllm`). vLLM sleeps during training. After every update,
  `W * 1[G > 0]` is written in place into vLLM's fused `gate_up_proj`, then read back and checked exactly (the run aborts
  on mismatch). `--engine hf` uses HF `generate` instead (~7× slower per step).
- Validation: vLLM vs HF greedy accuracy on the same step-0 mask, 256 test problems: 45.7% vs 46.5%. Zeroing
  `gate_proj` in vLLM breaks its output, and restoring it reproduces the original tokens exactly.
- Steady state is about 40–60 s/step, about 6 h per seed. A harmless `cumem_allocator.cpp:249` error prints at interpreter
  exit (vLLM sleep-mode teardown).
- Remaining differences from the paper: the model (Qwen3.5-0.8B, non-thinking default template, vs Llama 3.2 Instruct);
  the vLLM version; micro-batches run as padded length-sorted chunks (same loss weights as verl's packed micro-batches).

## Results

### Faithful runs (`runs/paper_seed*`): GSM8K test accuracy, all 1,319 problems, greedy, strict verl grader

| seed | step 0 | step 25 | step 50 | best (step) | last eval |
|---|---|---|---|---|---|
| 0 | 44.8 | 55.6 | 58.6 | 58.6 (50) | 58.2 (300) |
| 1 | 48.0 | 56.7 | 58.7 | 59.7 (100) | 58.5 (225) |
| 2 | 49.3 | 57.7 | 59.1 | 59.3 (125) | 57.4 (200) |
| 3 | 47.6 | 58.2 | 60.8 | 60.8 (50) | 60.5 (225) |
| mean | 47.4 ± 1.9 | 57.1 | 59.3 ± 1.0 | | |

Step 0 is the Wanda-initialised mask with about 0.18% of `gate_proj` weights deleted. Almost all of the gain arrives by
step 50, when about 0.22% of `gate_proj` weights (about 191K, or 0.026% of the model) are deleted. After that, both the
mask and the accuracy stay flat. The runs were stopped early (seed 0 at step 321 of 348, the others at steps 213–232)
because nothing was changing. A mask is saved at every eval as `mask_step{N}.pt`, a list of 24 int32 tensors holding the
flat indices of the deleted weights. `tools/apply_mask.py` bakes a mask into a copy of the HF checkpoint.

The first, scaled-down runs (`runs/dt_*`, `train_grpo.py`) used a lenient grader on the first 500 test problems. They
gained +1.8 ± 1.3 points.

### Long-context check: `llm-decode-bench --test-profile estonia` (`runs/bench/`)

Settings: a 136K-token prompt, 30 runs at concurrency 30, vLLM with `--max-model-len 262144`, and server-default
sampling. `tools/bench_masks.py` serves one masked model per GPU. Each cell is PASS out of 30; the base model scores
**17/30**.

| mask | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| step 0 | 17 | 19 | 20 | 14 |
| step 25 | 18 | 17 | 18 | 18 |
| step 50 | 17 | 17 | 17 | 16 |
| step 75 | 17 | 18 | 18 | 15 |
| step 100 | 15 | | | |

No mask differs from the base model by more than sampling noise, and there is no trend with training. The GSM8K gain
neither hurts nor helps long-context retrieval at this sample size.

### Long-context check: `--test-profile lavd-test`

This test hides human errors in about 48K characters of structured data; the model must find and fix them before
computing a ticket count and hours (expected 72 and 46). Settings: 10 runs per model and server-default sampling.
The base model and all 29 masks tested (steps 0–150 for every seed, plus seed 0's step 175) scored **0/10**. None of
the 249 parseable answers got either number right. The test is beyond a 0.8B model, so it can't distinguish the
masks. The run was stopped before the remaining later-step masks.

### Refit: retrain the weights Deletion Tuning deleted (`runs/refit_*`)

`train_grpo_paper.py --refit_mask runs/paper_seed3/mask_step50.pt` freezes the model and trains only the real values
of the 190,761 `gate_proj` coordinates that the best mask deleted. That mask is seed 3 at step 50, which scores 60.8%.
The recipe is the same GRPO setup, with weight decay 0 and 150 steps. `--refit_init zero` starts from the
deletion-tuned model. `--refit_init orig` starts from the original values, i.e. the unmodified model.
`--refit_random` trains the same number of random coordinates per layer instead. Each eval saves the values as
`refit_step{N}.pt`, and `tools/apply_mask.py` bakes them into a checkpoint.

| run (1 seed each) | trained coordinates | start | lr | step 0 | best (step) | step 150 |
|---|---|---|---|---|---|---|
| `refit_zero_lr1e-4_s0` | deleted by the mask | 0 | 1e-4 | 60.8 | **65.0** (125) | 64.6 |
| `refit_zero_lr1e-5_s0` | deleted by the mask | 0 | 1e-5 | 60.8 | 62.1 (150) | 62.1 |
| `refit_orig_lr1e-4_s0` | deleted by the mask | original | 1e-4 | 48.4 | 61.9 (150) | 61.9 |
| `refit_rand_lr1e-4_s0` | random, same count | original | 1e-4 | 49.1 | 60.3 (125) | 59.9 |

Deleting first and then retraining the deleted coordinates reaches about 65%. That beats the mask alone (60.8%),
training the same coordinates from their original values (61.9%), and training random coordinates (about 60%). There
is only one seed per arm, so the gaps of about 3 points between arms need more seeds to confirm.
`python3 dashboard/server.py --host 0.0.0.0 --port 8766 --include "refit_*" --page refit.html` serves a dashboard
for these runs.

### Estonia: repeats and a paired comparison

Single 30-request estonia runs are too noisy to compare models. The base model alone scored 17, 17, 12 and 18 out of
30 across four identical runs. The refit at step 100 scored 19, 15, 15 and 20. In total that is 64/120 vs 69/120, and a
two-proportion test gives p ≈ 0.5.

The paired comparison uses `tools/bench_masks.py --seed-bases 0,300 --runs 300`. Both models get 600 requests at
concurrency 30, and request i is sampled with seed base+i on both. `tools/paired_compare.py` matches requests by seed
and runs an exact McNemar test on the requests where the two models disagree. As a check, base rerun against itself
gave the identical answer on 57/60 requests.

| model | GSM8K | estonia pass (600 paired) | only base / only model | vs base |
|---|---|---|---|---|
| base | 48.4 | 48.0% | – | – |
| refit, step 100 | 64.1 | 48.5% | 33 / 36 | +0.5 pts, p = 0.81 |
| refit, step 125 | 65.0 | 46.0% | 39 / 27 | −2.0 pts, p = 0.18 |

The 16-point GSM8K gain comes with no measurable change in long-context retrieval, within about ±3 points.

## Dashboard

```bash
python3 dashboard/server.py            # http://127.0.0.1:8765  (stdlib only; --host 0.0.0.0 to expose, --port, --exclude)
ssh -L 8765:localhost:8765 <this-box>  # if viewing from another machine
```
Reads every `runs/*/log.jsonl` (hides `smoke`, `timing`), refreshes every 15s: per-run status tiles (step, latest/best
test acc, Δ vs step 0, ETA), test accuracy / reward / deletion fraction / response length vs. step, and an eval table
with mean ± std across runs.

## Presentation site

`site/index.html` is a self-contained, scrollable slide deck explaining the method, the paper's results, and this experiment's
runs (interactive α slider, Wanda toy, GRPO loop, hover tooltips on every chart). It reads `site/data.js`; open it directly
(`xdg-open site/index.html`) or serve it (`python3 -m http.server -d site 8000`). ↑/↓ or PgUp/PgDn step between slides.

```bash
docker run --rm -v "$PWD:/w" -w /w deletion-tuning python site/analyze_masks.py   # best_mask.pt -> site/mask_stats.json (CPU)
python3 site/build_data.py                                                       # runs/*/log.jsonl + mask stats -> site/data.js
```
Re-run both when runs progress; `?only=<slide-id>` renders a single slide (for screenshots).

## Paper settings vs. this setup
- Same: gate_proj-only mask, α (1e3→1e4), LR 3e-4 constant AdamW, λ=1e-5, Wanda init η=1/2048, n=4, no KL, greedy eval.
- Scaled down: prompt batch 32 (paper 64), max response 768 tokens (paper 4096), Wanda stats from 512 prompts (paper 2048),
  200 steps (paper ≈3 epochs), eval on first 500 GSM8K test problems. Non-thinking chat mode.
- One gradient step per rollout batch → PPO ratio ≡ 1, so the clipped objective is plain −A·log π.
- `causal_conv1d` isn't installed (runtime image has no nvcc), so the DeltaNet conv uses the PyTorch fallback — correct, a bit slower.

Natural next step: add FFT / LoRA / random-sparse baselines and the retention suite to compare forgetting.
