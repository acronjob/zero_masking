"""Deletion Tuning + GRPO on GSM8K, following the authors' released recipe as closely as one GPU + HF allows.

Source of truth: github.com/agentic-learning-ai-lab/deleting-weights (verl/verl/trainer/config/gsm8k.yaml,
src/verl_adapters.py, src/wanda_calibration.py, src/learnable_sparsity/*, verl dp_actor.py / core_algos.py).

Matches the repo:
  data      verl gsm8k preprocessing (question + ' Let's think step by step and output the final answer after "####".'),
            model's default chat template, train_batch_size 64, 3 epochs (drop_last -> 116 steps/epoch = 348 steps),
            shuffled with data seed 2, max prompt 512, max response 4096
  rollout   n=4, temperature 1.0, top_p 1, top_k off, hard mask (logit > 0)
  reward    verl gsm8k strict: last '#### <num>' in the final 300 chars, exact string match with the ground truth
  advantage GRPO, (r - mean) / (std + 1e-6) per group, no KL
  update    old log-probs from the hard-mask policy; 4 mini-batches of 16 prompts (64 seqs) per step, 1 PPO epoch;
            vanilla clipped PPO (eps 0.2, dual-clip 3.0); token-mean within each dynamic micro-batch
            (<= 32768 tokens), scaled by micro/mini sequence count; AdamW lr 3e-4 constant, betas (0.9, 0.999),
            wd 0.01, grad-clip 1.0
  mask      gate_proj only; soft mask sigmoid(alpha * G), alpha 1e3 -> 1e4 linear over all optimizer steps (1392);
            loss -= lambda * (1 - progress) * sum ||W * soft_mask||^2 with lambda 1e-5
  init      Wanda on 2048 train prompts (pandas sample, random_state=42), right-padded batches of 32, truncation 512;
            score |W| * sum_batches ||X_j||_2; delete score < kth value (k = numel / 2048); G ~ N(0, 0.01^2) shifted
            by -/+ 3 std
  eval      greedy, full GSM8K test (1319), before training, every 25 steps and at the last step
Differs (unavoidable / by request): Qwen3.5-0.8B (paper: Llama 3.2 Instruct); rollouts in vLLM 0.33 in-process
(paper: vLLM 0.11 via verl), or HF generate with --engine hf;
micro-batches executed as padded length-sorted chunks (same loss weights as verl's packed micro-batches).

Refit mode (--refit_mask PATH): instead of learning a mask, train the real values of a fixed set of gate_proj
coordinates (the ones a saved Deletion Tuning mask deleted, or --refit_random: the same number of random coordinates
per layer). --refit_init zero starts them at 0 (= the deletion-tuned model), orig at their original values. Same GRPO
recipe; no Wanda init, sharpness schedule or mask regularizer. Evals save refit_step{N}.pt ({"idx", "delta"} per layer).
"""
import argparse, json, math, os, random, re, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
p.add_argument("--seed", type=int, default=0, help="init noise + rollout sampling")
p.add_argument("--data_seed", type=int, default=2)
p.add_argument("--train_batch_size", type=int, default=64)
p.add_argument("--n", type=int, default=4)
p.add_argument("--ppo_mini_batch_size", type=int, default=16, help="prompts per optimizer step")
p.add_argument("--ppo_max_token_len", type=int, default=32768)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--max_prompt_length", type=int, default=512)
p.add_argument("--max_response_length", type=int, default=4096)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight_decay", type=float, default=0.01)
p.add_argument("--grad_clip", type=float, default=1.0)
p.add_argument("--clip_ratio", type=float, default=0.2)
p.add_argument("--clip_ratio_c", type=float, default=3.0)
p.add_argument("--sharpness_start", type=float, default=1e3)
p.add_argument("--sharpness_end", type=float, default=1e4)
p.add_argument("--mask_weight_reg", type=float, default=1e-5)
p.add_argument("--init_sparsity", type=float, default=1 / 2048)
p.add_argument("--prior_strength", type=float, default=3.0)
p.add_argument("--calib_samples", type=int, default=2048)
p.add_argument("--test_freq", type=int, default=25)
p.add_argument("--engine", choices=["vllm", "hf"], default="vllm", help="rollout/eval engine (paper: vLLM)")
p.add_argument("--vllm_mem", type=float, default=0.35, help="vLLM gpu_memory_utilization")
p.add_argument("--gen_bs", type=int, default=256, help="sequences per generate() call (throughput only)")
p.add_argument("--phys_tokens", type=int, default=16384, help="padded tokens per physical forward (memory only)")
p.add_argument("--eval_limit", type=int, default=0, help="smoke tests only: eval on first N test problems")
p.add_argument("--eval_only", action="store_true", help="step-0 eval then exit (engine comparison)")
p.add_argument("--max_steps", type=int, default=0, help="stop early (timing tests); schedules still use full length")
p.add_argument("--out", default="/workspace/runs/paper_seed0")
p.add_argument("--refit_mask", default="", help="train the values of the coordinates this saved mask deleted")
p.add_argument("--refit_init", choices=["zero", "orig"], default="zero")
p.add_argument("--refit_random", action="store_true", help="random coordinates instead (same count per layer)")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
random.seed(args.seed); torch.manual_seed(args.seed)
dev = "cuda"

# ---------------- data (verl examples/data_preprocess/gsm8k.py) ----------------
INSTR = 'Let\'s think step by step and output the final answer after "####".'
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def gt_of(ans):
    return re.search("#### (\\-?[0-9\\.\\,]+)", ans).group(0).split("#### ")[1].replace(",", "")


def chat(q):
    return tok.apply_chat_template([{"role": "user", "content": q + " " + INSTR}], tokenize=False,
                                   add_generation_prompt=True)


def reward(text, gt):  # verl/utils/reward_score/gsm8k.py, method="strict"
    text = text[-300:]
    sols = re.findall("#### (\\-?[0-9\\.\\,]+)", text)
    if not sols:
        return 0.0
    return 1.0 if sols[-1].replace(",", "").replace("$", "") == gt else 0.0


ds = load_dataset("openai/gsm8k", "main")
train_df = ds["train"].to_pandas()
train = [(tok(chat(q))["input_ids"], gt_of(a)) for q, a in zip(train_df["question"], train_df["answer"])]
train = [x for x in train if len(x[0]) <= args.max_prompt_length]  # filter_overlong_prompts
test = [(tok(chat(r["question"]))["input_ids"], gt_of(r["answer"])) for r in ds["test"]]
if args.eval_limit:
    test = test[:args.eval_limit]
steps_per_epoch = len(train) // args.train_batch_size  # drop_last
total_steps = steps_per_epoch * args.epochs
mini_per_step = args.train_batch_size // args.ppo_mini_batch_size
total_opt_steps = total_steps * mini_per_step
json.dump({**vars(args), "steps": total_steps, "total_opt_steps": total_opt_steps},
          open(os.path.join(args.out, "config.json"), "w"), indent=1)

# ---------------- rollout engine ----------------
# vLLM is created first so its memory reservation succeeds; it sleeps (weights to CPU, KV cache freed) during
# training and wakes for rollouts/eval, like verl's hybrid engine. Masked gate_proj weights are pushed in place
# before every generation.
llm = None
if args.engine == "vllm":
    from vllm import LLM, SamplingParams
    llm = LLM(args.model, gpu_memory_utilization=args.vllm_mem, seed=args.seed, enable_sleep_mode=True,
              max_model_len=args.max_prompt_length + args.max_response_length,
              limit_mm_per_prompt={"image": 0, "video": 0})
    llm.sleep(level=1)
    vllm_awake = [False]

# ---------------- model + masks (src/learnable_sparsity) ----------------
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
n_params = sum(p.numel() for p in model.parameters())


class SparseLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.register_buffer("weight", lin.weight.detach().clone(), persistent=False)
        self.gate = nn.Parameter(torch.empty(self.weight.shape, dtype=torch.float32, device=self.weight.device))
        nn.init.normal_(self.gate, std=0.01)
        self.sharpness = args.sharpness_start
        self.mode = "soft"  # soft | hard | dense
        self._hard = None
        self.idx = None  # refit mode: flat indices of trainable coordinates

    def setup_refit(self, idx, init):
        self.idx = idx.to(self.weight.device)
        del self.gate
        v = self.weight.view(-1)[self.idx].float()
        self.delta = nn.Parameter(torch.zeros_like(v) if init == "zero" else v.clone())

    def eff_weight(self):
        return self.weight.view(-1).index_put((self.idx,), self.delta.to(self.weight.dtype)).view_as(self.weight)

    def forward(self, x):
        if self.idx is not None:
            return F.linear(x, self.eff_weight())
        if self.mode == "dense":
            return F.linear(x, self.weight)
        if self.mode == "hard":
            if self._hard is None:
                self._hard = self.weight * (self.gate > 0).to(self.weight.dtype)
            return F.linear(x, self._hard)
        return F.linear(x, self.soft_weight())

    def soft_weight(self):
        return self.weight * torch.sigmoid(self.gate * self.sharpness).to(self.weight.dtype)


for prm in model.parameters():
    prm.requires_grad_(False)
masked = []
for name, mod in list(model.named_modules()):
    for cname, child in list(mod.named_children()):
        if isinstance(child, nn.Linear) and "gate_proj" in f"{name}.{cname}":
            sl = SparseLinear(child); setattr(mod, cname, sl); masked.append(sl)
            # checkpoint tensor name, used to push the hard-masked weight into vLLM
            sl.ckpt_name = "model.language_model." + f"{name}.{cname}".removeprefix("model.") + ".weight"
gate_total = sum(m.weight.numel() for m in masked)
print(f"masked {len(masked)} gate_proj layers ({gate_total/1e6:.1f}M logits); {total_steps} steps, "
      f"{total_opt_steps} optimizer steps", flush=True)


def set_mode(mode):
    for m in masked:
        m.mode = mode; m._hard = None


# Wanda prior (src/wanda_calibration.py + verl_adapters._wanda_scores)
@torch.no_grad()
def wanda_init():
    df = train_df.sample(n=args.calib_samples, random_state=42).reset_index(drop=True) \
        if len(train_df) > args.calib_samples else train_df
    texts = [chat(q) for q in df["question"]]
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    norms = [torch.zeros(m.weight.shape[1], device=dev) for m in masked]
    def hook(i):
        def f(mod, inp, out):
            x = inp[0].detach().float()
            norms[i].add_(x.pow(2).sum(dim=tuple(range(x.dim() - 1))).sqrt())
        return f
    hooks = [m.register_forward_hook(hook(i)) for i, m in enumerate(masked)]
    set_mode("dense"); model.eval()
    for i in range(0, len(texts), 32):
        model(input_ids=enc["input_ids"][i:i + 32].to(dev), attention_mask=enc["attention_mask"][i:i + 32].to(dev))
    for h in hooks:
        h.remove()
    for m, nrm in zip(masked, norms):
        s = (m.weight.float().abs() * nrm[None, :]).view(-1)
        k = max(1, int(s.numel() * args.init_sparsity))
        thr = s.kthvalue(k).values.item()
        sign = torch.where(s < thr, -1.0, 1.0).view_as(m.gate)
        m.gate.data += sign * m.gate.std().item() * args.prior_strength
    set_mode("soft")


REFIT = bool(args.refit_mask)
if REFIT:
    ref = torch.load(args.refit_mask)
    assert len(ref) == len(masked)
    rg = torch.Generator().manual_seed(1234)
    for m, idx in zip(masked, ref):
        if args.refit_random:
            idx = torch.randperm(m.weight.numel(), generator=rg)[:idx.numel()]
        m.setup_refit(idx.long(), args.refit_init)
    n_refit = sum(m.delta.numel() for m in masked)
    print(f"refit: {n_refit} coordinates ({'random' if args.refit_random else args.refit_mask}), "
          f"init={args.refit_init}", flush=True)
else:
    wanda_init()


def mask_stats():
    if REFIT:
        d = torch.cat([m.delta.detach() for m in masked])
        o = torch.cat([m.weight.view(-1)[m.idx].float() for m in masked])
        return {"n_trainable": d.numel(), "delta_rms": d.pow(2).mean().sqrt().item(),
                "orig_rms": o.pow(2).mean().sqrt().item(), "dist_from_orig_rms": (d - o).pow(2).mean().sqrt().item()}
    deleted = sum(int((m.gate <= 0).sum()) for m in masked)
    return {"deleted": deleted, "frac_of_gate": deleted / gate_total, "frac_of_model": deleted / n_params}


print("after wanda init:", mask_stats(), flush=True)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
gates = [m.delta for m in masked] if REFIT else [m.gate for m in masked]
opt = torch.optim.AdamW(gates, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
eos_ids = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
                 | ({model.generation_config.eos_token_id} if isinstance(model.generation_config.eos_token_id, int)
                    else set(model.generation_config.eos_token_id or [])))
pad_id = tok.pad_token_id
gen_seed = [args.seed * 1_000_003]


# ---------------- generation (hard mask) ----------------
@torch.no_grad()
def vllm_push():
    """Wake vLLM, write W * 1[G > 0] into its gate_proj in place, and verify by reading it back."""
    if not vllm_awake[0]:
        llm.wake_up(); vllm_awake[0] = True
    hard = [(m.ckpt_name, m.eff_weight() if REFIT else m.weight * (m.gate > 0).to(m.weight.dtype)) for m in masked]
    llm.apply_model(lambda mdl: mdl.load_weights(hard))
    llm.reset_prefix_cache()
    # vLLM fuses gate_proj and up_proj into gate_up_proj ([gate; up]); the gate half must equal our hard weight
    ok = llm.apply_model(lambda mdl: all(
        torch.equal(dict(mdl.named_parameters())[f"language_model.model.layers.{i}.mlp.gate_up_proj.weight"]
                    [:w.shape[0]], w) for i, (_, w) in enumerate(hard)))
    assert all(ok), "vLLM gate_proj does not match the hard mask after push"


def vllm_sleep():
    if vllm_awake[0]:
        llm.sleep(level=1); vllm_awake[0] = False
    torch.cuda.empty_cache()


@torch.no_grad()
def generate(prompts, sample):
    if llm is not None:
        vllm_push()
        sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_response_length,
                            stop_token_ids=eos_ids) if sample else \
            SamplingParams(temperature=0.0, max_tokens=args.max_response_length, stop_token_ids=eos_ids)
        outs = llm.generate([{"prompt_token_ids": pp} for pp in prompts], sp, use_tqdm=False)
        vllm_sleep()
        res = []
        for o in outs:
            row = list(o.outputs[0].token_ids)
            cut = next((k + 1 for k, t in enumerate(row) if t in eos_ids), len(row))
            res.append(row[:cut])
        return res
    set_mode("hard"); model.eval(); model.config.use_cache = True
    outs = []
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths -> less padding
    for i in range(0, len(order), args.gen_bs):
        idx = order[i:i + args.gen_bs]
        L = max(len(prompts[j]) for j in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for r, j in enumerate(idx):
            ids[r, L - len(prompts[j]):] = torch.tensor(prompts[j]); att[r, L - len(prompts[j]):] = 1
        if sample:
            gen_seed[0] += 1; torch.manual_seed(gen_seed[0])
        kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
        o = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=args.max_response_length,
                           eos_token_id=eos_ids, pad_token_id=pad_id, use_cache=True, **kw)[:, L:].tolist()
        for r, j in enumerate(idx):
            row = o[r]
            cut = next((k + 1 for k, t in enumerate(row) if t in eos_ids), len(row))
            outs.append((j, row[:cut]))
    model.config.use_cache = False; set_mode("soft")
    return [c for _, c in sorted(outs)]


def evaluate():
    comps = generate([x for x, _ in test], sample=False)
    return sum(reward(tok.decode(c, skip_special_tokens=True), g) for c, (_, g) in zip(comps, test)) / len(test)


# ---------------- log-probs ----------------
def _lp_chunk(h, tgt):
    return torch.gather(model.lm_head(h).float().log_softmax(-1), -1, tgt[..., None]).squeeze(-1)


def seq_logprobs(seqs, n_prompt, grad):
    """Per-sequence response-token log-probs for a list of full (prompt+response) token lists."""
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    ids, att = ids.to(dev), att.to(dev)
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    B, T, D = h.shape
    hf, tf = h.reshape(-1, D), ids[:, 1:].reshape(-1)
    lps = []
    for c in range(0, hf.shape[0], 4096):  # bounded fp32 logits: 4096 x vocab
        hc, tc = hf[c:c + 4096], tf[c:c + 4096]
        lps.append(checkpoint(_lp_chunk, hc, tc, use_reentrant=False) if grad else _lp_chunk(hc, tc))
    lp = torch.cat(lps).view(B, T)
    return [lp[i, n_prompt[i] - 1:len(s) - 1] for i, s in enumerate(seqs)]


def phys_chunks(idx, lens):
    """Split indices into length-sorted padded chunks of <= phys_tokens tokens."""
    idx = sorted(idx, key=lambda i: lens[i]); out, cur = [], []
    for i in idx:
        if cur and (len(cur) + 1) * lens[i] > args.phys_tokens:
            out.append(cur); cur = []
        cur.append(i)
    return out + [cur] if cur else out


def dynamic_micro_batches(idx, lens):
    """verl prepare_dynamic_batch: ceil(total/max_token_len) micro-batches balanced by token count."""
    k = max(1, math.ceil(sum(lens[i] for i in idx) / args.ppo_max_token_len))
    groups = [[] for _ in range(k)]; load = [0] * k
    for i in sorted(idx, key=lambda i: -lens[i]):
        g = load.index(min(load)); groups[g].append(i); load[g] += lens[i]
    return [g for g in groups if g]


# ---------------- training ----------------
log = open(os.path.join(args.out, "log.jsonl"), "a")


def do_eval(step):
    t0 = time.time(); acc = evaluate()
    rec = {"step": step, "eval_acc": acc, **mask_stats(), "eval_s": round(time.time() - t0, 1)}
    print("EVAL", rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
    if REFIT:
        torch.save([{"idx": m.idx.int().cpu(), "delta": m.delta.detach().cpu()} for m in masked],
                   os.path.join(args.out, f"refit_step{step}.pt"))
    else:
        torch.save([torch.nonzero((m.gate <= 0).flatten()).squeeze(1).int().cpu() for m in masked],
                   os.path.join(args.out, f"mask_step{step}.pt"))


do_eval(0)  # val_before_train
if args.eval_only:
    raise SystemExit(0)
g = torch.Generator().manual_seed(args.data_seed)
opt_step = 0
step = 0
for epoch in range(args.epochs):
    perm = torch.randperm(len(train), generator=g).tolist()
    for b in range(steps_per_epoch):
        step += 1
        t0 = time.time()
        batch = [train[j] for j in perm[b * args.train_batch_size:(b + 1) * args.train_batch_size]]
        prompts = [x for x, _ in batch for _ in range(args.n)]
        gts = [y for _, y in batch for _ in range(args.n)]
        comps = generate(prompts, sample=True)
        t_gen = time.time() - t0
        rewards = torch.tensor([reward(tok.decode(c, skip_special_tokens=True), y) for c, y in zip(comps, gts)])
        r = rewards.view(-1, args.n)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).flatten().tolist()
        seqs = [pp + c for pp, c in zip(prompts, comps)]
        n_prompt = [len(pp) for pp in prompts]
        lens = [len(s) for s in seqs]

        # old log-probs under the hard mask (verl compute_log_prob runs the actor in eval mode)
        set_mode("hard"); model.eval(); old_lp = [None] * len(seqs)
        with torch.no_grad():
            for ch in phys_chunks(range(len(seqs)), lens):
                for i, lp in zip(ch, seq_logprobs([seqs[i] for i in ch], [n_prompt[i] for i in ch], grad=False)):
                    old_lp[i] = lp
        set_mode("soft"); model.train()

        mstats = {"pg_loss": 0.0, "clipfrac": 0.0, "ppo_kl": 0.0, "grad_norm": [], "reg": 0.0}
        mb_seqs = args.ppo_mini_batch_size * args.n
        for mb in range(mini_per_step):
            progress = min(opt_step / total_opt_steps, 1.0)
            sharp = args.sharpness_start + (args.sharpness_end - args.sharpness_start) * progress
            for m in masked:
                m.sharpness = sharp
            lam = 0.0 if REFIT else args.mask_weight_reg * (1.0 - progress)
            mb_idx = list(range(mb * mb_seqs, (mb + 1) * mb_seqs))
            opt.zero_grad(set_to_none=True)
            for micro in dynamic_micro_batches(mb_idx, lens):
                scale = len(micro) / mb_seqs
                n_tok = sum(len(comps[i]) for i in micro)
                for ch in phys_chunks([i for i in micro if adv[i] != 0.0], lens):
                    lps = seq_logprobs([seqs[i] for i in ch], [n_prompt[i] for i in ch], grad=True)
                    pg_sum = 0.0
                    for i, lp in zip(ch, lps):
                        old = old_lp[i].to(lp.device)
                        logratio = (lp - old).clamp(-20, 20); ratio = logratio.exp()
                        A = adv[i]
                        l1 = -A * ratio; l2 = -A * ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio)
                        cl = torch.maximum(l1, l2)
                        if A < 0:
                            cl = torch.minimum(cl, torch.full_like(cl, -A * args.clip_ratio_c))
                        pg_sum = pg_sum + cl.sum()
                        mstats["clipfrac"] += float((l2 > l1).float().sum()) / n_tok * scale / mini_per_step
                        mstats["ppo_kl"] += float((-logratio).detach().sum()) / n_tok * scale / mini_per_step
                    loss = pg_sum / n_tok * scale
                    loss.backward()
                    mstats["pg_loss"] += float(pg_sum) / n_tok * scale / mini_per_step
            # regularizer: verl subtracts lam*||W~||^2*scale in every micro-batch; the scales sum to 1 and the mask
            # is fixed within an optimizer step, so this single term has the same value and gradient
            if lam > 0:
                reg = sum(m.soft_weight().pow(2).sum().float() for m in masked)
                (-lam * reg).backward()
                mstats["reg"] += float(lam * reg) / mini_per_step
            gn = torch.nn.utils.clip_grad_norm_(gates, args.grad_clip)
            if torch.isfinite(gn):
                opt.step()
            mstats["grad_norm"].append(float(gn))
            opt_step += 1

        rec = {"step": step, "epoch": epoch, "reward": rewards.mean().item(), **{k: v for k, v in mstats.items()
               if k != "grad_norm"}, "grad_norm": max(mstats["grad_norm"]), "alpha": sharp, "lambda": lam,
               **({"delta_rms": mask_stats()["delta_rms"]} if REFIT else {"frac_deleted_gate": mask_stats()["frac_of_gate"]}), "mean_len": sum(map(len, comps)) / len(comps),
               "clip_len_frac": sum(len(c) >= args.max_response_length for c in comps) / len(comps),
               "gen_s": round(t_gen, 1), "step_s": round(time.time() - t0, 1)}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

        if step % args.test_freq == 0 or step == total_steps:
            do_eval(step)
        if args.max_steps and step >= args.max_steps:
            raise SystemExit(0)
print("done", flush=True)
