"""GRPO on GSM8K with Deletion Tuning (arXiv:2611.10866) — single-GPU, plain HF generate.

Only gate_proj mask logits are trained; all pretrained weights are frozen.
One policy-gradient step per rollout batch (fully on-policy), so the PPO ratio is 1 and the
clipped GRPO objective reduces to -A * log pi; no KL term (matches the paper's GSM8K setting).
"""
import argparse, json, math, os, random, re, time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import deletion_tuning as dt

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
p.add_argument("--steps", type=int, default=200)
p.add_argument("--prompt_bs", type=int, default=32)
p.add_argument("--n", type=int, default=4, help="rollouts per prompt")
p.add_argument("--max_new_tokens", type=int, default=768)
p.add_argument("--micro_bs", type=int, default=8)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--alpha_start", type=float, default=1e3)
p.add_argument("--alpha_end", type=float, default=1e4)
p.add_argument("--lam", type=float, default=1e-5)
p.add_argument("--eta", type=float, default=1 / 2048)
p.add_argument("--wanda_prompts", type=int, default=512)
p.add_argument("--eval_every", type=int, default=25)
p.add_argument("--eval_n", type=int, default=500)
p.add_argument("--eval_bs", type=int, default=128)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="/workspace/runs/dt_gsm8k")
args = p.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
dev = "cuda"

# ---------------- data ----------------
SUFFIX = ' Let\'s think step by step and output the final answer after "####".'
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"


def fmt(q):
    return tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def reward(text, g):
    tail = text.split("####")[-1] if "####" in text else text
    nums = NUM.findall(tail)
    if not nums:
        return 0.0
    pred = nums[0] if "####" in text else nums[-1]
    try:
        return float(abs(float(pred.replace(",", "").rstrip(".")) - float(g)) < 1e-6)
    except ValueError:
        return 0.0


ds = load_dataset("openai/gsm8k", "main")
train = [(fmt(r["question"]), gold(r["answer"])) for r in ds["train"]]
test = [(fmt(r["question"]), gold(r["answer"])) for r in ds["test"]][: args.eval_n]

# ---------------- model ----------------
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
n_text_params = sum(p.numel() for n, p in model.named_parameters() if "visual" not in n)
masked = dt.wrap_gate_projs(model)
print(f"masked {len(masked)} gate_proj layers, {sum(m.logits.numel() for m in masked)/1e6:.1f}M logits; "
      f"text params {n_text_params/1e6:.1f}M")
eos_ids = model.generation_config.eos_token_id
eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
pad_id = tok.pad_token_id if tok.pad_token_id is not None else next(iter(eos_ids))

# Wanda init of mask logits
model.eval()
wp = random.sample(train, args.wanda_prompts)
batches = [tok([x for x, _ in wp[i:i + 32]], return_tensors="pt", padding=True).to(dev)
           for i in range(0, len(wp), 32)]
dt.wanda_init(model, masked, batches, eta=args.eta, seed=args.seed)
print("after wanda init:", dt.mask_stats(masked, n_text_params))

model.gradient_checkpointing_enable()
model.config.use_cache = False
opt = torch.optim.AdamW([m.logits for m in masked], lr=args.lr, weight_decay=0.0)


@torch.no_grad()
def generate(prompts, sample):
    dt.set_hard(masked, True)
    model.eval()
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=sample,
                         temperature=args.temperature if sample else None, top_p=None, top_k=None,
                         pad_token_id=pad_id, use_cache=True)
    comp = out[:, enc["input_ids"].shape[1]:]
    res = []
    for row in comp.tolist():
        cut = next((i + 1 for i, t in enumerate(row) if t in eos_ids), len(row))
        res.append(row[:cut])
    return res


def evaluate():
    correct = 0
    for i in range(0, len(test), args.eval_bs):
        chunk = test[i:i + args.eval_bs]
        comps = generate([x for x, _ in chunk], sample=False)
        correct += sum(reward(tok.decode(c, skip_special_tokens=True), g) for c, (_, g) in zip(comps, chunk))
    return correct / len(test)


def pg_loss(prompt_ids, comp_ids, advs, n_seqs):
    """Right-padded [prompt|completion] batch -> sum_seq A * mean_tok log pi, scaled by 1/n_seqs."""
    seqs = [p + c for p, c in zip(prompt_ids, comp_ids)]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros_like(ids)
    cmask = torch.zeros(len(seqs), L, dtype=torch.float32)
    for i, (p_, c) in enumerate(zip(prompt_ids, comp_ids)):
        ids[i, :len(p_) + len(c)] = torch.tensor(p_ + c)
        attn[i, :len(p_) + len(c)] = 1
        cmask[i, len(p_):len(p_) + len(c)] = 1
    ids, attn, cmask = ids.to(dev), attn.to(dev), cmask.to(dev)
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    logp = torch.gather(logits.log_softmax(-1), 2, ids[:, 1:, None]).squeeze(-1)
    m = cmask[:, 1:]
    seq_logp = (logp * m).sum(1) / m.sum(1).clamp(min=1)
    return -(torch.tensor(advs, device=dev) * seq_logp).sum() / n_seqs


log = open(os.path.join(args.out, "log.jsonl"), "a")
best = -1.0
order = list(range(len(train))); random.shuffle(order); ptr = 0
for step in range(args.steps + 1):
    if step % args.eval_every == 0 or step == args.steps:
        t0 = time.time()
        acc = evaluate()
        st = dt.mask_stats(masked, n_text_params)
        rec = {"step": step, "eval_acc": acc, **st, "eval_s": round(time.time() - t0, 1)}
        print("EVAL", rec); log.write(json.dumps(rec) + "\n"); log.flush()
        if acc > best:
            best = acc
            dt.save_mask(masked, os.path.join(args.out, "best_mask.pt"))
    if step == args.steps:
        break

    t0 = time.time()
    if ptr + args.prompt_bs > len(order):
        random.shuffle(order); ptr = 0
    batch = [train[j] for j in order[ptr:ptr + args.prompt_bs]]; ptr += args.prompt_bs
    prompts = [x for x, _ in batch for _ in range(args.n)]
    golds = [g for _, g in batch for _ in range(args.n)]
    comps = generate(prompts, sample=True)
    rewards = torch.tensor([reward(tok.decode(c, skip_special_tokens=True), g) for c, g in zip(comps, golds)])
    r = rewards.view(-1, args.n)
    advs = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).flatten().tolist()
    t_gen = time.time() - t0

    # policy update with soft mask
    alpha = dt.set_alpha(masked, step, args.steps, args.alpha_start, args.alpha_end)
    dt.set_hard(masked, False)
    model.train()
    opt.zero_grad(set_to_none=True)
    prompt_ids = [tok(x)["input_ids"] for x in prompts]
    keep = [i for i, a in enumerate(advs) if a != 0.0]  # zero-advantage seqs contribute no gradient
    pg = 0.0
    for i in range(0, len(keep), args.micro_bs):
        idx = keep[i:i + args.micro_bs]
        loss = pg_loss([prompt_ids[k] for k in idx], [comps[k] for k in idx], [advs[k] for k in idx], len(prompts))
        loss.backward(); pg += loss.item()
    reg = -args.lam * dt.norm_reg(masked)
    reg.backward()
    opt.step()

    rec = {"step": step, "reward": rewards.mean().item(), "pg_loss": pg, "reg": reg.item(), "alpha": alpha,
           "frac_deleted_gate": dt.mask_stats(masked, n_text_params)["frac_of_gate"],
           "mean_len": sum(map(len, comps)) / len(comps), "n_active": len(keep),
           "gen_s": round(t_gen, 1), "step_s": round(time.time() - t0, 1)}
    print(rec); log.write(json.dumps(rec) + "\n"); log.flush()

print(f"best eval acc {best:.4f}; mask saved to {args.out}/best_mask.pt")
