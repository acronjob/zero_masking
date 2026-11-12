"""Deletion Tuning (arXiv:2611.10866): learn a binary mask over gate_proj weights.

Each masked layer keeps its pretrained weight W frozen and learns a logit matrix G.
  train:  W~(t) = W * sigmoid(alpha(t) * G),  alpha(t) linearly annealed alpha_start -> alpha_end
  eval / rollouts:  W^ = W * 1[G > 0]
Init (Wanda prior): s_ij = |W_ij| * ||X_j||_2; bottom eta fraction -> G ~ N(-0.03, 0.01^2),
rest -> G ~ N(+0.03, 0.01^2).  Loss = task_loss - lam * sum ||W~||^2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert linear.bias is None
        self.register_buffer("weight", linear.weight.detach().clone(), persistent=False)
        self.logits = nn.Parameter(torch.full_like(self.weight, 0.03, dtype=torch.float32))
        self.alpha = 1e3
        self.hard = False
        self._hard_cache = None
        # Wanda statistics: running sum of squared input activations per input feature
        self.collect_stats = False
        self.register_buffer("act_sq", torch.zeros(self.weight.shape[1], dtype=torch.float32), persistent=False)

    def masked_weight(self):
        if self.hard:
            if self._hard_cache is None:
                self._hard_cache = self.weight * (self.logits > 0).to(self.weight.dtype)
            return self._hard_cache
        return self.weight * torch.sigmoid(self.alpha * self.logits).to(self.weight.dtype)

    def forward(self, x):
        if self.collect_stats:
            self.act_sq += x.detach().float().pow(2).reshape(-1, x.shape[-1]).sum(0)
        return F.linear(x, self.masked_weight())

    def set_hard(self, hard: bool):
        self.hard = hard
        self._hard_cache = None


def wrap_gate_projs(model) -> list[MaskedLinear]:
    """Replace every text-decoder mlp.gate_proj with a MaskedLinear. Freezes everything else."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = [(n, m) for n, m in model.named_modules()
               if n.endswith("mlp.gate_proj") and "visual" not in n and isinstance(m, nn.Linear)]
    assert targets, "no gate_proj modules found"
    masked = []
    for name, lin in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        ml = MaskedLinear(lin).to(lin.weight.device)
        setattr(parent, "gate_proj", ml)
        masked.append(ml)
    return masked


@torch.no_grad()
def wanda_init(model, masked, batches, eta=1 / 2048, seed=0):
    """batches: iterable of dicts (input_ids, attention_mask) used to collect ||X_j||."""
    for m in masked:
        m.act_sq.zero_()
        m.collect_stats = True
        m.set_hard(True)  # logits are all +0.03 -> identity mask
    for b in batches:
        model(**b)
    g = torch.Generator(device=masked[0].weight.device).manual_seed(seed)
    for m in masked:
        m.collect_stats = False
        score = m.weight.float().abs() * m.act_sq.sqrt()[None, :]
        k = max(1, int(eta * score.numel()))
        thresh = score.flatten().kthvalue(k).values
        noise = torch.randn(score.shape, generator=g, device=score.device) * 0.01
        m.logits.data = torch.where(score <= thresh, -0.03, 0.03) + noise
        m.set_hard(False)


def set_alpha(masked, step, total, a_start, a_end):
    a = a_start + (step / max(1, total)) * (a_end - a_start)
    for m in masked:
        m.alpha = a
    return a


def set_hard(masked, hard: bool):
    for m in masked:
        m.set_hard(hard)


def norm_reg(masked):
    """sum ||W~||_2^2 over masked layers (soft mask)."""
    return sum(m.masked_weight().float().pow(2).sum() for m in masked)


@torch.no_grad()
def mask_stats(masked, total_model_params):
    deleted = sum(int((m.logits <= 0).sum()) for m in masked)
    gate_total = sum(m.logits.numel() for m in masked)
    return {"deleted": deleted,
            "frac_of_gate": deleted / gate_total,
            "frac_of_model": deleted / total_model_params}


def save_mask(masked, path):
    """Save the hard mask compactly: list of flat indices of deleted weights per layer."""
    torch.save([torch.nonzero((m.logits <= 0).flatten()).squeeze(1).int().cpu() for m in masked], path)
