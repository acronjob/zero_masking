"""Paired comparison of two bench_masks.py paired-mode runs (same --completion-stats-seed bases).

python3 tools/paired_compare.py estonia base refit_zero_lr1e-4_s0_step100 [--bases 0,300] [--tag-b _dup]
Pairs runs/bench/<profile>_<A>_paired<base>.json with ..._<B>_paired<base><tag>.json by run_index (request i has
seed base+i in both) and reports both/neither/discordant counts and an exact two-sided McNemar test.
"""
import argparse, json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("profile"); ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--bases", default="0,300")
ap.add_argument("--tag-a", default=""); ap.add_argument("--tag-b", default="")
args = ap.parse_args()


def load(name, base, tag):
    d = json.load(open(ROOT / f"runs/bench/{args.profile}_{name}_paired{base}{tag}.json"))
    assert d["metadata"].get("seed_base") == base, (name, base, d["metadata"].get("seed_base"))
    return {r["run_index"]: r for r in d["runs"] if r.get("phase") == "profile"}


both = a_only = b_only = neither = same_answer = 0
for base in map(int, args.bases.split(",")):
    A, B = load(args.a, base, args.tag_a), load(args.b, base, args.tag_b)
    for i in sorted(A.keys() & B.keys()):
        ca, cb = bool(A[i]["correct"]), bool(B[i]["correct"])
        both += ca and cb; neither += not ca and not cb; a_only += ca and not cb; b_only += cb and not ca
        same_answer += (A[i].get("final_answer") or "").strip() == (B[i].get("final_answer") or "").strip()
n = both + a_only + b_only + neither
d = a_only + b_only
p = min(1.0, 2 * sum(comb(d, k) for k in range(min(a_only, b_only) + 1)) / 2 ** d) if d else 1.0
pa, pb = (both + a_only) / n, (both + b_only) / n
print(f"pairs {n}  |  {args.a}: {pa*100:.1f}%  {args.b}: {pb*100:.1f}%  diff {(pb-pa)*100:+.1f} pts")
print(f"both pass {both}, both fail {neither}, only {args.a} {a_only}, only {args.b} {b_only}  "
      f"(identical final answer in {same_answer}/{n} = {same_answer/n*100:.0f}%)")
print(f"exact McNemar p = {p:.3g}")
