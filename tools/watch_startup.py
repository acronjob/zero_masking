"""Print each seed's first EVAL, first training step, or first error once; exit when every seed is past step 1."""
import re, sys, time
from pathlib import Path

runs = sys.argv[1:] or [f"paper_seed{s}" for s in range(4)]
seen, done = set(), set()
pat_err = re.compile(r"Traceback|AssertionError|OutOfMemory|CUDA error|RuntimeError")
while len(done) < len(runs):
    for r in runs:
        f = Path(__file__).resolve().parent.parent / "runs" / f"{r}.out"
        if not f.exists():
            continue
        for line in f.read_text(errors="replace").splitlines():
            key = None
            if pat_err.search(line):
                key, msg = (r, "err"), f"{r} ERROR: {line[:200]}"
                done.add(r)
            elif line.startswith("EVAL"):
                key, msg = (r, "eval"), f"{r}: {line[:110]}"
            elif line.startswith("{'step': 1,"):
                m = re.search(r"'gen_s': [0-9.]+, 'step_s': [0-9.]+", line)
                key, msg = (r, "step1"), f"{r}: step 1 done ({m.group(0) if m else ''})"
                done.add(r)
            if key and key not in seen:
                seen.add(key); print(msg, flush=True)
    time.sleep(5)
