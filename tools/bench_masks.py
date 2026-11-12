"""Run the llm-decode-bench `estonia` profile on many saved masks, one vLLM server per GPU, queue-style.

python3 tools/bench_masks.py [--profile lavd-test] --gpus 0,1,6,7 base runs/paper_seed0/mask_step25.pt ...
`base` serves the unmodified model. Each job: a throwaway container bakes the mask into /tmp/model (tools/apply_mask.py) and serves it; the bench runs
against it with the same settings as the baseline run; the container is then removed. Results:
runs/bench/<profile>_<seed>_<step>.{json,log}.
"""
import argparse, queue, re, subprocess, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--gpus", default="0,1,6,7")
ap.add_argument("--profile", default="estonia")
ap.add_argument("--seed-bases", default="", help="paired mode: comma-separated --completion-stats-seed bases; every "
                "model runs every base (request i gets seed base+i), names get _paired<base>")
ap.add_argument("--runs", type=int, default=0, help="--profile-runs per job (paired mode)")
ap.add_argument("--tag", default="", help="extra suffix for output names")
ap.add_argument("--repeat", type=int, default=0, help="N independent reruns per model: names get _r1.._rN and the "
                "vLLM server uses --seed 1..N (the single default run uses the server default seed)")
ap.add_argument("masks", nargs="+")
args = ap.parse_args()

items = []
for m in args.masks:
    if m == "base":
        items.append((None, "base"))
        continue
    if "refit_" in m:  # runs/<refit run>/refit_step<N>.pt
        run, step = re.search(r"(refit_[^/]+)/refit_step(\d+)\.pt", m).groups()
        items.append((m, f"{run}_step{step}"))
        continue
    seed, step = re.search(r"paper_seed(\d+)/mask_step(\d+)\.pt", m).groups()
    items.append((m, f"seed{seed}_step{step}"))
jobs = queue.Queue()
for sb in ([int(x) for x in args.seed_bases.split(",")] if args.seed_bases else [None]):
    for r in (range(1, args.repeat + 1) if args.repeat else [None]):
        for m, name in items:
            n = name + ("" if sb is None else f"_paired{sb}") + ("" if r is None else f"_r{r}") + args.tag
            extra = [] if sb is None else ["--completion-stats-seed", str(sb), "--profile-concurrency", "30"] + \
                (["--profile-runs", str(args.runs)] if args.runs else [])
            jobs.put((m, n, r, extra))
lock = threading.Lock()


def log(msg):
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def healthy(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    except Exception:
        return False


def worker(gpu):
    port = 8010 + gpu
    while True:
        try:
            mask, name, rep_seed, extra = jobs.get_nowait()
        except queue.Empty:
            return
        cname = f"vllm_bench_{name}"
        tag = f"{args.profile}_{name}"
        out_json = ROOT / f"runs/bench/{tag}.json"
        server_log = open(ROOT / f"runs/bench/server_{tag}.log", "w")
        log(f"GPU {gpu}: {name} -> baking + serving on :{port}")
        srv = subprocess.Popen(
            ["docker", "run", "--rm", "--name", cname, "--device", f"nvidia.com/gpu={gpu}", "--shm-size", "16g",
             "-p", f"127.0.0.1:{port}:8000", "-v", f"{Path.home()}/.cache/huggingface:/hf_cache",
             "-v", f"{ROOT}:/workspace/src:ro", "-w", "/workspace/src", "-e", "VLLM_ENABLE_V1_MULTIPROCESSING=1",
             "deletion-tuning-vllm", "bash", "-c",
             (f"python3 tools/apply_mask.py {mask} /tmp/model && " if mask else "")
             + f"exec vllm serve {'/tmp/model' if mask else 'Qwen/Qwen3.5-0.8B'} "
             f"--served-model-name {name} {f'--seed {rep_seed} ' if rep_seed else ''}--max-model-len 262144 --gpu-memory-utilization 0.9 --language-model-only"],
            stdout=server_log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while not healthy(port):
            if srv.poll() is not None or time.time() - t0 > 900:
                log(f"GPU {gpu}: {name} server FAILED to start (see runs/bench/server_{tag}.log)")
                subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
                break
            time.sleep(5)
        else:
            with open(ROOT / f"runs/bench/{tag}.log", "w") as f:
                rc = subprocess.run(["llm-decode-bench", "--host", "127.0.0.1", "--port", str(port), "--model", name,
                                     "--test-profile", args.profile, "--display-mode", "plain", "--output",
                                     str(out_json.relative_to(ROOT))] + extra, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
            text = (ROOT / f"runs/bench/{tag}.log").read_text(errors="replace")
            score = re.findall(r"│ score\s+│\s+([^│]*?)\s*│", text)
            log(f"GPU {gpu}: {name} done rc={rc}: {score[-1] if score else 'no score found'}")
            subprocess.run(["docker", "stop", cname], capture_output=True)
            srv.wait()


threads = [threading.Thread(target=worker, args=(int(g),)) for g in args.gpus.split(",")]
for t in threads:
    t.start()
for t in threads:
    t.join()
log("all done")
