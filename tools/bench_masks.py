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
ap.add_argument("masks", nargs="+")
args = ap.parse_args()

jobs = queue.Queue()
for m in args.masks:
    if m == "base":
        jobs.put((None, "base"))
        continue
    seed, step = re.search(r"paper_seed(\d+)/mask_step(\d+)\.pt", m).groups()
    jobs.put((m, f"seed{seed}_step{step}"))
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
            mask, name = jobs.get_nowait()
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
             f"--served-model-name {name} --max-model-len 262144 --gpu-memory-utilization 0.9 --language-model-only"],
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
                                     str(out_json.relative_to(ROOT))], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
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
