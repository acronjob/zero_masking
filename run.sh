#!/usr/bin/env bash
# Usage: [GPU=3] [IMAGE=deletion-tuning|deletion-tuning-vllm] [SCRIPT=train_grpo.py] [NAME=container-name] ./run.sh [script args...]
#   e.g. GPU=2 SCRIPT=train_grpo_paper.py ./run.sh --seed 1 --out /workspace/runs/paper_seed1
set -euo pipefail
cd "$(dirname "$0")"
GPU="${GPU:-3}"; SCRIPT="${SCRIPT:-train_grpo.py}"; IMAGE="${IMAGE:-deletion-tuning}"
docker build -q -t "$IMAGE" -f "$([ "$IMAGE" = deletion-tuning-vllm ] && echo Dockerfile.vllm || echo Dockerfile)" . >/dev/null
mkdir -p runs "${HF_CACHE:-$HOME/.cache/huggingface}"
docker run --rm ${NAME:+--name "$NAME"} --device "nvidia.com/gpu=${GPU}" --shm-size 16g \
  -v "${HF_CACHE:-$HOME/.cache/huggingface}:/hf_cache" \
  -v "$PWD/runs:/workspace/runs" \
  -v "$PWD:/workspace/src:ro" -w /workspace/src \
  "$IMAGE" python3 "$SCRIPT" "$@"
