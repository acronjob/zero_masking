FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache \
    TOKENIZERS_PARALLELISM=false \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN pip install --no-cache-dir \
        "transformers>=4.57" accelerate datasets safetensors \
        flash-linear-attention

WORKDIR /workspace
COPY deletion_tuning.py train_grpo.py ./
