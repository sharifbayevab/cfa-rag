#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$HOME/cfa_spark"
HF_CACHE="$HOME/.cache/huggingface"
RUN_UID="$(id -u)"
RUN_GID="$(id -g)"

docker rm -f cfa-fresh-qa >/dev/null 2>&1 || true
docker run --rm --name cfa-fresh-qa --gpus all --ipc=host \
  --user "$RUN_UID:$RUN_GID" \
  -e HOME=/workspace/cfa_spark \
  -e HF_HOME=/workspace/hf-cache \
  -e HF_HUB_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$WORKDIR:/workspace/cfa_spark" \
  -v "$HF_CACHE:/workspace/hf-cache" \
  -w /workspace/cfa_spark \
  cfa-contextcite:latest \
  python run_fresh_qa.py \
    --num-masks 32 \
    --mask-batch-size 32 \
    --output results/fresh_qa.jsonl
