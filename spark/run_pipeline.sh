#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="nvcr.io/nvidia/pytorch:25.09-py3"
WORKDIR="$HOME/cfa_spark"
HF_CACHE="$HOME/.cache/huggingface"
IMAGE="cfa-contextcite:latest"
RUN_UID="$(id -u)"
RUN_GID="$(id -g)"

until docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; do
  echo "waiting for base image"
  sleep 10
done

docker build -t "$IMAGE" "$WORKDIR"

run_container() {
  docker run --rm --gpus all --ipc=host \
    --user "$RUN_UID:$RUN_GID" \
    -e HOME=/workspace/cfa_spark \
    -e HF_HOME=/workspace/hf-cache \
    -e TOKENIZERS_PARALLELISM=false \
    -v "$WORKDIR:/workspace/cfa_spark" \
    -v "$HF_CACHE:/workspace/hf-cache" \
    -w /workspace/cfa_spark \
    "$IMAGE" "$@"
}

echo "running three-question pilot"
run_container python run_fresh_qa.py \
  --limit 3 \
  --num-masks 8 \
  --mask-batch-size 4 \
  --output results/pilot.jsonl

echo "pilot passed; running full 450-question experiment"
run_container python run_fresh_qa.py \
  --num-masks 32 \
  --mask-batch-size 16 \
  --output results/fresh_qa.jsonl

echo "pipeline complete"
