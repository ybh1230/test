#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

bash scripts/prefetch_clip_weights.sh

BASE_CONFIG="configs/voc_rpc_clip.yaml"

run_variant() {
  local name="$1"
  shift
  python train.py --config "${BASE_CONFIG}" --output "runs/ablations/${name}" --set "$@"
  python evaluate.py --config "runs/ablations/${name}/config.yaml" --checkpoint "runs/ablations/${name}/best.pt" --output "runs/ablations/${name}"
}

run_variant text_only \
  rpc.text_weight=1.0 rpc.prototype_weight=0.0 rpc.decoder_weight=0.0 \
  rpc.reliability_mode=confidence train.lambda_boundary=0.0

run_variant text_proto \
  rpc.text_weight=0.60 rpc.prototype_weight=0.40 rpc.decoder_weight=0.0 \
  rpc.reliability_mode=confidence train.lambda_boundary=0.0

run_variant text_proto_reliable \
  rpc.text_weight=0.50 rpc.prototype_weight=0.35 rpc.decoder_weight=0.15 \
  rpc.reliability_mode=entropy train.lambda_boundary=0.0

run_variant full_rpc_clip \
  rpc.text_weight=0.45 rpc.prototype_weight=0.35 rpc.decoder_weight=0.20 \
  rpc.reliability_mode=entropy train.lambda_boundary=0.04
