#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train.py \
  --config configs/voc_rpc_clip.yaml \
  --output runs/rpc_clip_voc

python evaluate.py \
  --config configs/voc_rpc_clip.yaml \
  --checkpoint runs/rpc_clip_voc/best.pt \
  --output runs/rpc_clip_voc

python visualize.py \
  --config configs/voc_rpc_clip.yaml \
  --checkpoint runs/rpc_clip_voc/best.pt \
  --output runs/rpc_clip_voc/visuals \
  --limit 16

