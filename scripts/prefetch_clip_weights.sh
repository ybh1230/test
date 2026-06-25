#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

python - <<'PY'
from huggingface_hub import hf_hub_download

repo = "timm/vit_base_patch16_clip_224.openai"
for filename in ["open_clip_model.safetensors", "open_clip_pytorch_model.bin"]:
    try:
        path = hf_hub_download(repo_id=repo, filename=filename)
        print("downloaded:", path)
        break
    except Exception as exc:
        print("failed:", filename, exc)
else:
    raise SystemExit("Could not download any CLIP checkpoint file.")
PY

