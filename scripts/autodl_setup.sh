#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -r requirements.txt
bash scripts/prefetch_clip_weights.sh

echo "Environment ready. If open_clip weight download is slow, rerun the training command once."
