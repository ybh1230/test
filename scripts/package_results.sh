#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-runs/rpc_clip_voc}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ZIP_NAME="rpc_clip_results_${STAMP}.zip"

if [ ! -d "${RUN_DIR}" ]; then
  echo "Run directory not found: ${RUN_DIR}"
  exit 1
fi

zip -r "${ZIP_NAME}" "${RUN_DIR}" experiments paper configs docs README.md
echo "Packed ${ZIP_NAME}"
