#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/root/autodl-tmp/datasets}"
mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

if [ ! -d "VOCdevkit/VOC2012" ]; then
  wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar -xf VOCtrainval_11-May-2012.tar
fi

echo "VOC is ready at ${DATA_ROOT}/VOCdevkit/VOC2012"

