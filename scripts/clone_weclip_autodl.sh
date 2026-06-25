#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp

git config --global http.version HTTP/1.1
rm -rf WeCLIP
git clone --depth 1 https://github.com/zbf1991/WeCLIP.git WeCLIP

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf-cache}"

echo "WeCLIP cloned to /root/autodl-tmp/WeCLIP"
echo "Next: read /root/autodl-tmp/WeCLIP/WeCLIP/README.md and follow its dataset/checkpoint commands."

