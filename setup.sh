#!/usr/bin/env bash
# Run this fresh on every new A100/L4 grant — machines are ephemeral and wiped
# on window expiry, so there's no persisted environment between sessions.
set -euo pipefail

echo "== Installing pinned dependencies =="
pip install -r requirements.txt

echo "== Removing torchvision (import conflicts, unused here) =="
pip uninstall -y torchvision || true

echo "== Cloning Fast-dLLM =="
mkdir -p external
if [ ! -d "external/Fast-dLLM" ]; then
  git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git external/Fast-dLLM
else
  echo "external/Fast-dLLM already present, skipping clone"
fi

echo "== Done =="
echo "NOTE: do not pre-download the full LLaDA/Dream checkpoints or datasets"
echo "here as a separate step — model_loader.py and the eval scripts pull"
echo "only what's needed (streaming where possible). See README's shared-disk rule."
