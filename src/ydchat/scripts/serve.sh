#!/usr/bin/env bash
set -euo pipefail

ydchat-serve \
  --config configs/tiny.yaml \
  --checkpoint checkpoints/tiny/last.pt \
  --tokenizer artifacts/tokenizer/ydchat.model \
  --host 0.0.0.0 \
  --port 8000
