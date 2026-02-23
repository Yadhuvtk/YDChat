# YDChat

YDChat is an original decoder-only language model project built from scratch in PyTorch.

## What "from scratch" means in this repository

- All model weights are randomly initialized.
- No pretrained checkpoints are imported.
- Transformer layers, RoPE, attention, SwiGLU, and KV cache are implemented in this codebase.
- You train your own tokenizer and checkpoints on your own licensed data.

## Features

- Original PyTorch implementation of a causal decoder transformer
- RMSNorm + SwiGLU + rotary position embeddings (RoPE)
- KV cache for fast autoregressive decoding
- Pretraining loop (mixed precision, grad accumulation, clipping, checkpointing, resume)
- SFT loop with instruction format and response-only loss masking
- Text generation CLI with greedy, top-k, top-p, temperature, repetition penalty
- FastAPI inference server (`POST /generate`)
- Interactive chat CLI
- Toy dataset generator for offline smoke tests

## Repository layout

- `src/ydchat/config.py`: dataclass config system + YAML loading
- `src/ydchat/tokenizer/`: SentencePiece tokenizer training and runtime wrapper
- `src/ydchat/model/`: model implementation (embeddings, RoPE, attention, blocks, LM)
- `src/ydchat/data/`: text streaming, sequence packing, SFT dataset
- `src/ydchat/train/`: pretraining + SFT loops, optimizer, scheduler, checkpointing, logging
- `src/ydchat/infer/`: generation, API server, chat CLI
- `configs/`: tiny/small/template configs
- `tests/`: forward/tokenizer/generation tests

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## CLI commands

After install, these commands are available:

- `ydchat-train`
- `ydchat-sft`
- `ydchat-generate`
- `ydchat-serve`
- `ydchat-chat`

## 1) Generate toy data (no downloads)

```bash
python -m ydchat.data.stream_dataset --train-dir data/toy/train --val-dir data/toy/val --train-samples 5000 --val-samples 512
```

## 2) Train tokenizer from your corpus

From txt folder:

```bash
python -m ydchat.tokenizer.train_tokenizer --input data/toy/train --model-prefix artifacts/tokenizer/ydchat --vocab-size 32000
```

From jsonl (`{"text": "..."}` lines):

```bash
python -m ydchat.tokenizer.train_tokenizer --input data/corpus.jsonl --jsonl-key text --model-prefix artifacts/tokenizer/ydchat --vocab-size 32000
```

## 3) Pretrain a tiny model smoke test

```bash
ydchat-train \
  --config configs/tiny.yaml \
  --tokenizer artifacts/tokenizer/ydchat.model \
  --train-data data/toy/train \
  --val-data data/toy/val \
  --output checkpoints/tiny \
  --max-steps 200
```

Generate sample text:

```bash
ydchat-generate \
  --config configs/tiny.yaml \
  --checkpoint checkpoints/tiny/last.pt \
  --tokenizer artifacts/tokenizer/ydchat.model \
  --prompt "### Instruction:\nWrite one paragraph about safe AI deployment.\n### Response:\n" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-p 0.95
```

## 4) SFT (instruction tuning)

Input format (`.jsonl`):

```json
{"instruction":"...","input":"...","output":"..."}
```

Prompt template used:

```text
### Instruction:
...
### Input:
...
### Response:
```

Train:

```bash
ydchat-sft \
  --config configs/tiny.yaml \
  --tokenizer artifacts/tokenizer/ydchat.model \
  --sft-data data/sft/train.jsonl \
  --output checkpoints/sft \
  --init-checkpoint checkpoints/tiny/last.pt \
  --max-steps 300
```

Loss is masked so only response tokens contribute.

## 5) Serve model over HTTP

```bash
ydchat-serve \
  --config configs/tiny.yaml \
  --checkpoint checkpoints/tiny/last.pt \
  --tokenizer artifacts/tokenizer/ydchat.model \
  --host 0.0.0.0 \
  --port 8000
```

Request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello from YDChat","max_new_tokens":64,"temperature":0.8,"top_p":0.95}'
```

## 6) Interactive chat CLI

```bash
ydchat-chat \
  --config configs/tiny.yaml \
  --checkpoint checkpoints/tiny/last.pt \
  --tokenizer artifacts/tokenizer/ydchat.model
```

Type `/exit` to quit.

## Scaling guide

Primary scaling knobs in config:

- `model.n_layers`
- `model.d_model`
- `model.n_heads`
- `model.seq_len`
- `model.vocab_size`
- `train.micro_batch_size`
- `train.grad_accum_steps`

Practical rule:

- Bigger `d_model`/`n_layers` => more quality potential and much higher memory/compute.
- Bigger `seq_len` => higher context but quadratic attention cost.
- Bigger `batch` (effective batch = micro_batch * grad_accum) => smoother optimization but more memory.

## Commercialization checklist

See `COMMERCIALIZATION.md` for policy-level guidance. Core points:

- Only train on data you are licensed to use.
- Do not include copyrighted/private corpora without permission.
- Keep provenance logs (what data, when, rights status).
- Only distribute checkpoints you trained yourself.
- Add product safety layers (moderation, logging, abuse controls) before production deployment.

## Determinism and initialization

- Seed controls are applied to Python, NumPy, and PyTorch.
- Linear/embedding weights are initialized with normal distribution (`std = model.init_std`), biases set to zero.
- RMSNorm scale starts at ones.

## Quick scripts

- `src/ydchat/scripts/quickstart_tiny.sh`
- `src/ydchat/scripts/quickstart_tiny.ps1`
- `src/ydchat/scripts/serve.sh`

## License

This repository ships with `Proprietary - All Rights Reserved` by default (`LICENSE`).
If you want open distribution later, replace `LICENSE` explicitly and review dependency licenses and trained-data rights.
