$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Path "data/toy/train" -Force | Out-Null
New-Item -ItemType Directory -Path "data/toy/val" -Force | Out-Null
New-Item -ItemType Directory -Path "artifacts/tokenizer" -Force | Out-Null
New-Item -ItemType Directory -Path "checkpoints/tiny" -Force | Out-Null

python -m ydchat.data.stream_dataset `
  --train-dir data/toy/train `
  --val-dir data/toy/val `
  --train-samples 5000 `
  --val-samples 512

python -m ydchat.tokenizer.train_tokenizer `
  --input data/toy/train `
  --model-prefix artifacts/tokenizer/ydchat `
  --vocab-size 32000

ydchat-train `
  --config configs/tiny.yaml `
  --tokenizer artifacts/tokenizer/ydchat.model `
  --train-data data/toy/train `
  --val-data data/toy/val `
  --output checkpoints/tiny `
  --max-steps 200

ydchat-generate `
  --config configs/tiny.yaml `
  --checkpoint checkpoints/tiny/last.pt `
  --tokenizer artifacts/tokenizer/ydchat.model `
  --prompt "### Instruction:`nGive one practical tip for production ML monitoring.`n### Response:`n" `
  --max-new-tokens 100 `
  --temperature 0.8 `
  --top-p 0.95
