param(
    [string]$Device = "cpu",
    [string]$Checkpoint = "checkpoints/YDCHAT.pt",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv Python at $python. Create it first: python -m venv .venv"
}

$config = "configs/excel_smoke.yaml"
$tokenizer = "artifacts/tokenizer/ydchat_excel.model"

if (-not (Test-Path $config)) {
    throw "Missing config: $config"
}
if (-not (Test-Path $tokenizer)) {
    throw "Missing tokenizer: $tokenizer"
}
if (-not (Test-Path $Checkpoint)) {
    throw "Missing checkpoint: $Checkpoint. Run .\train_excel_bot.ps1 first."
}

$env:PYTHONUTF8 = "1"
$args = @(
    "-m", "ydchat.infer.chat_cli",
    "--config", $config,
    "--checkpoint", $Checkpoint,
    "--tokenizer", $tokenizer,
    "--device", $Device,
    "--mode", "instruction",
    "--temperature", "0.0",
    "--top-k", "0",
    "--top-p", "1.0",
    "--max-new-tokens", "120"
)

Write-Host (">> " + $python + " " + ($args -join " "))
if (-not $DryRun) {
    & $python @args
}
