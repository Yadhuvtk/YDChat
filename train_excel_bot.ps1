param(
    [int]$MaxSteps = 800,
    [string]$Device = "cpu",
    [switch]$KeepAllCheckpoints,
    [switch]$InstallDeps,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv Python at $python. Create it first: python -m venv .venv"
}

function Run-Cmd {
    param(
        [string]$Cmd,
        [string[]]$CommandArgs
    )
    Write-Host (">> " + $Cmd + " " + ($CommandArgs -join " "))
    if ($DryRun) { return }
    & $Cmd @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

if ($InstallDeps) {
    Run-Cmd -Cmd $python -CommandArgs @("-m", "pip", "install", "-e", ".[dev]")
}

$config = "configs/excel_smoke.yaml"
$tokenizer = "artifacts/tokenizer/ydchat_excel.model"
$tokenizerInput = "data/processed/qa_pretrain_train.jsonl"
$sftData = "data/processed/qa_sft_train.jsonl"
$output = "checkpoints/excel-bot"
$singleModelPath = "checkpoints/YDCHAT.pt"

if (-not (Test-Path $config)) {
    throw "Missing config: $config"
}
if (-not (Test-Path $sftData)) {
    throw "Missing SFT data: $sftData"
}

if (-not (Test-Path $tokenizer)) {
    if (-not (Test-Path $tokenizerInput)) {
        throw "Tokenizer not found and tokenizer input missing: $tokenizerInput"
    }
    Run-Cmd -Cmd $python -CommandArgs @(
        "-m", "ydchat.tokenizer.train_tokenizer",
        "--input", $tokenizerInput,
        "--jsonl-key", "text",
        "--model-prefix", "artifacts/tokenizer/ydchat_excel",
        "--vocab-size", "512"
    )
}

Run-Cmd -Cmd $python -CommandArgs @(
    "-m", "ydchat.train.sft",
    "--config", $config,
    "--output", $output,
    "--max-steps", "$MaxSteps"
)

if (-not $DryRun) {
    $lastCheckpoint = Join-Path $output "last.pt"
    if (-not (Test-Path $lastCheckpoint)) {
        throw "Expected checkpoint not found: $lastCheckpoint"
    }

    $singleModelDir = Split-Path -Parent $singleModelPath
    if (-not (Test-Path $singleModelDir)) {
        New-Item -ItemType Directory -Path $singleModelDir -Force | Out-Null
    }

    Copy-Item -Path $lastCheckpoint -Destination $singleModelPath -Force

    if (-not $KeepAllCheckpoints) {
        Remove-Item -Path $output -Recurse -Force
    }
}

Write-Host ""
Write-Host "Training complete. Model file: $singleModelPath"
Write-Host "Start chat with:"
Write-Host ".\chat_excel_bot.ps1 -Device $Device"
