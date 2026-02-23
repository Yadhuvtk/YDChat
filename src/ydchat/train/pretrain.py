from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from ydchat.config import YDChatConfig, load_config, save_config
from ydchat.data.packer import PackedTokenDataset
from ydchat.data.stream_dataset import TokenStreamDataset, write_toy_corpus
from ydchat.model.model import YDChatLM
from ydchat.tokenizer.tokenizer import load_tokenizer
from ydchat.train.checkpoints import find_last_checkpoint, load_checkpoint, save_checkpoint
from ydchat.train.log import TrainLogger
from ydchat.train.optim import create_adamw
from ydchat.train.sched import create_scheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def build_dataloader(cfg: YDChatConfig, tokenizer_path: str, split: str) -> DataLoader:
    tokenizer = load_tokenizer(tokenizer_path)
    path = cfg.data.train_path if split == "train" else cfg.data.val_path
    token_stream = TokenStreamDataset(path=path, tokenizer=tokenizer, jsonl_key=cfg.data.jsonl_key)
    packed = PackedTokenDataset(token_stream=token_stream, seq_len=cfg.model.seq_len)
    return DataLoader(
        packed,
        batch_size=cfg.train.micro_batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(
    model: YDChatLM,
    val_iter: Iterator[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    steps: int,
    precision: str,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []

    for _ in range(steps):
        batch = next(val_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast_context(device, precision):
            out = model(input_ids=input_ids, labels=labels)
        if out.loss is None:
            continue
        losses.append(float(out.loss.detach().cpu()))

    model.train()
    if not losses:
        return float("inf"), float("inf")

    avg_loss = float(sum(losses) / len(losses))
    ppl = float(math.exp(min(20.0, avg_loss)))
    return avg_loss, ppl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YDChat from scratch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", help="Override tokenizer path")
    parser.add_argument("--train-data", help="Override training data path (.txt folder or .jsonl)")
    parser.add_argument("--val-data", help="Override validation data path (.txt folder or .jsonl)")
    parser.add_argument("--output", help="Override output checkpoint directory")
    parser.add_argument("--max-steps", type=int, help="Override train.max_steps")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/last.pt")
    parser.add_argument("--resume-from", help="Explicit checkpoint to resume from")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--make-toy-data", action="store_true", help="Generate toy corpus before training")
    parser.add_argument("--toy-train-samples", type=int, default=5000)
    parser.add_argument("--toy-val-samples", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.tokenizer:
        cfg.data.tokenizer_path = args.tokenizer
    if args.train_data:
        cfg.data.train_path = args.train_data
    if args.val_data:
        cfg.data.val_path = args.val_data
    if args.output:
        cfg.runtime.output_dir = args.output
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps

    if args.make_toy_data:
        write_toy_corpus(
            train_dir=cfg.data.train_path,
            val_dir=cfg.data.val_path,
            train_samples=args.toy_train_samples,
            val_samples=args.toy_val_samples,
        )

    out_dir = Path(cfg.runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")

    set_seed(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    if tokenizer.vocab_size != cfg.model.vocab_size:
        print(
            f"Warning: tokenizer vocab ({tokenizer.vocab_size}) != config vocab ({cfg.model.vocab_size}). "
            "Model uses config vocab size."
        )

    model = YDChatLM(cfg.model).to(device)
    if cfg.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_loader = build_dataloader(cfg, cfg.data.tokenizer_path, split="train")
    val_loader = build_dataloader(cfg, cfg.data.tokenizer_path, split="val")
    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

    optimizer = create_adamw(model.parameters(), cfg.optim, device=device)
    scheduler = create_scheduler(
        optimizer,
        total_steps=cfg.train.max_steps,
        warmup_steps=cfg.train.warmup_steps,
    )

    scaler = None
    if device.type == "cuda" and cfg.train.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    start_step = 0
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume:
        resume_path = find_last_checkpoint(out_dir)

    if resume_path is not None and resume_path.exists():
        start_step = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        print(f"Resumed from {resume_path} at step {start_step}")

    logger = TrainLogger(enable_tensorboard=args.tensorboard, log_dir=str(out_dir / "tb"))
    model.train()

    for step in range(start_step, cfg.train.max_steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []

        for _ in range(cfg.train.grad_accum_steps):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast_context(device, cfg.train.precision):
                out = model(input_ids=input_ids, labels=labels)
                if out.loss is None:
                    raise RuntimeError("Model returned no loss during training")
                loss = out.loss / cfg.train.grad_accum_steps

            micro_losses.append(float(out.loss.detach().cpu()))

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        global_step = step + 1

        if global_step % cfg.train.log_interval == 0:
            avg_train_loss = float(sum(micro_losses) / max(1, len(micro_losses)))
            logger.log(
                step=global_step,
                metrics={
                    "train_loss": avg_train_loss,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
            )

        if global_step % cfg.train.eval_interval == 0:
            val_loss, val_ppl = evaluate(
                model,
                val_iter,
                device=device,
                steps=cfg.train.eval_batches,
                precision=cfg.train.precision,
            )
            logger.log(step=global_step, metrics={"val_loss": val_loss, "val_ppl": val_ppl})

        if global_step % cfg.train.save_interval == 0 or global_step == cfg.train.max_steps:
            ckpt = save_checkpoint(
                out_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=global_step,
                cfg=cfg,
            )
            print(f"Saved checkpoint: {ckpt}")

    logger.close()


if __name__ == "__main__":
    main()
