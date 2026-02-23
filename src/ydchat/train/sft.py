from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from ydchat.config import load_config, save_config
from ydchat.data.sft_dataset import SFTDataset, sft_collate_fn
from ydchat.model.model import YDChatLM
from ydchat.tokenizer.tokenizer import load_tokenizer
from ydchat.train.checkpoints import load_checkpoint, save_checkpoint
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


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def evaluate(
    model: YDChatLM,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    for batch in loader:
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
    return avg_loss, float(math.exp(min(20.0, avg_loss)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for YDChat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", help="Override tokenizer model path")
    parser.add_argument("--sft-data", help="Override SFT jsonl path")
    parser.add_argument("--output", help="Override output checkpoint dir")
    parser.add_argument("--init-checkpoint", help="Optional checkpoint to initialize model weights")
    parser.add_argument("--max-steps", type=int, help="Override training steps")
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.tokenizer:
        cfg.data.tokenizer_path = args.tokenizer
    if args.sft_data:
        cfg.sft.data_path = args.sft_data
    if args.output:
        cfg.runtime.output_dir = args.output
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps

    out_dir = Path(cfg.runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config_sft.yaml")

    set_seed(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)

    tokenizer = load_tokenizer(cfg.data.tokenizer_path)
    dataset = SFTDataset(
        path=cfg.sft.data_path,
        tokenizer=tokenizer,
        seq_len=cfg.model.seq_len,
        max_samples=cfg.sft.max_samples,
    )
    if len(dataset) < 2:
        raise ValueError("Need at least 2 SFT samples to train/evaluate")

    n_val = max(1, int(len(dataset) * cfg.sft.val_fraction))
    n_train = max(1, len(dataset) - n_val)
    train_ds, val_ds = random_split(
        dataset,
        [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(cfg.runtime.seed),
    )

    collate = lambda b: sft_collate_fn(b, pad_id=tokenizer.pad_id, seq_len=cfg.model.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.micro_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.micro_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate,
    )

    model = YDChatLM(cfg.model).to(device)
    if args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, model=model, map_location=device)

    optimizer = create_adamw(model.parameters(), cfg.optim, device=device)
    scheduler = create_scheduler(
        optimizer,
        total_steps=cfg.train.max_steps,
        warmup_steps=cfg.train.warmup_steps,
    )

    scaler = None
    if device.type == "cuda" and cfg.train.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    logger = TrainLogger(enable_tensorboard=args.tensorboard, log_dir=str(out_dir / "tb"))

    train_iter = iter(train_loader)
    model.train()

    for step in range(cfg.train.max_steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []

        for _ in range(cfg.train.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast_context(device, cfg.train.precision):
                out = model(input_ids=input_ids, labels=labels)
                if out.loss is None:
                    raise RuntimeError("Model returned no loss during SFT")
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
            logger.log(
                step=global_step,
                metrics={
                    "train_loss": float(sum(micro_losses) / len(micro_losses)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
            )

        if global_step % cfg.train.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device=device, precision=cfg.train.precision)
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
