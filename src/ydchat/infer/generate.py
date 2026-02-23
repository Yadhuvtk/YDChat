from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ydchat.config import load_config
from ydchat.model.model import YDChatLM
from ydchat.tokenizer.tokenizer import YDTokenizer, load_tokenizer
from ydchat.train.checkpoints import load_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not generated_ids:
        return logits

    adjusted = logits.clone()
    for tid in set(generated_ids):
        value = adjusted[..., tid]
        adjusted[..., tid] = torch.where(value < 0, value * penalty, value / penalty)
    return adjusted


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    cutoff = values[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, -torch.inf), logits)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)

    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_logits = torch.where(remove, torch.full_like(sorted_logits, -torch.inf), sorted_logits)
    filtered = torch.full_like(logits, -torch.inf)
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    generated_ids: list[int],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> int:
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    logits = top_k_filter(logits, top_k=top_k)
    logits = top_p_filter(logits, top_p=top_p)

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return int(next_id.item())


def load_model_and_tokenizer(
    config_path: str | Path,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: torch.device,
) -> tuple[YDChatLM, YDTokenizer]:
    cfg = load_config(config_path)
    model = YDChatLM(cfg.model).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer


@torch.no_grad()
def generate_text(
    model: YDChatLM,
    tokenizer: YDTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    device: torch.device,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if not prompt_ids:
        raise ValueError("Prompt encodes to empty token sequence")

    generated_ids = list(prompt_ids)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out = model(input_ids=input_ids, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values

    for _ in range(max_new_tokens):
        next_id = sample_next_token(
            logits,
            generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_id:
            break

        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        out = model(input_ids=next_input, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

    return tokenizer.decode(generated_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with YDChat")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    model, tokenizer = load_model_and_tokenizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=device,
    )

    text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
