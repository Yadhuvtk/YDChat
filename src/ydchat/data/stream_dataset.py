from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import IterableDataset

from ydchat.tokenizer.tokenizer import YDTokenizer


def iter_text_records(path: str | Path, jsonl_key: str = "text") -> Iterator[str]:
    src = Path(path)
    if src.is_dir():
        for txt_file in sorted(src.rglob("*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                yield text
        return

    if src.suffix.lower() == ".jsonl":
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get(jsonl_key)
                if isinstance(text, str) and text.strip():
                    yield text.strip()
        return

    if src.suffix.lower() == ".txt":
        text = src.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            yield text
        return

    raise ValueError(f"Unsupported data source: {src}")


class TokenStreamDataset(IterableDataset):
    def __init__(self, path: str | Path, tokenizer: YDTokenizer, jsonl_key: str = "text") -> None:
        super().__init__()
        self.path = str(path)
        self.tokenizer = tokenizer
        self.jsonl_key = jsonl_key

    def __iter__(self) -> Iterator[int]:
        for text in iter_text_records(self.path, jsonl_key=self.jsonl_key):
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            for tid in ids:
                yield tid


TOY_TOPICS = [
    "distributed systems",
    "safe AI deployment",
    "deterministic testing",
    "observability",
    "privacy engineering",
    "supply chain risk",
    "product analytics",
    "ML evaluation",
]


def build_toy_text(idx: int) -> str:
    random.seed(idx)
    topic = random.choice(TOY_TOPICS)
    quality = random.choice(["reliable", "secure", "maintainable", "scalable"])
    sentence_1 = f"Document {idx}: A {quality} product benefits from clear interfaces and strict regression checks."
    sentence_2 = f"Teams building {topic} systems should track experiments, data versions, and model behavior over time."
    sentence_3 = "The right workflow balances speed with correctness and keeps a complete decision log for audits."
    return " ".join([sentence_1, sentence_2, sentence_3])


def write_toy_corpus(train_dir: str | Path, val_dir: str | Path, train_samples: int, val_samples: int) -> None:
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    for i in range(train_samples):
        (train_path / f"sample_{i:05d}.txt").write_text(build_toy_text(i), encoding="utf-8")

    base = train_samples
    for i in range(val_samples):
        (val_path / f"sample_{i:05d}.txt").write_text(build_toy_text(base + i), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny local text corpus for YDChat smoke tests")
    parser.add_argument("--train-dir", default="data/toy/train")
    parser.add_argument("--val-dir", default="data/toy/val")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_toy_corpus(args.train_dir, args.val_dir, args.train_samples, args.val_samples)
    print(f"Toy data written to {args.train_dir} and {args.val_dir}")


if __name__ == "__main__":
    main()
