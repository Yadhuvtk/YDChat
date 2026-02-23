from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ydchat.tokenizer.tokenizer import YDTokenizer


def format_instruction_sample(instruction: str, input_text: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n"
        "### Input:\n"
        f"{input_text.strip()}\n"
        "### Response:\n"
    )


@dataclass
class SFTItem:
    input_ids: list[int]
    labels: list[int]


class SFTDataset(Dataset[SFTItem]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: YDTokenizer,
        seq_len: int,
        max_samples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples: list[SFTItem] = []

        src = Path(path)
        with src.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                instruction = str(row.get("instruction", "")).strip()
                input_text = str(row.get("input", "")).strip()
                output_text = str(row.get("output", "")).strip()
                if not instruction or not output_text:
                    continue

                prompt = format_instruction_sample(instruction, input_text)
                prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                output_ids = tokenizer.encode(output_text, add_bos=False, add_eos=True)

                input_ids = (prompt_ids + output_ids)[:seq_len]
                labels = ([-100] * len(prompt_ids) + output_ids)[:seq_len]

                if len(input_ids) < 2:
                    continue

                self.samples.append(SFTItem(input_ids=input_ids, labels=labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SFTItem:
        return self.samples[idx]


def sft_collate_fn(batch: list[SFTItem], pad_id: int, seq_len: int) -> dict[str, torch.Tensor]:
    input_ids = torch.full((len(batch), seq_len), fill_value=pad_id, dtype=torch.long)
    labels = torch.full((len(batch), seq_len), fill_value=-100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), seq_len), dtype=torch.long)

    for i, item in enumerate(batch):
        n = min(seq_len, len(item.input_ids))
        input_ids[i, :n] = torch.tensor(item.input_ids[:n], dtype=torch.long)
        labels[i, :n] = torch.tensor(item.labels[:n], dtype=torch.long)
        attention_mask[i, :n] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
