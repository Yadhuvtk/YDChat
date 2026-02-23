from __future__ import annotations

import torch

from ydchat.infer.generate import apply_repetition_penalty, sample_next_token, top_k_filter, top_p_filter


def test_repetition_penalty_changes_logits() -> None:
    logits = torch.tensor([[1.0, -0.5, 0.25]])
    out = apply_repetition_penalty(logits, generated_ids=[0, 2], penalty=1.2)
    assert out[0, 0] < logits[0, 0]
    assert out[0, 2] < logits[0, 2]


def test_top_k_filter() -> None:
    logits = torch.tensor([[1.0, 0.9, 0.1, -0.5]])
    filtered = top_k_filter(logits, top_k=2)
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_top_p_filter_keeps_mass() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    filtered = top_p_filter(logits, top_p=0.8)
    assert torch.isfinite(filtered[0, 0])
    assert torch.isfinite(filtered[0, 1])


def test_sample_next_token_greedy() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.2]])
    token = sample_next_token(
        logits,
        generated_ids=[],
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    assert token == 1
