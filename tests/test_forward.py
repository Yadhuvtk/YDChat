from __future__ import annotations

import torch

from ydchat.config import ModelConfig
from ydchat.model.model import YDChatLM


def test_forward_shapes_and_loss() -> None:
    cfg = ModelConfig(
        vocab_size=128,
        seq_len=32,
        n_layers=2,
        d_model=64,
        n_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    model = YDChatLM(cfg)

    x = torch.randint(low=0, high=cfg.vocab_size, size=(2, 16), dtype=torch.long)
    y = torch.randint(low=0, high=cfg.vocab_size, size=(2, 16), dtype=torch.long)

    out = model(input_ids=x, labels=y)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_kv_cache_path() -> None:
    torch.manual_seed(1)
    cfg = ModelConfig(
        vocab_size=64,
        seq_len=32,
        n_layers=2,
        d_model=32,
        n_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    model = YDChatLM(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 5), dtype=torch.long)
    out = model(prompt, use_cache=True)
    assert out.past_key_values is not None
    assert len(out.past_key_values) == cfg.n_layers

    next_token = torch.randint(0, cfg.vocab_size, (1, 1), dtype=torch.long)
    out2 = model(next_token, past_key_values=out.past_key_values, use_cache=True)
    assert out2.logits.shape == (1, 1, cfg.vocab_size)
