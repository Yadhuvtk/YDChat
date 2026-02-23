from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    seq_len: int = 1024
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_std: float = 0.02
    tie_embeddings: bool = True


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    fused: bool = True


@dataclass
class TrainConfig:
    micro_batch_size: int = 1
    grad_accum_steps: int = 16
    max_steps: int = 200
    warmup_steps: int = 20
    log_interval: int = 10
    eval_interval: int = 50
    eval_batches: int = 20
    save_interval: int = 50
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    compile: bool = False


@dataclass
class DataConfig:
    train_path: str = "data/toy/train"
    val_path: str = "data/toy/val"
    tokenizer_path: str = "artifacts/tokenizer/ydchat.model"
    jsonl_key: str = "text"
    num_workers: int = 0


@dataclass
class RuntimeConfig:
    output_dir: str = "checkpoints/tiny"
    device: str = "cuda"
    seed: int = 1337


@dataclass
class SFTConfig:
    data_path: str = "data/sft/train.jsonl"
    val_fraction: float = 0.05
    max_samples: int | None = None


@dataclass
class YDChatConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> YDChatConfig:
    cfg = YDChatConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    _merge_dataclass(cfg, data)
    return cfg


def save_config(cfg: YDChatConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")


def config_to_dict(cfg: YDChatConfig) -> dict[str, Any]:
    return asdict(cfg)
