from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    train_path: str = "train.csv"
    test_path: str = "test.csv"
    text_max_chars: int = 4000
    validation_size: float = 0.1
    random_state: int = 42


@dataclass(slots=True)
class ModelConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    cache_dir: str | None = None
    local_files_only: bool = False
    max_length: int = 512
    use_segment_budget: bool = True
    prompt_budget: int = 192
    response_budget: int = 384
    response_head_tokens: int = 256
    response_tail_tokens: int = 128
    dropout: float = 0.1
    freeze_encoder: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_task_type: str = "feature_extraction"
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "artifacts/default"
    learning_rate: float = 2e-5
    classifier_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 1
    epochs: int = 1
    grad_accum_steps: int = 8
    warmup_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    amp: bool = True
    gradient_checkpointing: bool = True
    log_every: int = 50


@dataclass(slots=True)
class SwanlabConfig:
    enabled: bool = False
    project: str = "arena-ranker"
    experiment_name: str | None = None
    workspace: str | None = None
    mode: str | None = None


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    swanlab: SwanlabConfig = field(default_factory=SwanlabConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        swanlab=SwanlabConfig(**raw.get("swanlab", {})),
    )
