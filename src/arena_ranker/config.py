"""
配置模块 — 定义训练和推理所需的全部默认参数。

本模块采用 dataclass 组织配置，支持 YAML 持久化。
主要分为三部分：
  - DataConfig:  数据路径、文本截断、验证集比例
  - ModelConfig: 基座模型、量化、LoRA 参数
  - TrainingConfig: 学习率、batch size、epoch 等 Trainer 参数
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ============================================================
#  标签相关常量
# ============================================================
LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
LABEL_TO_ID = {"winner_model_a": 0, "winner_model_b": 1, "winner_tie": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_LABELS = 3

# ============================================================
#  对话模板常量
# ============================================================
SYSTEM_PROMPT = "你是一个公正的评委，请全面评估两个回答，并预测人类最偏好的选项。"

USER_TEMPLATE = (
    "以下是用户的提问：\n{prompt}\n\n"
    "回答A：\n{response_a}\n\n"
    "回答B：\n{response_b}\n\n"
    "综合评估，你认为哪个回答更好？"
)


# ============================================================
#  DataConfig
# ============================================================
@dataclass(slots=True)
class DataConfig:
    train_path: str = "train.csv"
    test_path: str = "test.csv"
    text_max_chars: int = 6000
    validation_size: float = 0.1
    random_state: int = 42


# ============================================================
#  ModelConfig
# ============================================================
@dataclass(slots=True)
class ModelConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B"
    cache_dir: str | None = None
    local_files_only: bool = False
    max_length: int = 1024

    # --- 4-bit 量化 (QLoRA) ---
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # --- LoRA ---
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_modules_to_save: list[str] = field(
        default_factory=lambda: ["score"]
    )


# ============================================================
#  TrainingConfig
# ============================================================
@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "artifacts/default"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "log_loss"
    greater_is_better: bool = False
    seed: int = 42
    report_to: str = "none"
    dataloader_num_workers: int = 0


# ============================================================
#  AppConfig (顶层聚合)
# ============================================================
@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False),
            encoding="utf-8",
        )


def load_config(path: str | Path | None = None) -> AppConfig:
    """从 YAML 文件加载配置；path=None 时返回全部默认值。"""
    if path is None:
        return AppConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
