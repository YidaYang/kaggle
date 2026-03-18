"""
训练脚本 — 使用 HuggingFace Trainer 进行 QLoRA 微调。

完整流程：
  1. 加载配置 (默认值 + YAML 覆盖 + CLI 覆盖)
  2. 生成或加载训练数据
  3. Tokenization (apply_chat_template)
  4. 加载 Qwen3.5-0.8B + 4-bit 量化 + LoRA
  5. 使用 Trainer 训练
  6. 保存 adapter + tokenizer + 配置
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from arena_ranker.config import AppConfig, load_config
from arena_ranker.data import (
    build_dataset,
    compute_metrics,
    load_and_preprocess,
    split_train_valid,
)
from arena_ranker.hf import load_model, load_tokenizer

LOGGER = logging.getLogger("arena_ranker.train")


# ============================================================
#  CLI 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen3.5-0.8B for Arena preference prediction."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="train.csv / test.csv 所在目录")
    parser.add_argument("--model-name", type=str, default=None,
                        help="基座模型名称或本地路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="训练产物输出目录")
    parser.add_argument("--max-length", type=int, default=None,
                        help="tokenizer 最大长度")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="per_device_train_batch_size")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="学习率")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(use_lora=None, load_in_4bit=None)
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """将 CLI 参数覆盖到配置中。"""
    if args.model_name:
        config.model.model_name = args.model_name
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.max_length is not None:
        config.model.max_length = args.max_length
    if args.epochs is not None:
        config.training.num_train_epochs = args.epochs
    if args.batch_size is not None:
        config.training.per_device_train_batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.training.gradient_accumulation_steps = args.grad_accum_steps
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.cache_dir is not None:
        config.model.cache_dir = args.cache_dir
    if args.local_files_only:
        config.model.local_files_only = True
    if args.use_lora is not None:
        config.model.use_lora = args.use_lora
    if args.lora_r is not None:
        config.model.lora_r = args.lora_r
    if args.lora_alpha is not None:
        config.model.lora_alpha = args.lora_alpha
    if args.load_in_4bit is not None:
        config.model.load_in_4bit = args.load_in_4bit
    return config


# ============================================================
#  工具函数
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def format_seconds(seconds: float) -> str:
    total = max(int(seconds), 0)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    name = torch.cuda.get_device_name(device)
    mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"{name} ({mem:.1f} GB)"


# ============================================================
#  构建 TrainingArguments
# ============================================================

def build_training_args(config: AppConfig) -> TrainingArguments:
    """从 AppConfig 构建 HuggingFace TrainingArguments。"""
    tc = config.training
    return TrainingArguments(
        output_dir=tc.output_dir,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        num_train_epochs=tc.num_train_epochs,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        optim=tc.optim,
        fp16=tc.fp16,
        bf16=tc.bf16,
        gradient_checkpointing=tc.gradient_checkpointing,
        logging_steps=tc.logging_steps,
        eval_strategy=tc.eval_strategy,
        save_strategy=tc.save_strategy,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,
        seed=tc.seed,
        report_to=tc.report_to,
        dataloader_num_workers=tc.dataloader_num_workers,
        # PEFT + gradient checkpointing 兼容性
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if tc.gradient_checkpointing
        else None,
    )


# ============================================================
#  日志摘要
# ============================================================

def log_run_summary(
    config: AppConfig,
    device: torch.device,
    train_size: int,
    valid_size: int,
) -> None:
    mc = config.model
    tc = config.training
    LOGGER.info("=" * 60)
    LOGGER.info("训练启动")
    LOGGER.info("设备: %s", describe_device(device))
    LOGGER.info(
        "模型: %s | max_length=%s | 4bit=%s | LoRA=%s",
        mc.model_name, mc.max_length,
        "on" if mc.load_in_4bit else "off",
        "on" if mc.use_lora else "off",
    )
    if mc.use_lora:
        LOGGER.info(
            "LoRA 配置: r=%s, alpha=%s, dropout=%.3f, modules=%s, modules_to_save=%s",
            mc.lora_r, mc.lora_alpha, mc.lora_dropout,
            ",".join(mc.lora_target_modules),
            ",".join(mc.lora_modules_to_save),
        )
    LOGGER.info(
        "数据集: train=%s, valid=%s",
        train_size, valid_size,
    )
    LOGGER.info(
        "训练参数: lr=%.1e, epochs=%s, batch=%s, grad_accum=%s, scheduler=%s",
        tc.learning_rate, tc.num_train_epochs,
        tc.per_device_train_batch_size,
        tc.gradient_accumulation_steps,
        tc.lr_scheduler_type,
    )
    LOGGER.info("输出目录: %s", tc.output_dir)
    LOGGER.info("=" * 60)


# ============================================================
#  保存训练产物
# ============================================================

def save_artifacts(
    output_dir: Path,
    model,
    tokenizer,
    config: AppConfig,
    metrics: dict | None = None,
) -> None:
    """
    保存训练产物：
      - adapter 权重 (LoRA) + 分类头
      - tokenizer
      - 配置文件
      - 最佳指标 (如有)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 adapter (PEFT 模型) 或完整模型
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    config.save(output_dir / "config.yaml")

    if metrics:
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    LOGGER.info("训练产物已保存到: %s", output_dir)


# ============================================================
#  主函数
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(config.training.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(config.training.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started_at = time.perf_counter()

    # ---- 1. 加载数据 ----
    LOGGER.info("加载训练数据: %s", data_dir / config.data.train_path)
    train_df = load_and_preprocess(
        str(data_dir / config.data.train_path),
        max_chars=config.data.text_max_chars,
        is_train=True,
    )
    train_split, valid_split = split_train_valid(train_df, config.data)
    LOGGER.info("训练集: %s 条, 验证集: %s 条", len(train_split), len(valid_split))

    # ---- 2. 加载 tokenizer ----
    LOGGER.info("加载 tokenizer: %s", config.model.model_name)
    tokenizer = load_tokenizer(config.model)

    # ---- 3. Tokenize 数据集 ----
    LOGGER.info("Tokenizing 数据集 (max_length=%s)...", config.model.max_length)
    train_dataset = build_dataset(
        train_split, tokenizer, config.model.max_length, is_train=True,
    )
    valid_dataset = build_dataset(
        valid_split, tokenizer, config.model.max_length, is_train=True,
    )
    LOGGER.info(
        "Tokenization 完成: train=%s, valid=%s",
        len(train_dataset), len(valid_dataset),
    )

    # ---- 4. 加载模型 ----
    LOGGER.info("加载模型 (QLoRA): %s", config.model.model_name)
    model = load_model(config.model, tokenizer=tokenizer)

    log_run_summary(config, device, len(train_dataset), len(valid_dataset))

    # ---- 5. 构建 Trainer ----
    training_args = build_training_args(config)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---- 6. 训练 ----
    LOGGER.info("开始训练...")
    train_result = trainer.train()

    # ---- 7. 评估 ----
    LOGGER.info("训练完成，运行最终评估...")
    eval_result = trainer.evaluate()
    LOGGER.info(
        "最终评估: log_loss=%.4f, accuracy=%.4f",
        eval_result.get("eval_log_loss", float("nan")),
        eval_result.get("eval_accuracy", float("nan")),
    )

    # ---- 8. 保存 ----
    metrics = {
        "train_loss": train_result.training_loss,
        "eval_log_loss": eval_result.get("eval_log_loss"),
        "eval_accuracy": eval_result.get("eval_accuracy"),
    }
    save_artifacts(output_dir, model, tokenizer, config, metrics)

    elapsed = format_seconds(time.perf_counter() - started_at)
    LOGGER.info("全部完成，总耗时: %s", elapsed)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
