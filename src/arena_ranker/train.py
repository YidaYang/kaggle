"""
训练脚本 — 使用 HuggingFace Trainer 进行 QLoRA 微调。

完整流程：
  1. 加载配置 (默认值 + YAML 覆盖 + CLI 覆盖)
  2. 生成或加载训练数据
  3. Tokenization (apply_chat_template)
  4. 加载 Qwen3-0.6B + 4-bit 量化 + LoRA
  5. 使用 Trainer 训练
  6. 保存 adapter + tokenizer + 配置
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
        description="QLoRA fine-tune Qwen3-0.6B for Arena preference prediction."
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
    parser.add_argument("--warmup-ratio", type=float, default=None,
                        help="warmup 占总训练步数的比例")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="warmup 绝对步数；设置后覆盖 warmup_ratio")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--fp16", dest="fp16", action="store_true",
                        help="启用 FP16 混合精度")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="禁用 FP16 混合精度")
    parser.add_argument("--bf16", dest="bf16", action="store_true",
                        help="启用 BF16 混合精度")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="禁用 BF16 混合精度")
    parser.add_argument("--ddp-find-unused-parameters",
                        dest="ddp_find_unused_parameters",
                        action="store_true",
                        help="DDP 下启用 unused parameter 检测")
    parser.add_argument("--no-ddp-find-unused-parameters",
                        dest="ddp_find_unused_parameters",
                        action="store_false",
                        help="DDP 下禁用 unused parameter 检测")
    parser.set_defaults(
        use_lora=None,
        load_in_4bit=None,
        fp16=None,
        bf16=None,
        ddp_find_unused_parameters=None,
    )
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
    if args.warmup_ratio is not None:
        config.training.warmup_ratio = args.warmup_ratio
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
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
    if args.fp16 is not None:
        config.training.fp16 = args.fp16
    if args.bf16 is not None:
        config.training.bf16 = args.bf16
    if args.ddp_find_unused_parameters is not None:
        config.training.ddp_find_unused_parameters = args.ddp_find_unused_parameters
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


def get_local_rank() -> int:
    raw = os.environ.get("LOCAL_RANK")
    return int(raw) if raw is not None else -1


def get_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE")
    return int(raw) if raw is not None else 1


def get_visible_gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def get_runtime_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = get_local_rank()
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cuda")


def validate_parallelism_config(config: AppConfig) -> None:
    """提前拦截已知不兼容的并行配置，避免 Trainer 落到 DataParallel。"""
    visible_gpu_count = get_visible_gpu_count()
    world_size = get_world_size()
    if config.model.load_in_4bit and world_size > 1:
        raise RuntimeError(
            "检测到多进程多卡训练且启用了 bitsandbytes 4-bit / QLoRA。"
            "当前 Kaggle 环境下，这个组合容易在模型加载阶段卡住。"
            "请在双卡训练时关闭 4-bit（传入 --no-4bit），"
            "改用 LoRA + FP16 + DDP；若必须使用 4-bit，请改回单卡训练。"
        )
    if config.model.load_in_4bit and world_size == 1 and visible_gpu_count > 1:
        raise RuntimeError(
            "检测到单进程训练，但当前可见 GPU 数量大于 1。"
            "HuggingFace Trainer 会回退到 torch.nn.DataParallel，"
            "而 bitsandbytes 4-bit / QLoRA 与 DataParallel 不兼容。"
            "请改用 torch.distributed.run --nproc_per_node=<GPU数> 启动多进程训练，"
            "或在单进程训练前设置 CUDA_VISIBLE_DEVICES=0 只暴露一张 GPU。"
        )


# ============================================================
#  构建 TrainingArguments
# ============================================================

def build_training_args(config: AppConfig) -> TrainingArguments:
    """从 AppConfig 构建 HuggingFace TrainingArguments。"""
    tc = config.training
    has_cuda = torch.cuda.is_available()
    world_size = get_world_size()

    # CPU 模式下自动调整不兼容的参数
    optim = tc.optim
    fp16 = tc.fp16
    bf16 = tc.bf16
    if fp16 and bf16:
        raise ValueError("fp16 和 bf16 不能同时开启，请二选一。")
    if not has_cuda:
        if optim.startswith("paged_"):
            optim = "adamw_torch"
            LOGGER.warning("CPU 模式: 优化器从 %s 回退为 adamw_torch", tc.optim)
        fp16 = False
        bf16 = False
        LOGGER.warning("CPU 模式: 已禁用 fp16/bf16 混合精度")
    elif bf16 and not torch.cuda.is_bf16_supported():
        bf16 = False
        LOGGER.warning("当前 CUDA 设备不支持 bf16，已自动关闭 bf16")

    return TrainingArguments(
        output_dir=tc.output_dir,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        num_train_epochs=tc.num_train_epochs,
        warmup_ratio=tc.warmup_ratio,
        warmup_steps=tc.warmup_steps,
        lr_scheduler_type=tc.lr_scheduler_type,
        optim=optim,
        fp16=fp16,
        bf16=bf16,
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
        ddp_find_unused_parameters=(
            tc.ddp_find_unused_parameters if world_size > 1 else None
        ),
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
    world_size = get_world_size()
    local_rank = get_local_rank()
    LOGGER.info("=" * 60)
    LOGGER.info("训练启动")
    LOGGER.info(
        "设备: %s | world_size=%s | local_rank=%s",
        describe_device(device),
        world_size,
        local_rank if local_rank >= 0 else "single-process",
    )
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
        "训练参数: lr=%.1e, epochs=%s, batch=%s, grad_accum=%s, scheduler=%s, warmup=%s, fp16=%s, bf16=%s, ddp_unused=%s",
        tc.learning_rate, tc.num_train_epochs,
        tc.per_device_train_batch_size,
        tc.gradient_accumulation_steps,
        tc.lr_scheduler_type,
        f"{tc.warmup_steps} steps" if tc.warmup_steps > 0 else f"{tc.warmup_ratio:.2%}",
        tc.fp16,
        tc.bf16,
        tc.ddp_find_unused_parameters,
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


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


# ============================================================
#  主函数
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    validate_parallelism_config(config)
    device = get_runtime_device()
    set_seed(config.training.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(config.training.output_dir)
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
        train_split,
        tokenizer,
        config.model.max_length,
        is_train=True,
        include_swap=config.data.include_swap_train,
    )
    valid_dataset = build_dataset(
        valid_split,
        tokenizer,
        config.model.max_length,
        is_train=True,
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
    if trainer.is_world_process_zero():
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
    if trainer.is_world_process_zero():
        save_artifacts(output_dir, unwrap_model(trainer.model), tokenizer, config, metrics)

        elapsed = format_seconds(time.perf_counter() - started_at)
        LOGGER.info("全部完成，总耗时: %s", elapsed)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
