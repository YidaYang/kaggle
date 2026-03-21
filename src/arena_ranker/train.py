from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from arena_ranker.config import AppConfig, load_config
from arena_ranker.data import ArenaCollator, ArenaPreferenceDataset, split_train_valid, load_train_dataframe
from arena_ranker.hf import load_tokenizer
from arena_ranker.modeling import PreferenceClassifier
from arena_ranker.swanlab_utils import SwanlabTracker


LOGGER = logging.getLogger("arena_ranker.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen embedding classifier for Arena ranking.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--segment-budget", dest="use_segment_budget", action="store_true")
    parser.add_argument("--disable-segment-budget", dest="use_segment_budget", action="store_false")
    parser.add_argument("--prompt-budget", type=int, default=None)
    parser.add_argument("--response-budget", type=int, default=None)
    parser.add_argument("--response-head-tokens", type=int, default=None)
    parser.add_argument("--response-tail-tokens", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true")
    parser.add_argument("--disable-freeze-encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument("--classifier-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use-lora", dest="use_lora", action="store_true")
    parser.add_argument("--disable-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-bias", type=str, default=None)
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None)
    parser.add_argument("--enable-swanlab", dest="enable_swanlab", action="store_true")
    parser.add_argument("--disable-swanlab", dest="enable_swanlab", action="store_false")
    parser.add_argument("--swanlab-project", type=str, default=None)
    parser.add_argument("--swanlab-experiment-name", type=str, default=None)
    parser.add_argument("--swanlab-workspace", type=str, default=None)
    parser.add_argument("--swanlab-mode", type=str, default=None)
    parser.set_defaults(
        use_lora=None,
        gradient_checkpointing=None,
        freeze_encoder=None,
        enable_swanlab=None,
        use_segment_budget=None,
    )
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.model_name:
        config.model.model_name = args.model_name
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.training.grad_accum_steps = args.grad_accum_steps
    if args.max_length is not None:
        config.model.max_length = args.max_length
    if args.use_segment_budget is not None:
        config.model.use_segment_budget = args.use_segment_budget
    if args.prompt_budget is not None:
        config.model.prompt_budget = args.prompt_budget
    if args.response_budget is not None:
        config.model.response_budget = args.response_budget
    if args.response_head_tokens is not None:
        config.model.response_head_tokens = args.response_head_tokens
    if args.response_tail_tokens is not None:
        config.model.response_tail_tokens = args.response_tail_tokens
    if args.cache_dir is not None:
        config.model.cache_dir = args.cache_dir
    if args.local_files_only:
        config.model.local_files_only = True
    if args.freeze_encoder is not None:
        config.model.freeze_encoder = args.freeze_encoder
    if args.classifier_only:
        config.model.freeze_encoder = True
        config.model.use_lora = False
        config.training.gradient_checkpointing = False
    if args.gradient_checkpointing is not None:
        config.training.gradient_checkpointing = args.gradient_checkpointing
    if args.use_lora is not None:
        config.model.use_lora = args.use_lora
    if args.lora_r is not None:
        config.model.lora_r = args.lora_r
    if args.lora_alpha is not None:
        config.model.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        config.model.lora_dropout = args.lora_dropout
    if args.lora_bias is not None:
        config.model.lora_bias = args.lora_bias
    if args.lora_target_modules is not None:
        config.model.lora_target_modules = args.lora_target_modules
    if args.enable_swanlab is not None:
        config.swanlab.enabled = args.enable_swanlab
    if args.swanlab_project is not None:
        config.swanlab.project = args.swanlab_project
    if args.swanlab_experiment_name is not None:
        config.swanlab.experiment_name = args.swanlab_experiment_name
    if args.swanlab_workspace is not None:
        config.swanlab.workspace = args.swanlab_workspace
    if args.swanlab_mode is not None:
        config.swanlab.mode = args.swanlab_mode
    return config


def finalize_training_mode(config: AppConfig) -> AppConfig:
    if config.model.freeze_encoder and config.model.use_lora:
        LOGGER.info("检测到 freeze_encoder 与 LoRA 同时开启，已自动关闭 LoRA，仅训练分类头。")
        config.model.use_lora = False

    if config.model.freeze_encoder and config.training.gradient_checkpointing:
        LOGGER.info("检测到 encoder 已冻结，已自动关闭 gradient checkpointing。")
        config.training.gradient_checkpointing = False

    return config


def get_training_mode(config: AppConfig) -> str:
    if config.model.freeze_encoder:
        return "classifier-only"
    if config.model.use_lora:
        return "lora"
    return "full-finetune"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_inputs_to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def format_seconds(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    gpu_name = torch.cuda.get_device_name(device)
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"{gpu_name} ({total_memory_gb:.1f} GB)"


def log_run_summary(config: AppConfig, device: torch.device, train_size: int, valid_size: int, total_steps: int) -> None:
    training_mode = get_training_mode(config)
    LOGGER.info("训练启动")
    LOGGER.info("设备: %s", describe_device(device))
    LOGGER.info("训练模式: %s", training_mode)
    LOGGER.info(
        "数据集: train=%s, valid=%s, batch_size=%s, grad_accum=%s, epochs=%s",
        train_size,
        valid_size,
        config.training.batch_size,
        config.training.grad_accum_steps,
        config.training.epochs,
    )
    LOGGER.info(
        "模型: %s | max_length=%s | freeze_encoder=%s | LoRA=%s | gradient_checkpointing=%s | segment_budget=%s",
        config.model.model_name,
        config.model.max_length,
        "on" if config.model.freeze_encoder else "off",
        "on" if config.model.use_lora else "off",
        "on" if config.training.gradient_checkpointing else "off",
        "on" if config.model.use_segment_budget else "off",
    )
    if config.model.use_segment_budget:
        LOGGER.info(
            "分段预算: prompt=%s | response=%s | response_head=%s | response_tail=%s",
            config.model.prompt_budget,
            config.model.response_budget,
            config.model.response_head_tokens,
            config.model.response_tail_tokens,
        )
    if config.model.use_lora:
        LOGGER.info(
            "LoRA 配置: r=%s, alpha=%s, dropout=%.3f, target_modules=%s",
            config.model.lora_r,
            config.model.lora_alpha,
            config.model.lora_dropout,
            ",".join(config.model.lora_target_modules),
        )
    if training_mode == "classifier-only":
        LOGGER.info("当前仅训练分类头，encoder 作为冻结特征提取器使用。")
    elif training_mode == "full-finetune":
        LOGGER.info("当前进行全参数微调，encoder 和分类头都会参与训练。")
    LOGGER.info("优化步数: total=%s, warmup=%s", total_steps, int(total_steps * config.training.warmup_ratio))
    LOGGER.info("输出目录: %s", config.training.output_dir)
    LOGGER.info("SwanLab: %s", "on" if config.swanlab.enabled else "off")


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="valid", leave=False):
            outputs = model(
                inputs=move_inputs_to_device(batch.inputs, device),
                labels=batch.labels.to(device) if batch.labels is not None else None,
            )
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch.labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1, 2])),
    }


def build_optimizer(model: PreferenceClassifier, config: AppConfig) -> AdamW:
    encoder_params = []
    classifier_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("classifier"):
            classifier_params.append(parameter)
        else:
            encoder_params.append(parameter)

    return AdamW(
        [
            {
                "params": encoder_params,
                "lr": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
            },
            {
                "params": classifier_params,
                "lr": config.training.classifier_learning_rate,
                "weight_decay": config.training.weight_decay,
            },
        ]
    )


def save_artifacts(output_dir: Path, model: PreferenceClassifier, tokenizer, metrics: dict[str, float], config: AppConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    config.save(output_dir / "config.yaml")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()
    config = finalize_training_mode(apply_overrides(load_config(args.config), args))
    set_seed(config.training.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(config.training.output_dir)

    train_df = load_train_dataframe(data_dir, config.data)
    train_split, valid_split = split_train_valid(train_df, config.data)

    tokenizer = load_tokenizer(config.model)
    collator = ArenaCollator(tokenizer, config.model)
    train_loader = DataLoader(
        ArenaPreferenceDataset(train_split, with_labels=True),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        ArenaPreferenceDataset(valid_split, with_labels=True),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PreferenceClassifier(config.model).to(device)
    if config.training.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.print_trainable_parameters()
    optimizer = build_optimizer(model, config)

    total_steps = math.ceil(len(train_loader) / config.training.grad_accum_steps) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.amp and device.type == "cuda")
    log_run_summary(config, device, len(train_split), len(valid_split), total_steps)
    tracker = SwanlabTracker(config)
    tracker.start()

    best_metrics = {"accuracy": 0.0, "log_loss": float("inf")}
    best_state = None
    training_started_at = time.perf_counter()
    global_step = 0

    try:
        for epoch in range(config.training.epochs):
            epoch_started_at = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{config.training.epochs}")
            running_loss = 0.0

            for step, batch in enumerate(progress, start=1):
                with torch.amp.autocast(device_type=device.type, enabled=config.training.amp and device.type == "cuda"):
                    outputs = model(
                        inputs=move_inputs_to_device(batch.inputs, device),
                        labels=batch.labels.to(device) if batch.labels is not None else None,
                    )
                    loss = outputs.loss / config.training.grad_accum_steps

                batch_loss = loss.item() * config.training.grad_accum_steps
                running_loss += batch_loss
                scaler.scale(loss).backward()

                if step % config.training.grad_accum_steps == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    tracker.log(
                        {
                            "train/loss": batch_loss,
                            "train/avg_loss": running_loss / step,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

                if step % config.training.log_every == 0 or step == len(train_loader):
                    avg_loss = running_loss / step
                    progress.set_postfix(loss=f"{batch_loss:.4f}", avg_loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            metrics = evaluate(model, valid_loader, device)
            avg_epoch_loss = running_loss / max(len(train_loader), 1)
            epoch_duration = format_seconds(time.perf_counter() - epoch_started_at)
            LOGGER.info(
                "Epoch %s/%s 完成 | train_loss=%.4f | valid_accuracy=%.4f | valid_log_loss=%.4f | 耗时=%s",
                epoch + 1,
                config.training.epochs,
                avg_epoch_loss,
                metrics["accuracy"],
                metrics["log_loss"],
                epoch_duration,
            )
            tracker.log(
                {
                    "epoch/train_loss": avg_epoch_loss,
                    "epoch/valid_accuracy": metrics["accuracy"],
                    "epoch/valid_log_loss": metrics["log_loss"],
                    "epoch/index": epoch + 1,
                },
                step=global_step,
            )
            if metrics["log_loss"] < best_metrics["log_loss"]:
                best_metrics = metrics
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                LOGGER.info(
                    "刷新最佳结果 | valid_accuracy=%.4f | valid_log_loss=%.4f",
                    best_metrics["accuracy"],
                    best_metrics["log_loss"],
                )
                tracker.log(
                    {
                        "best/accuracy": best_metrics["accuracy"],
                        "best/log_loss": best_metrics["log_loss"],
                        "best/epoch": epoch + 1,
                    },
                    step=global_step,
                )
    finally:
        tracker.finish()

    if best_state is not None:
        model.load_state_dict(best_state)

    save_artifacts(output_dir, model, tokenizer, best_metrics, config)
    LOGGER.info("训练完成，总耗时=%s", format_seconds(time.perf_counter() - training_started_at))
    LOGGER.info("最佳指标: accuracy=%.4f | log_loss=%.4f", best_metrics["accuracy"], best_metrics["log_loss"])
    LOGGER.info("已保存模型、tokenizer、配置和指标到: %s", output_dir)
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
