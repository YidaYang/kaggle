"""
推理脚本 — 加载训练好的 QLoRA adapter，对测试集生成 submission.csv。

流程：
  1. 重新加载基座模型 (4-bit 量化)
  2. 加载保存的 LoRA adapter + 分类头
  3. 遍历测试集，生成三分类概率
  4. 概率裁剪 + 归一化后写入 submission.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

from arena_ranker.config import ID_TO_LABEL, NUM_LABELS, load_config
from arena_ranker.data import build_dataset, load_and_preprocess

LOGGER = logging.getLogger("arena_ranker.predict")
PROBABILITY_EPSILON = 1e-6


# ============================================================
#  CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate submission with trained QLoRA Arena ranker."
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="训练产物目录 (包含 model/, tokenizer/, config.yaml)")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="test.csv 所在目录")
    parser.add_argument("--output-path", type=str, default=None,
                        help="输出 submission.csv 路径")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="推理 batch size")
    return parser.parse_args()


# ============================================================
#  工具函数
# ============================================================

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


def normalize_probabilities(
    logits: torch.Tensor,
    epsilon: float = PROBABILITY_EPSILON,
) -> np.ndarray:
    """logits → 裁剪后的概率（确保 log_loss 不会爆炸）。"""
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs


# ============================================================
#  加载推理模型
# ============================================================

def load_inference_model(checkpoint_dir: Path, config):
    """
    加载训练好的 QLoRA 模型用于推理。

    步骤：
      1. 重新加载基座模型 (4-bit 量化)
      2. 从 adapter 目录加载 LoRA 权重 + score 分类头
    """
    mc = config.model
    adapter_dir = checkpoint_dir / "model"

    # 量化配置
    bnb_config = None
    if mc.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mc.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=mc.bnb_4bit_use_double_quant,
        )

    # 加载配置并处理 VLM num_labels 传播
    model_config = AutoConfig.from_pretrained(
        mc.model_name,
        num_labels=NUM_LABELS,
        trust_remote_code=True,
        cache_dir=mc.cache_dir,
        local_files_only=mc.local_files_only,
    )
    if hasattr(model_config, "text_config"):
        model_config.text_config.num_labels = NUM_LABELS

    # 加载基座模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        mc.model_name,
        config=model_config,
        quantization_config=bnb_config,
        trust_remote_code=True,
        cache_dir=mc.cache_dir,
        local_files_only=mc.local_files_only,
        torch_dtype=torch.float16,
    )

    # 加载 LoRA adapter
    if mc.use_lora and adapter_dir.exists():
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        LOGGER.info("已加载 LoRA adapter: %s", adapter_dir)
    else:
        model = base_model
        LOGGER.info("未使用 LoRA adapter，加载完整模型")

    model.eval()
    return model


# ============================================================
#  主函数
# ============================================================

def main() -> None:
    setup_logging()
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    started_at = time.perf_counter()

    # ---- 1. 加载配置 ----
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"缺少配置文件: {config_path}")
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("预测启动")
    LOGGER.info("checkpoint: %s", checkpoint_dir)
    LOGGER.info("设备: %s", describe_device(device))

    # ---- 2. 加载 tokenizer ----
    tokenizer_dir = checkpoint_dir / "tokenizer"
    if not tokenizer_dir.exists():
        raise RuntimeError(f"缺少 tokenizer 目录: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ---- 3. 加载模型 ----
    model = load_inference_model(checkpoint_dir, config)

    # ---- 4. 加载测试数据 ----
    test_csv = Path(args.data_dir) / config.data.test_path
    LOGGER.info("加载测试数据: %s", test_csv)
    test_df = load_and_preprocess(
        str(test_csv),
        max_chars=config.data.text_max_chars,
        is_train=False,
    )
    test_ids = test_df["id"].tolist()

    test_dataset = build_dataset(
        test_df, tokenizer, config.model.max_length, is_train=False,
    )
    LOGGER.info("待预测样本: %s", len(test_dataset))

    # ---- 5. 推理 ----
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = normalize_probabilities(outputs.logits)
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    # ---- 6. 写出 submission.csv ----
    rows = []
    for sample_id, prob in zip(test_ids, all_probs, strict=True):
        rows.append({
            "id": sample_id,
            ID_TO_LABEL[0]: prob[0],
            ID_TO_LABEL[1]: prob[1],
            ID_TO_LABEL[2]: prob[2],
        })

    output_path = (
        Path(args.output_path)
        if args.output_path
        else checkpoint_dir / "submission.csv"
    )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    LOGGER.info(
        "预测完成: %s 行, 耗时=%s",
        len(rows), format_seconds(time.perf_counter() - started_at),
    )
    LOGGER.info("submission 已保存到: %s", output_path)
    print(f"saved submission to {output_path}")


if __name__ == "__main__":
    main()
