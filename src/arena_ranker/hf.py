"""
模型加载模块 — 负责 tokenizer 和 QLoRA 分类模型的加载。

核心策略：
  1. Tokenizer: 加载后检查 pad_token，若缺失则设为 eos_token
  2. 模型: 通过 BitsAndBytesConfig 进行 4-bit 量化加载
  3. LoRA: 使用 PEFT 的 LoraConfig 注入 adapter，同时保存分类头 (score)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from arena_ranker.config import ModelConfig, NUM_LABELS

LOGGER = logging.getLogger("arena_ranker.hf")


# ============================================================
#  辅助函数
# ============================================================

def _describe_model_source(model_name: str) -> str:
    """描述模型来源，方便排查加载问题。"""
    source_path = Path(model_name).expanduser()
    if source_path.exists():
        config_path = source_path / "config.json"
        if config_path.exists():
            return f"检测到本地模型目录：{source_path}"
        return (
            f"检测到同名本地目录 {source_path}，但缺少 config.json。"
            "transformers 会把它当成本地模型路径并直接加载失败。"
        )
    if "/" in model_name:
        return (
            f"`{model_name}` 将被当作 HuggingFace 仓库 ID。"
            "首次运行需联网下载；离线环境请预下载到本地再修改 model_name。"
        )
    return f"`{model_name}` 既不是本地目录，也不是标准 HuggingFace 仓库 ID。"


def _get_local_rank() -> int | None:
    """读取 torch.distributed 注入的 LOCAL_RANK。"""
    raw = os.environ.get("LOCAL_RANK")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("忽略非法 LOCAL_RANK=%r", raw)
        return None


# ============================================================
#  Tokenizer
# ============================================================

def load_tokenizer(config: ModelConfig):
    """
    加载 tokenizer 并配置 pad_token。

    Qwen 系列模型通常没有默认 pad_token，这里将其设为 eos_token。
    同时设置 padding_side="left"（decoder-only 模型做分类时的推荐做法）。
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
    except OSError as exc:
        raise RuntimeError(
            f"加载 tokenizer 失败: {exc}\n"
            f"{_describe_model_source(config.model_name)}"
        ) from exc

    # 设置 pad_token（Qwen 通常没有默认值）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        LOGGER.info(
            "Tokenizer 缺少 pad_token，已设为 eos_token: '%s' (id=%s)",
            tokenizer.pad_token,
            tokenizer.pad_token_id,
        )

    # decoder-only 模型做分类时，左侧填充可避免最后一个 token 为 pad
    tokenizer.padding_side = "left"

    return tokenizer


# ============================================================
#  量化配置
# ============================================================

def _build_bnb_config(config: ModelConfig) -> BitsAndBytesConfig | None:
    """构建 BitsAndBytesConfig 用于 4-bit QLoRA 量化。"""
    if not config.load_in_4bit:
        return None
    # 4-bit 量化需要 CUDA；CPU 模式下跳过
    if not torch.cuda.is_available():
        LOGGER.warning("未检测到 CUDA，跳过 4-bit 量化（将以 float32 加载模型）")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


# ============================================================
#  LoRA 配置
# ============================================================

def _build_lora_config(config: ModelConfig) -> LoraConfig:
    """
    构建 LoRA 适配器配置。

    关键参数说明:
      - target_modules: 对所有 attention + FFN 投影层注入 LoRA
      - modules_to_save=["score"]: 分类头 (score) 必须全量训练并保存，
        否则随机初始化的分类头不会被持久化
      - task_type=SEQ_CLS: 告诉 PEFT 这是一个序列分类任务
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_CLS,
        modules_to_save=config.lora_modules_to_save,
    )


# ============================================================
#  加载分类模型 (QLoRA)
# ============================================================

def load_model(config: ModelConfig, tokenizer=None):
    """
    加载 QLoRA 分类模型，完整流程：
      1. 使用 BitsAndBytesConfig 进行 4-bit 量化加载
      2. 通过 AutoModelForSequenceClassification 替换语言建模头为分类头 (num_labels=3)
      3. 使用 prepare_model_for_kbit_training 准备量化模型
      4. 注入 LoRA adapter

    Returns:
        PEFT 包装后的模型（若 use_lora=True），否则原始模型。
    """
    bnb_config = _build_bnb_config(config)
    local_rank = _get_local_rank()

    # Qwen 系列部分配置会带嵌套 text_config，需要确保 num_labels 正确传播。
    try:
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=NUM_LABELS,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
    except OSError as exc:
        raise RuntimeError(
            f"加载模型配置失败: {exc}\n"
            f"{_describe_model_source(config.model_name)}"
        ) from exc

    # 某些 Qwen 配置不会自动把 num_labels 传播到 text_config
    if hasattr(model_config, "text_config"):
        model_config.text_config.num_labels = NUM_LABELS
        LOGGER.info("已手动将 num_labels=%s 传播到 text_config", NUM_LABELS)

    if local_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        LOGGER.info("分布式训练: 当前进程绑定到 cuda:%s", local_rank)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs = dict(
        pretrained_model_name_or_path=config.model_name,
        config=model_config,
        quantization_config=bnb_config,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
        torch_dtype=dtype,
    )
    # QLoRA + DDP 需要确保每个 rank 只把量化模型加载到本地 GPU。
    if bnb_config is not None and local_rank is not None:
        model_kwargs["device_map"] = {"": local_rank}
    try:
        model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
    except OSError as exc:
        raise RuntimeError(
            f"加载模型失败: {exc}\n"
            f"{_describe_model_source(config.model_name)}"
        ) from exc

    # 对齐 pad_token_id
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.config, "text_config"):
            model.config.text_config.pad_token_id = tokenizer.pad_token_id

    if not config.use_lora:
        return model

    # QLoRA 特有步骤: 准备量化模型以适配训练（仅在实际量化时）
    if config.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lora_config = _build_lora_config(config)
    model = get_peft_model(model, lora_config)

    # 打印可训练参数统计
    model.print_trainable_parameters()

    return model
