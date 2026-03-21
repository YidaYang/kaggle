"""
数据处理模块 — 加载 CSV、文本清洗、chat template tokenization、metrics 计算。

核心流程：
  1. 读取 CSV → 清洗文本字段 (解析 JSON 数组 / Python list 等格式)
  2. 使用 tokenizer.apply_chat_template() 构建 system + user 对话
  3. 通过 Dataset.map() 完成 tokenization，输出 input_ids / attention_mask / labels
  4. 提供 compute_metrics() 供 Trainer 使用
"""

from __future__ import annotations

import ast
import json
from itertools import zip_longest
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from arena_ranker.config import (
    LABEL_COLUMNS,
    LABEL_TO_ID,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    VERDICT_SUFFIX,
    DataConfig,
)


# ============================================================
#  文本解析 & 清洗
# ============================================================

def _parse_conversation_field(value: Any) -> list[str]:
    """
    将原始 CSV 字段解析为字符串列表。
    支持三种输入格式：
      - 普通字符串 → 直接包入列表
      - JSON 数组字符串 → json.loads 解析
      - Python list 字符串 → ast.literal_eval 解析
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not isinstance(value, str):
        return [str(value)]
    text = value.strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            continue
    return [text]


def _normalize_conversation_turns(value: Any, max_chars: int) -> list[str]:
    """将原始字段转为按轮次保序的字符串列表，并在总字符预算内截断。"""
    chunks = _parse_conversation_field(value)
    normalized = [chunk.strip() for chunk in chunks if str(chunk).strip()]
    if not normalized:
        return []

    kept: list[str] = []
    used_chars = 0
    for chunk in normalized:
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        truncated = chunk[:remaining]
        if truncated:
            kept.append(truncated)
            used_chars += len(truncated)
    return kept


def _build_conversation_text(
    prompt_turns: list[str],
    response_a_turns: list[str],
    response_b_turns: list[str],
) -> str:
    """按轮次交错拼接 prompt / A / B，保留多轮对话结构。"""
    head = "<|The Start of Conversation between a User and two Assistants|>"
    tail = "<|The End of Conversation between a User and two Assistants|>\n"
    parts = []
    for prompt, response_a, response_b in zip_longest(
        prompt_turns,
        response_a_turns,
        response_b_turns,
        fillvalue="null",
    ):
        parts.append(
            f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n\n### Assistant B:\n{response_b}\n"
        )
    return head + "".join(parts) + tail


def _build_label(row: pd.Series) -> int:
    """从 one-hot 标签列 → 单一整数标签 (0=A胜, 1=B胜, 2=平局)。"""
    for col, label_id in LABEL_TO_ID.items():
        if int(row[col]) == 1:
            return label_id
    raise ValueError(f"无效标签行: {row[LABEL_COLUMNS].to_dict()}")


# ============================================================
#  数据加载 & 预处理
# ============================================================

def load_and_preprocess(
    csv_path: str,
    max_chars: int = 6000,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    加载 CSV 并预处理文本字段。

    Returns:
        DataFrame，包含 id / prompt_turns / response_a_turns / response_b_turns
        以及 (仅训练集) labels 列。
    """
    df = pd.read_csv(csv_path)
    df["prompt_turns"] = df["prompt"].map(
        lambda x: _normalize_conversation_turns(x, max_chars)
    )
    df["response_a_turns"] = df["response_a"].map(
        lambda x: _normalize_conversation_turns(x, max_chars)
    )
    df["response_b_turns"] = df["response_b"].map(
        lambda x: _normalize_conversation_turns(x, max_chars)
    )
    if is_train:
        df["labels"] = df.apply(_build_label, axis=1)
    return df


def split_train_valid(
    df: pd.DataFrame,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 stratified split 划分训练集和验证集。"""
    train_df, valid_df = train_test_split(
        df,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=df["labels"],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


# ============================================================
#  Chat Template Tokenization
# ============================================================

def _build_chat_messages(conversation: str) -> list[dict]:
    """
    构建对话消息列表，用于 apply_chat_template。
      - system: 评委角色指令
      - user:   保留轮次结构的完整对话
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(conversation=conversation),
        },
    ]


def _tokenize_single(example: dict, tokenizer, max_length: int) -> dict:
    """
    对单条样本做 tokenization（供 Dataset.map 调用）。

    步骤：
      1. 用 apply_chat_template 将对话格式化为文本
      2. 用 tokenizer 编码并截断到 max_length
    """
    conversation = _build_conversation_text(
        example["prompt_turns"],
        example["response_a_turns"],
        example["response_b_turns"],
    )
    messages = _build_chat_messages(conversation)
    # 先得到格式化文本，再单独 tokenize 以精确控制截断
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + VERDICT_SUFFIX
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,  # chat template 已经包含所有特殊 token
    )
    return encoded


def build_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 1024,
    is_train: bool = True,
    include_swap: bool = False,
    swap_pairs: bool = False,
) -> Dataset:
    """
    从 DataFrame 构建 tokenized HuggingFace Dataset。

    输出列（训练）: input_ids, attention_mask, labels
    输出列（测试）: input_ids, attention_mask
    """
    cols = ["id", "prompt_turns", "response_a_turns", "response_b_turns"]
    if is_train:
        cols.append("labels")
    base_df = df[cols].copy()

    if swap_pairs or include_swap:
        swap_df = base_df.copy()
        swap_df["response_a_turns"] = base_df["response_b_turns"]
        swap_df["response_b_turns"] = base_df["response_a_turns"]
        if is_train:
            swap_df["labels"] = swap_df["labels"].map(
                lambda x: 1 if x == 0 else 0 if x == 1 else x
            )
        if swap_pairs:
            base_df = swap_df
        elif include_swap:
            base_df = pd.concat([base_df, swap_df], ignore_index=True)

    dataset = Dataset.from_pandas(base_df, preserve_index=False)

    dataset = dataset.map(
        lambda x: _tokenize_single(x, tokenizer, max_length),
        desc="Tokenizing",
    )

    # 移除文本列，只保留模型需要的数值列
    remove_cols = ["prompt_turns", "response_a_turns", "response_b_turns", "id"]
    dataset = dataset.remove_columns(
        [c for c in remove_cols if c in dataset.column_names]
    )
    return dataset


# ============================================================
#  Metrics
# ============================================================

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax 实现（避免引入 scipy 依赖）。"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def compute_metrics(eval_pred) -> dict[str, float]:
    """
    Trainer 的 compute_metrics 回调。

    计算:
      - log_loss: 概率经裁剪后计算，避免 log(0) 导致极端值
      - accuracy: 分类准确率
    """
    logits, labels = eval_pred

    # 处理可能的 NaN（CPU 模式或数值不稳定时可能出现）
    nan_mask = np.isnan(logits).any(axis=-1)
    if nan_mask.any():
        logits = np.copy(logits)
        logits[nan_mask] = 0.0  # NaN 行替换为均匀分布

    # logits → 概率
    probs = _softmax(logits, axis=-1)

    # 概率裁剪 (对 log_loss 评分非常重要)
    eps = 1e-7
    probs = np.clip(probs, eps, 1.0 - eps)
    probs = probs / probs.sum(axis=1, keepdims=True)

    logloss = float(log_loss(labels, probs, labels=[0, 1, 2]))
    preds = np.argmax(logits, axis=-1)
    acc = float(accuracy_score(labels, preds))

    return {"log_loss": logloss, "accuracy": acc}
