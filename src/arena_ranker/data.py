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


def _normalize_text(value: Any, max_chars: int) -> str:
    """将原始字段转为纯文本，按 max_chars 截断。"""
    chunks = _parse_conversation_field(value)
    text = "\n".join(chunk.strip() for chunk in chunks if str(chunk).strip())
    return text[:max_chars]


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
        DataFrame，包含 id / prompt_clean / response_a_clean / response_b_clean
        以及 (仅训练集) labels 列。
    """
    df = pd.read_csv(csv_path)
    df["prompt_clean"] = df["prompt"].map(lambda x: _normalize_text(x, max_chars))
    df["response_a_clean"] = df["response_a"].map(lambda x: _normalize_text(x, max_chars))
    df["response_b_clean"] = df["response_b"].map(lambda x: _normalize_text(x, max_chars))
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

def _build_chat_messages(prompt: str, response_a: str, response_b: str) -> list[dict]:
    """
    构建对话消息列表，用于 apply_chat_template。
      - system: 评委角色指令
      - user:   prompt + response_a + response_b 的完整内容
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
            ),
        },
    ]


def _tokenize_single(example: dict, tokenizer, max_length: int) -> dict:
    """
    对单条样本做 tokenization（供 Dataset.map 调用）。

    步骤：
      1. 用 apply_chat_template 将对话格式化为文本
      2. 用 tokenizer 编码并截断到 max_length
    """
    messages = _build_chat_messages(
        example["prompt_clean"],
        example["response_a_clean"],
        example["response_b_clean"],
    )
    # 先得到格式化文本，再单独 tokenize 以精确控制截断
    # add_generation_prompt=False: 分类任务不需要 assistant 角色前缀
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
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
) -> Dataset:
    """
    从 DataFrame 构建 tokenized HuggingFace Dataset。

    输出列（训练）: input_ids, attention_mask, labels
    输出列（测试）: input_ids, attention_mask
    """
    cols = ["id", "prompt_clean", "response_a_clean", "response_b_clean"]
    if is_train:
        cols.append("labels")
    dataset = Dataset.from_pandas(df[cols], preserve_index=False)

    dataset = dataset.map(
        lambda x: _tokenize_single(x, tokenizer, max_length),
        desc="Tokenizing",
    )

    # 移除文本列，只保留模型需要的数值列
    remove_cols = ["prompt_clean", "response_a_clean", "response_b_clean", "id"]
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
