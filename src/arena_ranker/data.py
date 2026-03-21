from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from arena_ranker.config import DataConfig, ModelConfig

LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
LABEL_TO_ID = {"winner_model_a": 0, "winner_model_b": 1, "winner_tie": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def _parse_conversation_field(value: Any) -> list[str]:
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
    chunks = _parse_conversation_field(value)
    text = "\n".join(chunk.strip() for chunk in chunks if str(chunk).strip())
    return text[:max_chars]


def _build_cross_encoder_text(prompt: str, response_a: str, response_b: str) -> str:
    return (
        "[PROMPT]\n"
        f"{prompt}\n\n"
        "[RESPONSE A]\n"
        f"{response_a}\n\n"
        "[RESPONSE B]\n"
        f"{response_b}"
    )


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def _truncate_text_front(tokenizer, text: str, budget: int) -> str:
    if budget <= 0 or not text:
        return ""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= budget:
        return text
    return _decode_token_ids(tokenizer, token_ids[:budget])


def _truncate_text_head_tail(tokenizer, text: str, budget: int, head_tokens: int, tail_tokens: int) -> str:
    if budget <= 0 or not text:
        return ""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= budget:
        return text

    head_size = min(max(head_tokens, 0), budget)
    tail_budget = max(budget - head_size, 0)
    tail_size = min(max(tail_tokens, 0), tail_budget)

    if head_size + tail_size < budget:
        tail_size = min(len(token_ids) - head_size, budget - head_size)

    head_text = _decode_token_ids(tokenizer, token_ids[:head_size])
    tail_text = _decode_token_ids(tokenizer, token_ids[-tail_size:]) if tail_size > 0 else ""
    if head_text and tail_text:
        return f"{head_text}\n...\n{tail_text}"
    return head_text or tail_text


def _build_target(row: pd.Series) -> int:
    for column, label_id in LABEL_TO_ID.items():
        if int(row[column]) == 1:
            return label_id
    raise ValueError(f"Invalid target row: {row[LABEL_COLUMNS].to_dict()}")


def load_train_dataframe(data_dir: str | Path, config: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / config.train_path)
    df = df.copy()
    df["prompt_text"] = df["prompt"].map(lambda value: _normalize_text(value, config.text_max_chars))
    df["response_a_text"] = df["response_a"].map(lambda value: _normalize_text(value, config.text_max_chars))
    df["response_b_text"] = df["response_b"].map(lambda value: _normalize_text(value, config.text_max_chars))
    df["label"] = df.apply(_build_target, axis=1)
    return df


def load_test_dataframe(data_dir: str | Path, config: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / config.test_path)
    df = df.copy()
    df["prompt_text"] = df["prompt"].map(lambda value: _normalize_text(value, config.text_max_chars))
    df["response_a_text"] = df["response_a"].map(lambda value: _normalize_text(value, config.text_max_chars))
    df["response_b_text"] = df["response_b"].map(lambda value: _normalize_text(value, config.text_max_chars))
    return df


def split_train_valid(df: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, valid_df = train_test_split(
        df,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def _batch_encode(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


@dataclass(slots=True)
class EncodedBatch:
    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor | None
    ids: list[int]


class ArenaPreferenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, with_labels: bool) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        item = {
            "id": int(row["id"]),
            "prompt_text": row["prompt_text"],
            "response_a_text": row["response_a_text"],
            "response_b_text": row["response_b_text"],
        }
        if self.with_labels:
            item["label"] = int(row["label"])
        return item


class ArenaCollator:
    def __init__(self, tokenizer, model_config: ModelConfig) -> None:
        self.tokenizer = tokenizer
        self.model_config = model_config

    def _build_model_input(self, item: dict[str, Any]) -> str:
        prompt_text = item["prompt_text"]
        response_a_text = item["response_a_text"]
        response_b_text = item["response_b_text"]

        if self.model_config.use_segment_budget:
            prompt_text = _truncate_text_front(self.tokenizer, prompt_text, self.model_config.prompt_budget)
            response_a_text = _truncate_text_head_tail(
                self.tokenizer,
                response_a_text,
                self.model_config.response_budget,
                self.model_config.response_head_tokens,
                self.model_config.response_tail_tokens,
            )
            response_b_text = _truncate_text_head_tail(
                self.tokenizer,
                response_b_text,
                self.model_config.response_budget,
                self.model_config.response_head_tokens,
                self.model_config.response_tail_tokens,
            )

        return _build_cross_encoder_text(prompt_text, response_a_text, response_b_text)

    def __call__(self, batch: list[dict[str, Any]]) -> EncodedBatch:
        inputs = _batch_encode(self.tokenizer, [self._build_model_input(item) for item in batch], self.model_config.max_length)
        labels = None
        if "label" in batch[0]:
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return EncodedBatch(
            inputs=inputs,
            labels=labels,
            ids=[item["id"] for item in batch],
        )
