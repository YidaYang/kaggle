from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from arena_ranker.config import load_config
from arena_ranker.data import ArenaCollator, ArenaPreferenceDataset, ID_TO_LABEL, load_test_dataframe
from arena_ranker.hf import ensure_supported_transformers_version
from arena_ranker.modeling import PreferenceClassifier


LOGGER = logging.getLogger("arena_ranker.predict")
PROBABILITY_EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission with trained Arena ranker.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--output-path", type=str, default=None)
    return parser.parse_args()


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


def normalize_probabilities(logits: torch.Tensor, epsilon: float = PROBABILITY_EPSILON) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    probs = probs.clamp(min=epsilon, max=1.0 - epsilon)
    return probs / probs.sum(dim=-1, keepdim=True)


def main() -> None:
    setup_logging()
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    config = load_config(checkpoint_dir / "config.yaml")
    started_at = time.perf_counter()

    ensure_supported_transformers_version()
    tokenizer_dir = checkpoint_dir / "tokenizer"
    if not (tokenizer_dir / "tokenizer_config.json").exists():
        raise RuntimeError(f"缺少 tokenizer 目录或文件：{tokenizer_dir}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    model = PreferenceClassifier(config.model)
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    LOGGER.info("预测启动")
    LOGGER.info("checkpoint: %s", checkpoint_dir)
    LOGGER.info("设备: %s", describe_device(device))

    test_df = load_test_dataframe(args.data_dir, config.data)
    loader = DataLoader(
        ArenaPreferenceDataset(test_df, with_labels=False),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=ArenaCollator(tokenizer, config.model),
    )
    LOGGER.info(
        "待预测样本: %s | batch_size=%s | max_length=%s",
        len(test_df),
        config.training.batch_size,
        config.model.max_length,
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            outputs = model(
                inputs=move_inputs_to_device(batch.inputs, device),
            )
            probs = normalize_probabilities(outputs.logits).cpu().numpy()
            for sample_id, prob in zip(batch.ids, probs, strict=True):
                rows.append(
                    {
                        "id": sample_id,
                        ID_TO_LABEL[0]: prob[0],
                        ID_TO_LABEL[1]: prob[1],
                        ID_TO_LABEL[2]: prob[2],
                    }
                )

    output_path = Path(args.output_path) if args.output_path else checkpoint_dir / "submission.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    LOGGER.info("预测完成，生成 %s 行结果，耗时=%s", len(rows), format_seconds(time.perf_counter() - started_at))
    LOGGER.info("submission 已保存到: %s", output_path)
    print(f"saved submission to {output_path}")


if __name__ == "__main__":
    main()
