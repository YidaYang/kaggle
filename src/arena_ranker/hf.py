from __future__ import annotations

from pathlib import Path

import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

from arena_ranker.config import ModelConfig


def _is_transformers_v5_or_newer() -> bool:
    major, *_ = transformers.__version__.split(".", 1)
    return int(major) >= 5


def ensure_supported_transformers_version() -> None:
    if not _is_transformers_v5_or_newer():
        return

    raise RuntimeError(
        "当前环境的 transformers 版本为 "
        f"{transformers.__version__}。该项目只验证了 4.x 版本，并且已在 pyproject.toml 中收紧为 "
        "\"transformers>=4.55.0,<5\"。"
        "请先在项目目录执行 `uv sync`，再重新运行训练或预测。"
    )


def _describe_model_source(model_name: str) -> str:
    source_path = Path(model_name).expanduser()
    if source_path.exists():
        config_path = source_path / "config.json"
        if config_path.exists():
            return f"检测到本地模型目录：{source_path}"
        return (
            f"检测到同名本地目录：{source_path}，但其中缺少 `config.json`。"
            "这会让 transformers 把它当成本地模型目录处理，并直接加载失败。"
        )

    if "/" in model_name:
        return (
            f"`{model_name}` 会被当作 Hugging Face 仓库 ID。"
            "首次运行需要联网下载；离线环境下请先预下载到本地目录，再把 `model_name` 改成本地绝对路径。"
        )

    return f"`{model_name}` 既不是已存在的本地目录，也不是标准的 Hugging Face 仓库 ID。"


def _load_error(action: str, model_name: str, exc: OSError) -> RuntimeError:
    return RuntimeError(
        f"{action}失败：{exc}\n"
        f"{_describe_model_source(model_name)}\n"
        "排查建议：\n"
        "1. 如果你依赖在线下载，确认当前环境能访问 huggingface.co。\n"
        "2. 如果你在离线环境运行，先把模型下载到本地，再把 `model_name` 改为本地路径。\n"
        "3. 如果工作目录下存在同名目录 `Qwen/Qwen3-Embedding-0.6B`，请删除或重命名该目录。"
    )


def load_tokenizer(config: ModelConfig):
    ensure_supported_transformers_version()
    try:
        return AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
    except OSError as exc:
        raise _load_error("加载 tokenizer", config.model_name, exc)


def load_encoder(config: ModelConfig):
    ensure_supported_transformers_version()
    try:
        encoder = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
        if not config.use_lora:
            return encoder

        return get_peft_model(
            encoder,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,
                target_modules=config.lora_target_modules,
                task_type=TaskType[config.lora_task_type.upper()],
            ),
        )
    except OSError as exc:
        raise _load_error("加载 encoder", config.model_name, exc)
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"无效的 LoRA task_type 配置：{config.lora_task_type}") from exc
