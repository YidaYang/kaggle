"""Generate a Kaggle-ready notebook for Arena Ranker (QLoRA) training and inference."""

from __future__ import annotations

import base64
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "arena_ranker"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "kaggle_notebook.ipynb"

SOURCE_FILES = [
    "__init__.py",
    "config.py",
    "data.py",
    "hf.py",
    "modeling.py",
    "dummy_data.py",
    "train.py",
    "predict.py",
]


def make_markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(True),
    }


def make_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(True),
        "outputs": [],
        "execution_count": None,
    }


def read_source(name: str) -> str:
    return (SRC_DIR / name).read_text(encoding="utf-8")


def build_write_files_code() -> str:
    lines = [
        "import base64",
        "from pathlib import Path",
        "",
        'PKG_DIR = Path("/kaggle/working/arena_ranker")',
        "PKG_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "_FILES = {",
    ]
    for name in SOURCE_FILES:
        content = read_source(name)
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        lines.append(f'    "{name}": "{b64}",')
    lines.append("}")
    lines.append("")
    lines.extend([
        "for _name, _b64 in _FILES.items():",
        "    (PKG_DIR / _name).write_text(",
        "        base64.b64decode(_b64).decode('utf-8'),",
        "        encoding='utf-8',",
        "    )",
        '    print(f"  写入 {_name}")',
        "",
        "import sys",
        "if str(PKG_DIR.parent) not in sys.path:",
        "    sys.path.insert(0, str(PKG_DIR.parent))",
        "",
        'print("\\narena_ranker 已写入:", PKG_DIR)',
        'print("sys.path 已更新")',
    ])
    return "\n".join(lines)


def build_cells() -> list[dict]:
    cells = []

    # ── Title ──
    cells.append(make_markdown(
        "# Arena Ranker — Kaggle GPU 训练与推理\n"
        "\n"
        "本 notebook 在 Kaggle 提供的 **GPU 环境（T4 x2 / P100 x1）** 中完成：\n"
        "\n"
        "1. 安装依赖\n"
        "2. 设置项目源码\n"
        "3. 使用 **QLoRA** 微调 `Qwen/Qwen3-0.6B` 偏好分类模型\n"
        "4. 推理并生成 `submission.csv`\n"
        "\n"
        "### 核心架构\n"
        "\n"
        "```\n"
        "prompt + response_a + response_b\n"
        "        ↓\n"
        "  apply_chat_template (system + user)\n"
        "        ↓\n"
        "  Qwen3-0.6B (4-bit NF4 量化)\n"
        "        ↓\n"
        "  AutoModelForSequenceClassification\n"
        "        ↓\n"
        "  score (分类头, 3-class logits)\n"
        "```\n"
        "\n"
        "> **前置准备**\n"
        ">\n"
        "> | 项目 | 操作 |\n"
        "> |------|------|\n"
        "> | **竞赛数据** | Notebook 右侧 → Add Input → 搜索并添加竞赛数据集 |\n"
        "> | **GPU** | Settings → Accelerator → 选择 **GPU T4 ×2** 或 **GPU P100** |\n"
        "> | **联网** | Settings → Internet → **On**（用于下载 HuggingFace 模型）|\n"
        ">\n"
        "> 如果不想联网下载模型，请参考最后一节「离线模式」。\n"
    ))

    # ── Section: Install deps ──
    cells.append(make_markdown(
        "## 1. 安装依赖\n"
        "\n"
        "Kaggle 环境已预装 PyTorch，这里补装 QLoRA 训练所需的包。"
    ))
    cells.append(make_code(
        "!pip install -q \\\n"
        '    "transformers>=4.55.0" \\\n'
        '    "peft>=0.17.0" \\\n'
        '    "bitsandbytes>=0.45.0" \\\n'
        '    "datasets>=3.0.0" \\\n'
        '    "accelerate>=1.0.0" \\\n'
        '    "scikit-learn>=1.5.0" \\\n'
        '    "tqdm>=4.66.0" \\\n'
        '    "pyyaml>=6.0.2"'
    ))

    # ── Section: Write source code ──
    cells.append(make_markdown(
        "## 2. 写入项目源码\n"
        "\n"
        "将 `arena_ranker` 包的所有源文件写入 `/kaggle/working/arena_ranker/`。\n"
        "\n"
        "> **替代方案**：也可以把本仓库上传为 Kaggle Dataset，\n"
        "> 然后 `!pip install /kaggle/input/<your-dataset-slug>/` 来安装。"
    ))
    cells.append(make_code(build_write_files_code()))

    # ── Verify import ──
    cells.append(make_code(
        "import arena_ranker\n"
        "print('导入成功:', arena_ranker.__file__)"
    ))

    # ── Section: Paths & Config ──
    cells.append(make_markdown(
        "## 3. 配置\n"
        "\n"
        "设置竞赛数据路径和训练参数。\n"
        "\n"
        "⚠️ **请根据你的实际竞赛修改 `COMPETITION_SLUG`。**\n"
        "\n"
        "双 T4 会自动走分布式双卡训练；P100 保持单卡训练。\n"
        "T4 不支持 bf16，因此默认保持 `BF16 = False`。"
    ))
    cells.append(make_code(
        "import torch\n"
        "\n"
        "# ============================================================\n"
        "# 🔧 根据你的情况修改以下参数\n"
        "# ============================================================\n"
        'COMPETITION_SLUG = "llm-classification-finetuning"   # ← 改成你的竞赛 slug\n'
        'MODEL_NAME       = "Qwen/Qwen3-0.6B"                # 基座模型\n'
        "EPOCHS           = 3\n"
        "BATCH_SIZE       = 2        # per-device batch size\n"
        "GRAD_ACCUM_STEPS = 4        # T4 x2 时推荐 4；P100 单卡可改回 8\n"
        "MAX_LENGTH       = 1024     # T4/P100 16GB 推荐 1024\n"
        "LEARNING_RATE    = 2e-4\n"
        "USE_LORA         = True\n"
        "LOAD_IN_4BIT     = True     # QLoRA 4-bit 量化\n"
        "FP16             = True     # T4 Tensor Cores 建议开启\n"
        "BF16             = False    # T4 不支持 bf16，请保持关闭\n"
        "DDP_FIND_UNUSED_PARAMETERS = False\n"
        "LOCAL_FILES_ONLY = False\n"
        "# ============================================================\n"
        "\n"
        "NUM_PROCESSES = torch.cuda.device_count() if torch.cuda.is_available() else 1\n"
        'DATA_DIR    = f"/kaggle/input/{COMPETITION_SLUG}"\n'
        'OUTPUT_DIR  = "/kaggle/working/artifacts/default"\n'
        "\n"
        'device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"\n'
        'vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0\n'
        'effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS * max(NUM_PROCESSES, 1)\n'
        'print(f"设备: {device_name} ({vram_gb:.1f} GB)")\n'
        'print(f"GPU 数量: {NUM_PROCESSES}")\n'
        'print(f"有效 batch size: {effective_batch}")\n'
        'print(f"数据目录: {DATA_DIR}")\n'
        'print(f"输出目录: {OUTPUT_DIR}")\n'
    ))

    # ── Verify data ──
    cells.append(make_code(
        "import os\n"
        "\n"
        "data_files = os.listdir(DATA_DIR)\n"
        'print("竞赛数据文件:", data_files)\n'
        'assert "train.csv" in data_files, f"找不到 train.csv，请检查 COMPETITION_SLUG。当前目录: {DATA_DIR}"\n'
        'assert "test.csv"  in data_files, f"找不到 test.csv，请检查 COMPETITION_SLUG。当前目录: {DATA_DIR}"\n'
        'print("✅ 数据验证通过")\n'
    ))

    # ── Section: Training ──
    cells.append(make_markdown(
        "## 4. 训练\n"
        "\n"
        "使用 QLoRA 微调 `Qwen3-0.6B`，通过 HuggingFace Trainer 训练。\n"
        "\n"
        "| 参数 | T4 x2 | P100 x1 | 8GB 显存 | 说明 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `BATCH_SIZE` | 2 | 2 | 1 | per-device |\n"
        "| `GRAD_ACCUM_STEPS` | 4 | 8 | 16 | 有效 batch = `batch × grad_accum × GPU数` |\n"
        "| `MAX_LENGTH` | 1024 | 1024 | 512 | 输入序列最大 token 数 |\n"
        "| `EPOCHS` | 3 | 3 | 3 | 训练轮数 |\n"
        "| `LEARNING_RATE` | 2e-4 | 2e-4 | 2e-4 | AdamW 学习率 |\n"
        "| `LOAD_IN_4BIT` | True | True | True | 4-bit NF4 量化 (QLoRA) |\n"
        "| `FP16` | True | True | True | 启用混合精度 |\n"
        "| `BF16` | False | False | False | T4 / P100 都不要开启 |\n"
        "\n"
        "> **双卡训练说明**: notebook 会自动检测 GPU 数量。\n"
        "> 当检测到 `T4 x2` 时，会使用 `torch.distributed.run --nproc_per_node=2` 启动双卡训练。\n"
        "> 若想保持与单卡默认配置接近的有效 batch，推荐 `BATCH_SIZE=2, GRAD_ACCUM_STEPS=4`。"
    ))
    cells.append(make_code(
        "import os\n"
        "import shlex\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "train_command = [sys.executable]\n"
        "if NUM_PROCESSES > 1:\n"
        "    train_command.extend([\n"
        '        "-m", "torch.distributed.run", "--standalone",\n'
        '        "--nproc_per_node", str(NUM_PROCESSES),\n'
        '        "-m", "arena_ranker.train",\n'
        "    ])\n"
        "else:\n"
        '    train_command.extend(["-m", "arena_ranker.train"])\n'
        "\n"
        "train_command.extend([\n"
        '    "--data-dir", DATA_DIR,\n'
        '    "--output-dir", OUTPUT_DIR,\n'
        '    "--model-name", MODEL_NAME,\n'
        '    "--epochs", str(EPOCHS),\n'
        '    "--batch-size", str(BATCH_SIZE),\n'
        '    "--grad-accum-steps", str(GRAD_ACCUM_STEPS),\n'
        '    "--max-length", str(MAX_LENGTH),\n'
        '    "--learning-rate", str(LEARNING_RATE),\n'
        "])\n"
        "\n"
        "if not USE_LORA:\n"
        '    train_command.append("--disable-lora")\n'
        "if not LOAD_IN_4BIT:\n"
        '    train_command.append("--no-4bit")\n'
        "if FP16:\n"
        '    train_command.append("--fp16")\n'
        "else:\n"
        '    train_command.append("--no-fp16")\n'
        "if BF16:\n"
        '    train_command.append("--bf16")\n'
        "else:\n"
        '    train_command.append("--no-bf16")\n'
        "if DDP_FIND_UNUSED_PARAMETERS:\n"
        '    train_command.append("--ddp-find-unused-parameters")\n'
        "else:\n"
        '    train_command.append("--no-ddp-find-unused-parameters")\n'
        "if LOCAL_FILES_ONLY:\n"
        '    train_command.append("--local-files-only")\n'
        "\n"
        "env = os.environ.copy()\n"
        'env["PYTHONPATH"] = "/kaggle/working"\n'
        'print("训练命令:", " ".join(shlex.quote(part) for part in train_command))\n'
        "subprocess.run(train_command, env=env, check=True)\n"
    ))

    # ── Section: Inference ──
    cells.append(make_markdown(
        "## 5. 推理与生成提交文件\n"
        "\n"
        "加载训练好的 QLoRA adapter + 分类头，对 `test.csv` 推理生成 `submission.csv`。"
    ))
    cells.append(make_code(
        "import importlib, sys\n"
        "\n"
        'SUBMISSION_PATH = "/kaggle/working/submission.csv"\n'
        "\n"
        "sys.argv = [\n"
        '    "arena-predict",\n'
        '    "--checkpoint-dir", OUTPUT_DIR,\n'
        '    "--data-dir",       DATA_DIR,\n'
        '    "--output-path",    SUBMISSION_PATH,\n'
        '    "--batch-size",     "4",\n'
        "]\n"
        "\n"
        "from arena_ranker.predict import main as predict_main\n"
        "predict_main()\n"
    ))

    # ── Section: Review submission ──
    cells.append(make_markdown("## 6. 检查提交文件"))
    cells.append(make_code(
        "import pandas as pd\n"
        "\n"
        "sub = pd.read_csv(SUBMISSION_PATH)\n"
        'print(f"行数: {len(sub)}")\n'
        'print(f"列名: {list(sub.columns)}")\n'
        "print()\n"
        "print(sub.head(10))\n"
        "print()\n"
        'print("各列概率统计:")\n'
        "print(sub.describe())\n"
        "print()\n"
        "\n"
        "row_sums = sub[[\"winner_model_a\", \"winner_model_b\", \"winner_tie\"]].sum(axis=1)\n"
        'print(f"概率行和范围: [{row_sums.min():.6f}, {row_sums.max():.6f}]")\n'
        'print("✅ submission.csv 已生成:", SUBMISSION_PATH)\n'
    ))

    # ── Section: Offline Mode ──
    cells.append(make_markdown(
        "## 附录 A：离线模式（无需联网）\n"
        "\n"
        "如果 notebook 不能联网（例如最终提交时），需要提前将模型上传到 Kaggle。\n"
        "\n"
        "### 步骤\n"
        "\n"
        "1. **上传模型到 Kaggle**\n"
        "   - 在本地下载好 `Qwen/Qwen3-0.6B` 的完整文件\n"
        "   - 前往 [kaggle.com/models](https://kaggle.com/models) → New Model\n"
        "   - 上传模型文件夹（包含 config.json, model.safetensors 等）\n"
        "   - 或者使用 Kaggle Datasets 上传也可以\n"
        "\n"
        "2. **在 notebook 中添加模型数据集**\n"
        "   - 右侧 Add Input → 搜索你上传的模型\n"
        "\n"
        "3. **修改配置**\n"
        "   ```python\n"
        '   MODEL_NAME = "/kaggle/input/<model-dataset-slug>"  # 改为本地路径\n'
        "   ```\n"
        "\n"
        "4. **训练时加上 `--local-files-only`**\n"
        "   ```python\n"
        '   sys.argv.append("--local-files-only")\n'
        "   ```\n"
    ))

    # ── Section: Upload code as dataset ──
    cells.append(make_markdown(
        "## 附录 B：将代码上传为 Kaggle Dataset\n"
        "\n"
        "如果不想在 notebook 里内联代码，可以把仓库上传为 Kaggle Dataset：\n"
        "\n"
        "1. 打包源码：\n"
        "   ```bash\n"
        "   zip -r arena-ranker-code.zip src/ pyproject.toml README.md\n"
        "   ```\n"
        "\n"
        "2. 上传到 Kaggle Datasets\n"
        "\n"
        "3. 在 notebook 中安装：\n"
        "   ```python\n"
        '   !pip install /kaggle/input/arena-ranker-code/\n'
        "   ```\n"
        "\n"
        "4. 使用 CLI 命令：\n"
        "   ```python\n"
        "   !arena-train --data-dir /kaggle/input/<slug>/ --output-dir /kaggle/working/artifacts/default\n"
        "   !arena-predict --checkpoint-dir /kaggle/working/artifacts/default --data-dir /kaggle/input/<slug>/ --output-path /kaggle/working/submission.csv\n"
        "   ```\n"
    ))

    # ── Section: 8GB VRAM ──
    cells.append(make_markdown(
        "## 附录 C：显存不足时的参数调整\n"
        "\n"
        "如果遇到 OOM，优先按以下顺序调整：\n"
        "\n"
        "1. 降低 `MAX_LENGTH`（如 512）\n"
        "2. 降低 `BATCH_SIZE` 到 1\n"
        "3. 提高 `GRAD_ACCUM_STEPS`\n"
        "4. 确保 `LOAD_IN_4BIT = True`\n"
        "\n"
        "```python\n"
        "# 8GB 显存推荐参数\n"
        "BATCH_SIZE   = 1\n"
        "GRAD_ACCUM   = 16\n"
        "MAX_LENGTH   = 512\n"
        "LOAD_IN_4BIT = True\n"
        "```"
    ))

    return cells


def main() -> None:
    cells = build_cells()
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Notebook written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
