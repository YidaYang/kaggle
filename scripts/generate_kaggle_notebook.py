"""Generate a Kaggle-ready notebook for Arena Ranker training and inference."""

from __future__ import annotations

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
    import base64

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
        "本 notebook 在 Kaggle 提供的 **GPU 环境（T4 16GB / P100 16GB）** 中完成：\n"
        "\n"
        "1. 安装依赖\n"
        "2. 设置项目源码\n"
        "3. 训练 `Qwen/Qwen3-Embedding-0.6B` + LoRA 偏好分类器\n"
        "4. 推理并生成 `submission.csv`\n"
        "\n"
        "> **前置准备**\n"
        ">\n"
        "> | 项目 | 操作 |\n"
        "> |------|------|\n"
        "> | **竞赛数据** | Notebook 右侧 → Add Input → 搜索并添加竞赛数据集 |\n"
        "> | **GPU** | Settings → Accelerator → 选择 **GPU T4 ×2** 或 **GPU P100** |\n"
        "> | **联网** | Settings → Internet → **On**（用于下载 HuggingFace 模型）|\n"
        ">\n"
        "> 如果不想联网下载模型，请参考最后一节「离线模式」，提前把模型上传为 Kaggle Model。\n"
    ))

    # ── Section: Install deps ──
    cells.append(make_markdown(
        "## 1. 安装依赖\n"
        "\n"
        "Kaggle 环境已预装 PyTorch 和部分常用库，这里只需补装缺失的包。"
    ))
    cells.append(make_code(
        '!pip install -q "transformers>=4.55.0,<5" "peft>=0.17.0" "scikit-learn>=1.5.0" "tqdm>=4.66.0" "pyyaml>=6.0.2"'
    ))

    # ── Section: Write source code ──
    cells.append(make_markdown(
        "## 2. 写入项目源码\n"
        "\n"
        "将 `arena_ranker` 包的所有源文件写入 `/kaggle/working/arena_ranker/`，并添加到 `sys.path`。\n"
        "\n"
        "> **替代方案**：你也可以把本仓库上传为 Kaggle Dataset，然后直接\n"
        "> `!pip install /kaggle/input/<your-dataset-slug>/` 来安装，不需要这一步。"
    ))
    cells.append(make_code(build_write_files_code()))

    # ── Section: Verify import ──
    cells.append(make_code(
        "import arena_ranker\n"
        "print('导入成功:', arena_ranker.__file__)"
    ))

    # ── Section: Paths & Config ──
    cells.append(make_markdown(
        "## 3. 配置\n"
        "\n"
        "设置竞赛数据路径和训练超参数。\n"
        "\n"
        "⚠️ **请根据你的实际竞赛修改 `COMPETITION_SLUG`。**"
    ))
    cells.append(make_code(
        "import torch\n"
        "\n"
        "# ============================================================\n"
        "# 🔧 修改这里\n"
        "# ============================================================\n"
        'COMPETITION_SLUG = "llm-classification-finetuning"   # ← 改成你的竞赛 slug\n'
        'MODEL_NAME       = "Qwen/Qwen3-Embedding-0.6B"      # 基础模型\n'
        "EPOCHS           = 1\n"
        "BATCH_SIZE       = 2        # T4 16GB 可以用 2~4\n"
        "GRAD_ACCUM_STEPS = 4\n"
        "MAX_LENGTH       = 512\n"
        "LEARNING_RATE    = 2e-5\n"
        "USE_LORA         = True\n"
        "# ============================================================\n"
        "\n"
        'DATA_DIR    = f"/kaggle/input/{COMPETITION_SLUG}"\n'
        'OUTPUT_DIR  = "/kaggle/working/artifacts/default"\n'
        "\n"
        'device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"\n'
        'vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0\n'
        'print(f"设备: {device_name} ({vram_gb:.1f} GB)")\n'
        'print(f"数据目录: {DATA_DIR}")\n'
        'print(f"输出目录: {OUTPUT_DIR}")\n'
    ))

    # ── Verify data ──
    cells.append(make_code(
        "import os\n"
        "\n"
        'data_files = os.listdir(DATA_DIR)\n'
        'print("竞赛数据文件:", data_files)\n'
        'assert "train.csv" in data_files, f"找不到 train.csv，请检查 COMPETITION_SLUG 是否正确。当前数据目录: {DATA_DIR}"\n'
        'assert "test.csv"  in data_files, f"找不到 test.csv，请检查 COMPETITION_SLUG 是否正确。当前数据目录: {DATA_DIR}"\n'
        'print("✅ 数据验证通过")\n'
    ))

    # ── Section: Training ──
    cells.append(make_markdown(
        "## 4. 训练\n"
        "\n"
        "使用 LoRA 微调 Qwen Embedding 模型 + MLP 分类头。\n"
        "\n"
        "Kaggle T4 (16GB) 相比本地 8GB 显卡有更多余量，默认 `batch_size=2`，训练速度更快。\n"
        "\n"
        "| 参数 | 默认值 | 说明 |\n"
        "| --- | --- | --- |\n"
        "| `BATCH_SIZE` | 2 | T4 可用 2~4，P100 可用 2~4 |\n"
        "| `GRAD_ACCUM_STEPS` | 4 | 有效 batch = BATCH_SIZE × GRAD_ACCUM_STEPS |\n"
        "| `MAX_LENGTH` | 512 | 输入序列最大 token 数 |\n"
        "| `EPOCHS` | 1 | 训练轮数 |"
    ))
    cells.append(make_code(
        "import sys\n"
        "\n"
        "sys.argv = [\n"
        '    "arena-train",\n'
        '    "--data-dir",        DATA_DIR,\n'
        '    "--output-dir",      OUTPUT_DIR,\n'
        '    "--model-name",      MODEL_NAME,\n'
        '    "--epochs",          str(EPOCHS),\n'
        '    "--batch-size",      str(BATCH_SIZE),\n'
        '    "--grad-accum-steps", str(GRAD_ACCUM_STEPS),\n'
        '    "--max-length",      str(MAX_LENGTH),\n'
        "]\n"
        "\n"
        "if USE_LORA:\n"
        '    sys.argv.append("--use-lora")\n'
        "else:\n"
        '    sys.argv.append("--disable-lora")\n'
        "\n"
        "from arena_ranker.train import main as train_main\n"
        "train_main()\n"
    ))

    # ── Section: Inference ──
    cells.append(make_markdown(
        "## 5. 推理与生成提交文件\n"
        "\n"
        "加载训练好的 checkpoint，对 `test.csv` 进行推理，生成 `submission.csv`。"
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
        "## 附录：离线模式（无需联网）\n"
        "\n"
        "如果你的 notebook 不能联网（例如最终提交时），需要提前将模型上传到 Kaggle。\n"
        "\n"
        "### 步骤\n"
        "\n"
        "1. **上传模型到 Kaggle**\n"
        "   - 在本地下载好 `Qwen/Qwen3-Embedding-0.6B` 的完整文件\n"
        "   - 前往 [kaggle.com/models](https://kaggle.com/models) → New Model\n"
        "   - 上传模型文件夹\n"
        "   - 或者使用 Kaggle Datasets 上传也可以\n"
        "\n"
        "2. **在 notebook 中添加模型数据集**\n"
        "   - 右侧 Add Input → 搜索你上传的模型\n"
        "   - 模型路径一般为 `/kaggle/input/<model-dataset-slug>/`\n"
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
        "\n"
        "这样 notebook 就不需要联网下载模型了。"
    ))

    # ── Section: Upload code as dataset ──
    cells.append(make_markdown(
        "## 附录：将代码上传为 Kaggle Dataset（替代方案）\n"
        "\n"
        "如果你不想在 notebook 里内联所有代码，可以把整个仓库上传为 Kaggle Dataset：\n"
        "\n"
        "1. 在本地项目目录下，打包源码：\n"
        "   ```bash\n"
        "   # 确保在项目根目录\n"
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
        "4. 然后就可以直接使用命令行工具：\n"
        "   ```python\n"
        "   !arena-train --data-dir /kaggle/input/llm-classification-finetuning/ --output-dir /kaggle/working/artifacts/default\n"
        "   !arena-predict --checkpoint-dir /kaggle/working/artifacts/default --data-dir /kaggle/input/llm-classification-finetuning/ --output-path /kaggle/working/submission.csv\n"
        "   ```\n"
        "\n"
        "这种方式代码更整洁，且保持与本地仓库同步。"
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
