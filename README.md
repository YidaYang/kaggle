# Arena Ranker

基于 `Qwen3-0.6B` + QLoRA 的三分类微调方案，用于预测 ChatBot Arena 偏好标签：

- `winner_model_a`（A 胜）
- `winner_model_b`（B 胜）
- `winner_tie`（平局）

## 核心架构

```text
prompt + response_a + response_b
        ↓
  apply_chat_template (system + user)
        ↓
  Qwen3-0.6B (4-bit 量化)
        ↓
  AutoModelForSequenceClassification
        ↓
  score (分类头, 3-class logits)
```

与 embedding + MLP 方案不同，本方案：

- 使用 **生成式模型** 的分类模式（`AutoModelForSequenceClassification`）
- 输入通过 **chat template** 构建，模型能理解对话结构
- 使用 **QLoRA** (4-bit NF4 量化 + LoRA) 降低显存占用
- 使用 HuggingFace **Trainer** 训练

## 主要特点

- 基座模型：`Qwen/Qwen3-0.6B`
- 量化：4-bit NF4 + double quantization
- LoRA：r=32, alpha=64, 覆盖 attention + FFN 全部投影层
- 分类头 (`score`) 通过 `modules_to_save` 全量训练
- 训练框架：HuggingFace Trainer
- 评估指标：Log Loss + Accuracy（概率裁剪）

## 在 Kaggle GPU Notebook 中运行

本项目提供了开箱即用的 Kaggle Notebook：[`kaggle_notebook.ipynb`](./kaggle_notebook.ipynb)。

### 快速上手

1. **上传 Notebook**：将 `kaggle_notebook.ipynb` 上传到 Kaggle（New Notebook → File → Import Notebook）
2. **添加竞赛数据**：右侧 Add Input → 搜索并添加竞赛数据集
3. **开启 GPU**：Settings → Accelerator → **GPU T4 ×2** 或 **GPU P100**
4. **开启联网**：Settings → Internet → **On**（首次运行需从 HuggingFace 下载模型）
5. **修改竞赛 slug**：在 notebook 的「配置」单元格中修改 `COMPETITION_SLUG`
6. **Run All**

### Kaggle GPU 推荐参数

| 参数 | 8GB 显存 | P100 x1 | T4 x2 |
| --- | --- | --- | --- |
| `batch_size` | 1 | 2 | 2 |
| `grad_accum_steps` | 16 | 8 | 4 |
| `max_length` | 512 | 1024 | 1024 |
| `load_in_4bit` | True | True | False |
| `fp16` | True | True | True |
| `bf16` | False | False | False |

> **双卡 T4 说明**
>
> `kaggle_notebook.ipynb` 会在检测到两张 GPU 时自动使用
> `torch.distributed.run --nproc_per_node=2` 启动双卡训练。
> 若想保持与单卡默认配置接近的有效 batch，推荐 `batch_size=2, grad_accum_steps=4`。
> 当前 Kaggle 环境下，双卡多进程 + bitsandbytes 4-bit 容易在模型加载阶段卡住，
> 因此 notebook 会在双卡训练时自动关闭 `LOAD_IN_4BIT`，改用 LoRA + FP16 + DDP。
> notebook 还会先把远程模型预下载到 `/kaggle/working`，再切换到本地只读加载，
> 以减少双进程并发拉取模型导致的卡顿。
>
> 如果你在双 T4 环境里手动把进程数改回 1，请同时只暴露一张 GPU
> （如 `CUDA_VISIBLE_DEVICES=0`），否则 `Trainer` 会退回 `DataParallel`，
> 而 bitsandbytes 4-bit / QLoRA 与它不兼容。
>
> **P100 注意事项**
>
> P100 (Pascal, compute capability 6.0) 没有 Tensor Core，FP16 矩阵运算比 T4 慢，
> 但仍然支持 bitsandbytes 4-bit NF4 量化和 FP16 混合精度。
> 对于 0.6B 参数模型，VRAM 参数与 T4 相同，主要差异体现在训练速度上。
> **不支持 bf16**，请保持 `bf16=False`（已是默认值）。

### 离线模式

将模型提前上传到 Kaggle Models/Datasets，在 notebook 中修改：

```python
MODEL_NAME = "/kaggle/input/<model-dataset-slug>"
```

训练命令里加上 `--local-files-only`。

### 替代方案：将代码上传为 Kaggle Dataset

```bash
zip -r arena-ranker-code.zip src/ pyproject.toml README.md
# 上传到 Kaggle Datasets 后，在 notebook 中：
# !pip install /kaggle/input/arena-ranker-code/
```

### Notebook 维护

Notebook 由 `scripts/generate_kaggle_notebook.py` 从源码自动生成。修改源码后运行：

```bash
python3 scripts/generate_kaggle_notebook.py
```

## 安装

```bash
uv sync
```

## 快速开始

### 1. 生成假数据（本地测试用）

```bash
uv run arena-dummy-data
```

会在当前目录生成 `train.csv` (100 条) 和 `test.csv` (20 条)。

### 2. 训练

```bash
uv run arena-train
```

默认配置：
- 模型：`Qwen/Qwen3-0.6B`
- QLoRA 4-bit 量化
- LoRA r=32, alpha=64
- 学习率 2e-4, cosine scheduler
- batch_size=2, grad_accum=8
- 3 个 epoch

### 3. 预测

```bash
uv run arena-predict --checkpoint-dir "./artifacts/default"
```

输出 `submission.csv` 到 checkpoint 目录。

## 常用参数

```bash
# 调整 batch size 和 epoch
uv run arena-train --batch-size 1 --grad-accum-steps 16 --epochs 5

# 使用本地模型
uv run arena-train --model-name "/path/to/local/model" --local-files-only

# 调整 LoRA
uv run arena-train --lora-r 16 --lora-alpha 32

# 关闭 LoRA
uv run arena-train --disable-lora --no-4bit

# 调整最大长度
uv run arena-train --max-length 2048

# 使用 YAML 配置
uv run arena-train --config my_config.yaml
```

## 目录结构

```text
.
├── pyproject.toml
├── README.md
├── sample_submission.csv
└── src/arena_ranker/
    ├── __init__.py
    ├── config.py        # 配置定义
    ├── data.py          # 数据加载、tokenization、metrics
    ├── hf.py            # 模型和 tokenizer 加载 (QLoRA)
    ├── modeling.py       # 占位 (模型定义已移至 hf.py)
    ├── train.py         # 训练脚本 (Trainer)
    ├── predict.py       # 推理脚本
    └── dummy_data.py    # 假数据生成
```

## LoRA 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `r` | 32 | LoRA 秩 |
| `lora_alpha` | 64 | LoRA 缩放因子 |
| `target_modules` | q/k/v/o/gate/up/down_proj | 覆盖注意力层和 FFN |
| `lora_dropout` | 0.05 | LoRA dropout |
| `modules_to_save` | `["score"]` | 分类头全量训练并保存 |
| `task_type` | `SEQ_CLS` | 序列分类任务 |

## 显存建议

T4 x2:
```bash
torchrun --nproc_per_node=2 -m arena_ranker.train --batch-size 2 --grad-accum-steps 4 --max-length 1024
```

P100 16GB:
```bash
uv run arena-train --batch-size 2 --grad-accum-steps 8 --max-length 1024
```

8GB 显存:
```bash
uv run arena-train --batch-size 1 --grad-accum-steps 16 --max-length 512
```

## Kaggle / Colab 使用

推荐直接使用 [`kaggle_notebook.ipynb`](./kaggle_notebook.ipynb)，详见上方「在 Kaggle GPU Notebook 中运行」章节。

也可以在自己的 Notebook 中导入使用：

```python
from arena_ranker.config import AppConfig
from arena_ranker.data import load_and_preprocess, build_dataset, split_train_valid, compute_metrics
from arena_ranker.hf import load_tokenizer, load_model
from arena_ranker.dummy_data import generate_dummy_data
```
