# Arena Ranker

基于 `Qwen3.5-0.8B` + QLoRA 的三分类微调方案，用于预测 ChatBot Arena 偏好标签：

- `winner_model_a`（A 胜）
- `winner_model_b`（B 胜）
- `winner_tie`（平局）

## 核心架构

```text
prompt + response_a + response_b
        ↓
  apply_chat_template (system + user)
        ↓
  Qwen3.5-0.8B (4-bit 量化)
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

- 基座模型：`Qwen/Qwen3.5-0.8B`
- 量化：4-bit NF4 + double quantization
- LoRA：r=32, alpha=64, 覆盖 attention + FFN 全部投影层
- 分类头 (`score`) 通过 `modules_to_save` 全量训练
- 训练框架：HuggingFace Trainer
- 评估指标：Log Loss + Accuracy（概率裁剪）

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
- 模型：`Qwen/Qwen3.5-0.8B`
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

T4 16GB:
```bash
uv run arena-train --batch-size 2 --max-length 1024
```

8GB 显存:
```bash
uv run arena-train --batch-size 1 --grad-accum-steps 16 --max-length 512
```

## Kaggle / Colab 使用

在 Notebook 中可以直接导入使用：

```python
from arena_ranker.config import AppConfig
from arena_ranker.data import load_and_preprocess, build_dataset, split_train_valid, compute_metrics
from arena_ranker.hf import load_tokenizer, load_model
from arena_ranker.dummy_data import generate_dummy_data
```
