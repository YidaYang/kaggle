# Arena Ranker

基于 `Qwen` embedding encoder + 分类头的三分类方案，用于预测 ChatBot Arena 偏好标签：

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

当前实现已经支持：

- 基于 `Qwen/Qwen3-Embedding-0.6B` 的 LoRA 微调
- 8GB 显存下可运行的默认训练配置
- 训练过程的人性化日志输出
- checkpoint 恢复后生成 `submission.csv`

项目定位很直接：保持训练链路简单，不引入额外训练框架，在当前代码结构上完成可训练、可预测、可调参的最小实现。

## 当前默认配置

- 基础模型：`Qwen/Qwen3-Embedding-0.6B`
- 微调方式：默认开启 LoRA
- `transformers` 版本：`>=4.55.0,<5`
- 8GB 显存默认训练参数：
  - `batch_size=1`
  - `grad_accum_steps=8`
  - `max_length=512`
  - `gradient_checkpointing=true`

如果你不明确需要改什么，先直接跑默认配置。

## 目录结构

```text
.
├── pyproject.toml
├── README.md
├── train.csv
├── test.csv
├── sample_submission.csv
└── src/arena_ranker
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── hf.py
    ├── modeling.py
    ├── predict.py
    └── train.py
```

## 安装

```bash
cd "/media/starandhonor/Data/code/kaggle/llm-classification-finetuning"
uv sync
```

如果你需要代理下载模型，可以先导出代理环境变量：

```bash
export https_proxy="http://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"
export all_proxy="socks5://127.0.0.1:7890"
```

## 数据格式

训练集和测试集默认读取项目根目录下的 `train.csv`、`test.csv`。

代码会从原始表中读取这些字段：

- `id`
- `prompt`
- `response_a`
- `response_b`

训练集还要求存在以下标签列之一为 `1`：

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

文本字段支持以下形式：

- 普通字符串
- JSON 数组字符串
- Python list 字符串

数据处理逻辑会自动：

- 解析多轮对话字段
- 将多段文本按换行拼接
- 按 `text_max_chars` 截断
- 为训练集构建三分类标签

## 训练

直接启动默认训练：

```bash
uv run arena-train
```

默认会：

- 加载 `Qwen/Qwen3-Embedding-0.6B`
- 启用 LoRA
- 启用 gradient checkpointing
- 将产物保存到 `artifacts/default`

### 常用训练参数

切换模型：

```bash
uv run arena-train --model-name "Qwen/Qwen3-Embedding-4B"
```

调整训练轮数和 batch：

```bash
uv run arena-train --epochs 2 --batch-size 1 --grad-accum-steps 16
```

调整输入长度：

```bash
uv run arena-train --max-length 384
```

离线加载本地模型：

```bash
uv run arena-train --model-name "/abs/path/to/Qwen3-Embedding-0.6B" --local-files-only
```

### LoRA 微调

当前默认启用 LoRA。默认参数：

- `lora_r=16`
- `lora_alpha=32`
- `lora_dropout=0.05`
- `lora_bias=none`
- `lora_target_modules=q_proj,k_proj,v_proj,o_proj`

调整 LoRA 参数：

```bash
uv run arena-train --lora-r 32 --lora-alpha 64 --lora-dropout 0.1
```

关闭 LoRA，退回全参数训练：

```bash
uv run arena-train --disable-lora
```

不建议在 8GB 显存卡上直接关闭 LoRA。

### 8GB 显存建议

如果你使用的是 8GB 左右显存的消费级 GPU，例如 RTX 4060 Laptop，优先从这组参数开始：

```bash
uv run arena-train --batch-size 1 --grad-accum-steps 8 --max-length 512
```

如果依然显存不足，继续收紧：

```bash
uv run arena-train --batch-size 1 --grad-accum-steps 16 --max-length 384
```

如果你遇到 CUDA 内存碎片问题，可以加上：

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

### 训练日志

训练启动后会输出更易读的日志，例如：

```text
20:57:45 | INFO | 训练启动
20:57:45 | INFO | 设备: NVIDIA GeForce RTX 4060 Laptop GPU (7.6 GB)
20:57:45 | INFO | 数据集: train=51729, valid=5748, batch_size=1, grad_accum=8, epochs=1
20:57:45 | INFO | 模型: Qwen/Qwen3-Embedding-0.6B | max_length=512 | LoRA=on | gradient_checkpointing=on
20:57:45 | INFO | LoRA 配置: r=16, alpha=32, dropout=0.050, target_modules=q_proj,k_proj,v_proj,o_proj
20:57:45 | INFO | 优化步数: total=6467, warmup=646
20:57:45 | INFO | 输出目录: artifacts/default
```

每个 epoch 结束后还会输出：

- 平均训练损失
- 验证集准确率
- 验证集 log loss
- epoch 耗时
- 是否刷新最佳结果

### 训练产物

默认输出目录为 `artifacts/default`，会生成：

- `model.pt`
- `config.yaml`
- `metrics.json`
- `tokenizer/`

## 预测

基于训练好的 checkpoint 生成提交文件：

```bash
uv run arena-predict --checkpoint-dir "./artifacts/default"
```

输出文件默认写到：

```text
./artifacts/default/submission.csv
```

也可以手动指定输出路径：

```bash
uv run arena-predict \
  --checkpoint-dir "./artifacts/default" \
  --output-path "./artifacts/default/submission_custom.csv"
```

预测阶段也会打印启动摘要、设备信息、样本数和最终输出路径。

## 配置说明

配置定义在 [src/arena_ranker/config.py](./src/arena_ranker/config.py)。

主要分三部分：

- `DataConfig`：数据路径、文本截断长度、验证集比例
- `ModelConfig`：模型名、最大长度、LoRA 参数
- `TrainingConfig`：输出目录、学习率、batch、epoch、梯度累积、AMP、gradient checkpointing

如果你已经有自己的 YAML 配置，可以通过：

```bash
uv run arena-train --config "./your-config.yaml"
```

命令行参数会覆盖配置文件中的同名项。

## 常见问题

### 1. `Can't load the configuration of 'Qwen/Qwen3-Embedding-0.6B'`

优先检查：

1. 当前环境是否能访问 `huggingface.co`
2. 本地是否有同名目录 `Qwen/Qwen3-Embedding-0.6B`，但目录里缺少 `config.json`
3. 是否执行过 `uv sync`

### 2. `transformers` 版本不兼容

当前代码只验证了 `4.x`，不要放宽到 `5.x`。

标准做法：

```bash
uv sync
```

### 3. CUDA OOM

优先顺序：

1. 降低 `max_length`
2. 保持 `batch_size=1`
3. 提高 `grad_accum_steps`
4. 确保 LoRA 开启
5. 打开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 4. 训练速度比预期慢

这是当前实现的结构性结果，不是异常。

原因很直接：

- 每个样本会分别编码 `prompt`、`response_a`、`response_b`
- 为了适配 8GB 显存，默认启用了更保守的 batch 和 gradient checkpointing

如果你更关心速度而不是显存，可以尝试：

- 提高 `batch_size`
- 提高 `max_length` 前先确认显存
- 关闭 `gradient_checkpointing`

## 开发说明

当前实现刻意保持简单：

- 不引入 Trainer
- 不做多折训练
- 不做复杂特征工程
- 不做额外实验管理系统

这符合当前项目的 KISS 和 YAGNI 目标。如果你后续要继续扩展，建议优先做这些事情：

- 只保存 LoRA adapter 和分类头，减少 checkpoint 体积
- 增加验证集更细粒度的指标输出
- 增加显存和吞吐量日志
- 增加基础测试
