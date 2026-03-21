# Arena Ranker

基于 `Qwen` embedding encoder + 分类头的三分类方案，用于预测 ChatBot Arena 偏好标签：

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

当前实现已经支持：

- 基于 `Qwen/Qwen3-Embedding-0.6B` 的 LoRA 微调
- 8GB 显存下可运行的默认训练配置
- 训练过程的人性化日志输出
- 可选上传训练指标到 `SwanLab`
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

如果你要使用 `SwanLab`，同步依赖后再登录：

```bash
uv sync
uv run swanlab login
```

## 数据格式

训练集和测试集默认读取项目根目录下的 `train.csv`、`test.csv`。

### 训练集输入列

代码会从训练集读取这些字段：

- `id`
- `prompt`
- `response_a`
- `response_b`
- `winner_model_a`
- `winner_model_b`
- `winner_tie`

其中：

- `id`：样本唯一标识
- `prompt`：用户问题或上下文
- `response_a`：候选回答 A
- `response_b`：候选回答 B
- `winner_model_a` / `winner_model_b` / `winner_tie`：三分类 one-hot 标签，且每行必须恰好有一个值为 `1`

### 测试集输入列

代码会从测试集读取这些字段：

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

### 字段内容示例

单轮字符串形式：

```csv
id,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie
1,"What is LoRA?","LoRA is a parameter-efficient fine-tuning method.","LoRA is a new optimizer.",1,0,0
```

多轮对话数组形式：

```csv
id,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie
2,"[""Hello"",""Explain gradient checkpointing.""]","[""It reduces activation memory during training.""]","[""It only speeds up inference.""]",1,0,0
```

代码会将数组内容解析后按换行拼接，例如：

```text
["Hello", "Explain gradient checkpointing."]
```

会被处理成：

```text
Hello
Explain gradient checkpointing.
```

### 训练输入与标签映射

训练时，每条样本会被整理成这三个文本输入：

- `prompt_text`
- `response_a_text`
- `response_b_text`

标签映射关系固定为：

- `winner_model_a -> 0`
- `winner_model_b -> 1`
- `winner_tie -> 2`

也就是说，模型最终学习的是一个三分类问题，而不是生成式打分任务。

### 预测输入与输出

预测阶段输入来自 `test.csv`，只需要：

- `id`
- `prompt`
- `response_a`
- `response_b`

模型输出为每个类别的概率，最终写入 `submission.csv`，列结构如下：

- `id`
- `winner_model_a`
- `winner_model_b`
- `winner_tie`

输出示例：

```csv
id,winner_model_a,winner_model_b,winner_tie
1,0.7321,0.2015,0.0664
2,0.1250,0.7310,0.1440
```

三列概率之和约等于 `1.0`。

当前实现会在写出 `submission.csv` 前再次做一次裁剪和归一化，保证最终输出概率严格落在 `(0, 1)` 区间，并保持每行概率和为 `1`。

## 模型架构

当前模型不是生成式微调，而是一个标准的三分类判别模型。整体结构可以概括为：

```text
prompt ------> encoder ------
                            |
response_a --> encoder ---- |--> 特征拼接 --> MLP classifier --> 3-class logits
                            |
response_b --> encoder ------
```

### 1. 输入编码

每条样本包含三段文本：

- `prompt`
- `response_a`
- `response_b`

这三段文本会分别经过同一个 `Qwen` embedding encoder 编码，共享权重，不是三套独立模型。

encoder 输出 `last_hidden_state` 后，代码使用 `masked mean pooling` 得到三个句向量：

- `prompt_emb`
- `response_a_emb`
- `response_b_emb`

对应实现见：

- [src/arena_ranker/modeling.py](./src/arena_ranker/modeling.py)
- [src/arena_ranker/data.py](./src/arena_ranker/data.py)

### 2. 特征构造

当前分类器输入不是只拼接三个 embedding，而是构造了 6 组特征：

- `prompt_emb`
- `response_a_emb`
- `response_b_emb`
- `response_a_emb - response_b_emb`
- `response_a_emb - prompt_emb`
- `response_b_emb - prompt_emb`

最终特征维度为：

```text
classifier_input = hidden_size * 6
```

这种设计的目的很直接：

- 保留 `prompt`、`response_a`、`response_b` 的绝对表示
- 显式引入 `A vs B`、`A vs prompt`、`B vs prompt` 的相对差异

这比只拼接三段向量更适合当前偏好比较任务。

### 3. 分类头

拼接后的特征会进入一个两层 MLP：

```text
Linear(hidden_size * 6, hidden_size * 2)
GELU
Dropout
Linear(hidden_size * 2, 3)
```

输出为 3 维 logits，对应：

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

训练时使用 `CrossEntropyLoss`。

### 4. LoRA 微调位置

当前默认开启 LoRA，LoRA 注入在 encoder 注意力层的这些模块上：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

默认参数：

- `r=16`
- `alpha=32`
- `dropout=0.05`

也就是说，当前训练策略是：

- encoder 主体参数大部分冻结为预训练权重
- 只训练 LoRA adapter 和上层分类头

这也是它能在 8GB 显存下运行的关键原因之一。

### 5. 训练与推理流程

训练阶段：

1. 分别编码 `prompt`、`response_a`、`response_b`
2. 做 mean pooling 得到三个向量
3. 拼接 6 组特征
4. 通过 MLP 得到三分类 logits
5. 用 `CrossEntropyLoss` 计算损失

推理阶段：

1. 走同样的前向流程
2. 对 logits 做 `softmax`
3. 输出三列概率到 `submission.csv`

### 6. 当前架构的特点

优点：

- 结构简单，容易训练和调试
- 明确适配 Arena 偏好比较任务
- 比全参数微调更节省显存

限制：

- 每个样本要对三段文本各跑一次 encoder，速度不会太快
- 当前只用了 pooling + MLP，没有更复杂的交叉注意力比较模块
- 架构偏向稳健基线，不是专门追求 SOTA 的复杂方案

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

### 三种训练模式

当前项目明确支持三种训练模式：

| 模式 | 命令 | encoder 参数 | LoRA | 适用场景 |
| --- | --- | --- | --- | --- |
| LoRA 微调 | `uv run arena-train` | 冻结主干，仅训练 adapter | 开启 | 默认推荐，兼顾效果与显存 |
| 冻结 encoder，仅训练分类头 | `uv run arena-train --classifier-only` | 全冻结 | 关闭 | 把 Qwen 当固定特征提取器，训练成本最低 |
| 全参数微调 | `uv run arena-train --disable-lora --disable-freeze-encoder` | 全量训练 | 关闭 | 显存充足，且希望整体继续适配任务 |

推荐顺序：

1. 先尝试 LoRA 微调
2. 如果你只想快速验证分类头是否有效，用 `classifier-only`
3. 只有在显存、训练时间都充足时再尝试全参数微调

训练日志启动时也会明确打印当前模式，例如：

```text
21:10:00 | INFO | 训练模式: lora
```

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

### 冻结 encoder，只训练分类头

如果你想冻结 `Qwen/Qwen3-Embedding-0.6B`，只训练上层分类头，可以直接运行：

```bash
uv run arena-train --classifier-only
```

这个模式会自动做三件事：

- 冻结 encoder 全部参数
- 关闭 LoRA
- 关闭 gradient checkpointing

如果你只想显式冻结 encoder，也可以这样写：

```bash
uv run arena-train --freeze-encoder --disable-lora --disable-gradient-checkpointing
```

这适合你把 `Qwen/Qwen3-Embedding-0.6B` 作为固定特征提取器，仅训练分类层的场景。

### 全参数微调

如果你要让 encoder 和分类头一起训练，可以显式关闭 LoRA：

```bash
uv run arena-train --disable-lora --disable-freeze-encoder
```

这个模式下：

- encoder 参数参与训练
- 分类头参数参与训练
- 不再使用 LoRA adapter

这通常需要更多显存，也更容易训练更慢或不稳定。对 8GB 显存卡不作为默认推荐。

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

### SwanLab 上传

项目现在支持把训练指标上传到 `SwanLab`。默认关闭，只有显式开启时才会初始化，避免影响原有训练链路。

最小用法：

```bash
uv run arena-train --classifier-only --enable-swanlab
```

指定项目名和实验名：

```bash
uv run arena-train \
  --classifier-only \
  --enable-swanlab \
  --swanlab-project "arena-ranker" \
  --swanlab-experiment-name "classifier-only-baseline"
```

如果你需要上传到指定工作空间，也可以补充：

```bash
uv run arena-train \
  --enable-swanlab \
  --swanlab-workspace "your-workspace"
```

训练过程中会上传这些指标：

- `train/loss`
- `train/avg_loss`
- `train/lr`
- `epoch/train_loss`
- `epoch/valid_accuracy`
- `epoch/valid_log_loss`
- `best/accuracy`
- `best/log_loss`

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

主要分四部分：

- `DataConfig`：数据路径、文本截断长度、验证集比例
- `ModelConfig`：模型名、最大长度、LoRA 参数
- `TrainingConfig`：输出目录、学习率、batch、epoch、梯度累积、AMP、gradient checkpointing
- `SwanlabConfig`：是否启用上传、项目名、实验名、workspace、运行模式

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
