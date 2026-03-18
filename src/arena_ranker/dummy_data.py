"""
假数据生成模块 — 生成 dummy train.csv 和 test.csv 用于本地测试。

使用方法：
  uv run arena-dummy-data                    # 默认 100 条训练 + 20 条测试
  uv run arena-dummy-data --n-train 50 --n-test 10
  uv run arena-dummy-data --output-dir ./data

也可以在 Python 中直接调用：
  from arena_ranker.dummy_data import generate_dummy_data
  generate_dummy_data(n_train=100, n_test=20)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

# 假提问列表
_PROMPTS = [
    "What is machine learning? Explain it in simple terms.",
    "Write a Python function that sorts a list using merge sort.",
    "Explain the theory of relativity to a 10-year-old.",
    "How does photosynthesis work? Please be detailed.",
    "What are the main differences between TCP and UDP?",
    "Summarize the plot of Romeo and Juliet.",
    "Write a poem about the ocean at sunset.",
    "Explain quantum entanglement in simple terms.",
    "What is the capital of France and what is it known for?",
    "How do neural networks learn? Explain backpropagation.",
    "Compare Python and Rust for systems programming.",
    "What are the benefits of meditation?",
    "Describe how a CPU executes instructions.",
    "Write a SQL query to find duplicate emails in a table.",
    "Explain the difference between supervised and unsupervised learning.",
]

_RESPONSE_TEMPLATES_A = [
    "That's a great question! Let me explain. {topic} is fundamentally about "
    "understanding patterns in data. The key insight is that we can use mathematical "
    "models to approximate complex relationships. Here's a more detailed breakdown: "
    "First, we need to collect relevant data. Then, we preprocess it to remove noise. "
    "Finally, we train our model and evaluate its performance.",
    "Sure! Here's my take on this. {topic} involves several important concepts. "
    "The most fundamental one is that learning happens through iterative optimization. "
    "We start with a random guess and gradually improve it based on feedback from the data.",
    "Great question. In simple terms, {topic} is like teaching a computer to recognize "
    "patterns. Imagine showing a child thousands of pictures of cats and dogs - eventually "
    "they learn to tell them apart. That's essentially what happens in this process.",
]

_RESPONSE_TEMPLATES_B = [
    "Thanks for asking! {topic} is actually simpler than most people think. "
    "At its core, it's about finding the best function that maps inputs to outputs. "
    "We do this by minimizing a loss function using gradient-based optimization. "
    "The beauty of this approach is its generality - it works across many domains.",
    "Let me break this down. {topic} can be understood through a simple analogy. "
    "Think of it as a recipe - you have ingredients (data), instructions (algorithm), "
    "and a final dish (predictions). The quality of each ingredient matters, but so does "
    "how you combine them.",
    "I'd be happy to explain. {topic} is a fascinating area. The basic idea is "
    "that we can use statistics and computation to make predictions about the world. "
    "The key challenge is generalization - making sure our model works on new, unseen data.",
]


def generate_dummy_data(
    output_dir: str = ".",
    n_train: int = 100,
    n_test: int = 20,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    生成用于本地测试的假数据。

    标签分布大致为：A胜 40%, B胜 40%, 平局 20%（模拟真实数据分布）。

    Returns:
        (train_path, test_path)
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 训练集 ----
    train_rows = []
    for i in range(n_train):
        prompt = _PROMPTS[i % len(_PROMPTS)]
        topic = prompt.split("?")[0].split(".")[0].strip()

        resp_a = random.choice(_RESPONSE_TEMPLATES_A).format(topic=topic)
        resp_b = random.choice(_RESPONSE_TEMPLATES_B).format(topic=topic)

        label = rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        train_rows.append({
            "id": 100000 + i,
            "model_a": f"model_x_{rng.randint(1, 5)}",
            "model_b": f"model_y_{rng.randint(1, 5)}",
            "prompt": prompt,
            "response_a": resp_a,
            "response_b": resp_b,
            "winner_model_a": 1 if label == 0 else 0,
            "winner_model_b": 1 if label == 1 else 0,
            "winner_tie": 1 if label == 2 else 0,
        })

    train_path = out / "train.csv"
    pd.DataFrame(train_rows).to_csv(train_path, index=False)

    # ---- 测试集 ----
    test_rows = []
    for i in range(n_test):
        prompt = _PROMPTS[i % len(_PROMPTS)]
        topic = prompt.split("?")[0].split(".")[0].strip()

        resp_a = random.choice(_RESPONSE_TEMPLATES_A).format(topic=topic)
        resp_b = random.choice(_RESPONSE_TEMPLATES_B).format(topic=topic)

        test_rows.append({
            "id": 200000 + i,
            "prompt": prompt,
            "response_a": resp_a,
            "response_b": resp_b,
        })

    test_path = out / "test.csv"
    pd.DataFrame(test_rows).to_csv(test_path, index=False)

    print(f"已生成训练集: {train_path} ({n_train} 条)")
    print(f"已生成测试集: {test_path} ({n_test} 条)")
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成假数据用于本地测试。")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="输出目录")
    parser.add_argument("--n-train", type=int, default=100,
                        help="训练集行数")
    parser.add_argument("--n-test", type=int, default=20,
                        help="测试集行数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dummy_data(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
