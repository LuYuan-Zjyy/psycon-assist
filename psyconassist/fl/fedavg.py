"""FedAvg 与小规模个性化适配器（纯 numpy，CPU 可跑）。

隐私边界：原始视频/音频/AU 序列不出本机；离开设备的只有
  - 每个受试者的 6 个标量（线性头权重 w[5] + 偏置 b）
  - 以及个体基线 4 个标量（v_mean, v_sd, a_mean, a_sd）
这是本项目「个性化联邦」最小可行的交换物。
"""
from __future__ import annotations

import numpy as np


def train_linear(X, y, epochs: int = 300, lr: float = 0.05, l2: float = 1e-3) -> dict:
    """线性回归头：y ~ X w + b。返回 {w, b}。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        err = X @ w + b - y
        w -= lr * (X.T @ err / n + l2 * w)
        b -= lr * float(err.mean())
    return {"w": w, "b": b}


def predict(head: dict, X) -> np.ndarray:
    return np.asarray(X, dtype=float) @ head["w"] + head["b"]


def mae(head: dict, X, y) -> float:
    return float(np.mean(np.abs(predict(head, X) - np.asarray(y, dtype=float))))


def fedavg(heads: list[dict], counts: list[int]) -> dict:
    """按样本数加权平均各客户端的头。"""
    total = float(sum(counts))
    w = sum(h["w"] * (c / total) for h, c in zip(heads, counts))
    b = sum(h["b"] * (c / total) for h, c in zip(heads, counts))
    return {"w": w, "b": b}


def fine_tune(head: dict, X, y, epochs: int = 30, lr: float = 0.02) -> dict:
    """用本机数据在 FedAvg 结果上做少量个性化微调（数据不出本机）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = head["w"].copy()
    b = float(head["b"])
    n = X.shape[0]
    for _ in range(epochs):
        err = X @ w + b - y
        w -= lr * (X.T @ err / n)
        b -= lr * float(err.mean())
    return {"w": w, "b": b}
