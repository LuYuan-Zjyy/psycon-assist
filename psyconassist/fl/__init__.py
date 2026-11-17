"""联邦学习原型：个体基线 + 个性化适配器，只交换低维权重。"""
from .fedavg import fedavg, train_linear, predict, mae

__all__ = ["fedavg", "train_linear", "predict", "mae"]
