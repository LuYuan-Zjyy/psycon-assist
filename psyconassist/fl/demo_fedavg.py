"""跑一遍：人群共享 vs 仅本机 vs 联邦（合成受试者，可重复）。

用法：python -m psyconassist.fl.demo_fedavg

关注点不是「联邦一定更好」，而是：**在个体样本稀疏时，联邦能否接近集中训练**。
"""
from __future__ import annotations

import numpy as np

from .fedavg import fedavg, fine_tune, mae, train_linear


def make_client(seed: int, n: int = 240):
    """一个受试者：有自己的静态基线 b_i 与惯性 alpha_i。"""
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 0.35)
    alpha = rng.uniform(0.35, 0.9)
    v = b + rng.normal(0, 0.2)
    rows, ys = [], []
    for _ in range(n):
        nxt = b + alpha * (v - b) + rng.normal(0, 0.15)
        rows.append([v, 0.4 * v + rng.normal(0, 0.3),
                     0.6 * v + rng.normal(0, 0.4),
                     0.3 * v + rng.normal(0, 0.3),
                     0.5 * v + rng.normal(0, 0.5)])
        ys.append(nxt)
        v = nxt
    return b, alpha, np.nan_to_num(np.array(rows)), np.array(ys)


def evaluate(n_train: int, seeds=range(6)):
    clients = [make_client(seed=i) for i in seeds]
    splits = []
    for _, _, X, y in clients:
        k = int(len(X) * 0.7)
        splits.append((X[:min(n_train, k)], y[:min(n_train, k)], X[k:], y[k:]))

    # 人群共享（集中数据，仅作上界参考，隐私不可接受）
    Xall = np.vstack([s[0] for s in splits])
    yall = np.hstack([s[1] for s in splits])
    head_g = train_linear(Xall, yall)
    mae_g = float(np.mean([mae(head_g, s[2], s[3]) for s in splits]))

    # 仅本机
    heads_l = [train_linear(s[0], s[1]) for s in splits]
    mae_l = float(np.mean([mae(h, s[2], s[3]) for h, s in zip(heads_l, splits)]))

    # FedAvg（只交换 6 个参数）
    head_f = fedavg(heads_l, [len(s[0]) for s in splits])
    mae_f = float(np.mean([mae(head_f, s[2], s[3]) for s in splits]))

    # FedAvg + 本机微调（个性化）
    heads_p = [fine_tune(head_f, s[0], s[1], epochs=40) for s in splits]
    mae_p = float(np.mean([mae(h, s[2], s[3]) for h, s in zip(heads_p, splits)]))
    return mae_g, mae_l, mae_f, mae_p


def main() -> None:
    print("合成 6 名受试者；特征 5 维 + 偏置 = 每个客户端只交换 6 个浮点")
    print()
    print("每客户端训练样本   集中训练(上界)   仅本机    FedAvg   FedAvg+本机微调")
    for n in (24, 48, 96, 168):
        g, l, f, p = evaluate(n)
        print("  %3d             %.4f        %.4f    %.4f    %.4f" % (n, g, l, f, p))
    print()
    print("读法：样本很少时，仅本机容易过拟合、FedAvg 更稳；样本充足时本地已够用，")
    print("      联邦的价值退化为「冷启动先验」。真实多机构数据下需重跑此表。")


if __name__ == "__main__":
    main()
