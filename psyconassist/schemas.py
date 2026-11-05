"""轻量契约校验：保证每阶段产物满足不变量，不符合即报错（不静默）。"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from .core.errors import ContractError

__all__ = ["ContractError", "require", "check_times_ascending", "check_within", "check_covers"]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def check_times_ascending(items: Iterable, get_t: Callable[[Any], float]) -> None:
    """时间序列必须严格递增。"""
    prev = None
    for it in items:
        t = get_t(it)
        if prev is not None and t <= prev:
            raise ContractError(f"时间未严格递增：{t} <= {prev}")
        prev = t


def check_within(
    items: Iterable,
    get_t0: Callable[[Any], float],
    get_t1: Callable[[Any], float],
    duration: float,
) -> None:
    for it in items:
        t0, t1 = get_t0(it), get_t1(it)
        if t0 < 0 or t1 > duration + 1e-6:
            raise ContractError(f"时间越界 [{t0}, {t1}) 超出 [0, {duration}]")
        if t1 < t0:
            raise ContractError(f"区间非法 [{t0}, {t1})")


def check_covers(
    items: list,
    get_t0: Callable[[Any], float],
    get_t1: Callable[[Any], float],
    duration: float,
    tol: float = 0.5,
) -> None:
    """chunk 首尾相接，覆盖 [0, duration]，不留空洞。"""
    if not items:
        raise ContractError("空 chunk 列表")
    items = sorted(items, key=get_t0)
    require(abs(get_t0(items[0])) <= tol, f"首块未从 0 开始: {get_t0(items[0])}")
    for a, b in zip(items, items[1:]):
        require(
            abs(get_t1(a) - get_t0(b)) <= tol,
            f"chunk 之间有空洞: [{get_t1(a)}, {get_t0(b)})",
        )
    require(
        abs(get_t1(items[-1]) - duration) <= tol,
        f"末块未到 duration: {get_t1(items[-1])} vs {duration}",
    )
