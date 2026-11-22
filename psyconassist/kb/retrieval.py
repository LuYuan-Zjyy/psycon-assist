"""轻量知识库检索：字符二元组 + 标签匹配，无需向量库。

条目字段：id / title / tags / content / source / use
"""
from __future__ import annotations

import json
import re
from pathlib import Path

KB_PATH = Path(__file__).with_name("seed_kb.json")
_CACHE: list[dict] | None = None


def get_all() -> list[dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(KB_PATH.read_text(encoding="utf-8"))
    return _CACHE


def _tokens(s: str) -> set[str]:
    s = (s or "").lower()
    toks = set(re.findall(r"[a-z0-9]+", s))
    zh = re.findall(r"[\u4e00-\u9fff]", s)
    toks.update(zh)
    toks.update("".join(zh[i:i + 2]) for i in range(len(zh) - 1))
    return toks


def search(query: str, k: int = 4) -> list[dict]:
    q = _tokens(query)
    if not q:
        return []
    scored: list[tuple[float, dict]] = []
    for e in get_all():
        text = e["title"] + " " + " ".join(e.get("tags", [])) + " " + e["content"]
        t = _tokens(text)
        inter = q & t
        if not inter:
            continue
        score = len(inter) / (1 + len(t) ** 0.5)
        for tag in e.get("tags", []):
            if tag.lower() in (query or "").lower():
                score += 0.6
        scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:k]]


def cite(entries: list[dict]) -> str:
    return "、".join(e["id"] for e in entries) if entries else ""
