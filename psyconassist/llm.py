"""LLM 客户端（DeepSeek，OpenAI 兼容）。供主题抽取 / 风险复核 / 草稿合成使用。"""
from __future__ import annotations

import json

from openai import OpenAI

from . import config

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    return _client


def available() -> bool:
    return bool(config.LLM_API_KEY)


def chat_json(system: str, user: str, temperature: float = 0.2) -> dict | list | None:
    """请求严格 JSON 输出，解析失败返回 None。"""
    if not available():
        return None
    try:
        resp = get_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or ""
        return json.loads(text)
    except Exception:
        return None


def chat_text(system: str, user: str, temperature: float = 0.3) -> str | None:
    if not available():
        return None
    try:
        resp = get_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception:
        return None
