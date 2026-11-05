"""流水线各阶段（S0–S6）。

每个模块导出 ``run(...)``，返回 ``StageReport`` 风格 dict；产物只写 SessionPaths 约定路径。
"""
from . import (
    s0_ingest,
    s1_text,
    s1_visual,
    s1_voice,
    s2_chunk,
    s3_events,
    s4_agent,
    s5_feedback,
    s6_profile,
)

__all__ = [
    "s0_ingest",
    "s1_visual",
    "s1_voice",
    "s1_text",
    "s2_chunk",
    "s3_events",
    "s4_agent",
    "s5_feedback",
    "s6_profile",
]
