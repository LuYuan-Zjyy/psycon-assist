"""会话上下文：把一场会谈压缩成助手可用的紧凑视图。"""
from __future__ import annotations

from ..core.io import read_json_optional
from ..core.paths import SessionPaths


def build(session_id: str) -> dict:
    p = SessionPaths(session_id)
    media = read_json_optional(p.media_json, default={}) or {}
    chunks = read_json_optional(p.chunks_json, default=[]) or []
    events = read_json_optional(p.events_json, default=[]) or []
    claims = read_json_optional(p.claims_json, default=[]) or []
    transcript = read_json_optional(p.transcript_json, default=[]) or []
    baseline = read_json_optional(p.baseline_json, default=None)
    return {
        "session_id": session_id,
        "duration": media.get("duration"),
        "chunks": chunks,
        "events": events,
        "claims": claims,
        "transcript": transcript,
        "baseline": baseline,
    }


def brief(ctx: dict, max_claims: int = 8, max_events: int = 8) -> str:
    """给 LLM 的精简背景（避免整段转写塞进 prompt）。"""
    lines = [f"会谈 {ctx['session_id']}，时长 {ctx.get('duration')} 秒。"]
    b = ctx.get("baseline") or {}
    if b.get("available"):
        lines.append(
            "本场基线：效价 %s±%s，唤醒 %s±%s（窗口 %s 秒）。" % (
                b.get("v_mean"), b.get("v_sd"), b.get("a_mean"), b.get("a_sd"), b.get("window"))
        )
    evs = ctx.get("events", [])[:max_events]
    if evs:
        lines.append("事件：")
        for e in evs:
            lines.append("  - %s %s-%ss 严重度%s" % (e.get("type"), e.get("t0"), e.get("t1"), e.get("severity")))
    cls = ctx.get("claims", [])[:max_claims]
    if cls:
        lines.append("已有草稿（前 %d 条）：" % len(cls))
        for c in cls:
            lines.append("  - [%s] %s %s" % (c.get("sid"), c.get("state"), (c.get("text") or "")[:60]))
    return "\n".join(lines)
