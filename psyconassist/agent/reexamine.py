"""反馈后的再审视（ReAct 的第一步）：咨询师改了/否决了草稿，重新看证据与基线。

输出写 agent/reexamine.json，供界面展示；不改动原始草稿（改动由 S5 负责）。
"""
from __future__ import annotations

import datetime
import json

from .. import llm
from ..core.io import read_json_optional, write_json
from ..core.paths import SessionPaths
from ..kb import search
from . import context as ctx_mod

SYSTEM = """你是心理咨询督导助手。咨询师修改或否决了一条系统草稿，请重新审视证据。
输出严格 JSON：
{"verdict":"support|revise|reject",
 "why":"一句话说明你为什么同意/不同意咨询师的判断（<=60字）",
 "suggestion":"给咨询师的下一步建议（<=80字，具体、可执行）",
 "baseline_note":"是否需要调整本场基线判断：none|recheck（<=40字）",
 "persona_note":"是否需要调整对该来访者的理解：none|update（<=50字）"}
不做诊断；只基于给定证据与知识库。
"""


def run(session_id: str, claim: dict, action: str, reason: str | None, revised_text: str | None) -> dict | None:
    ctx = ctx_mod.build(session_id)
    event = next((e for e in ctx["events"] if e.get("event_id") == claim.get("event_id")), None)
    kb = search((claim.get("text") or "") + " " + (revised_text or "") + " " + (reason or ""), k=4)
    payload = {
        "claim": {"sid": claim.get("sid"), "text": claim.get("text"), "state": claim.get("state")},
        "event": event,
        "counselor_action": action,
        "counselor_reason": reason,
        "counselor_revision": revised_text,
        "baseline": ctx.get("baseline"),
        "kb": [{"id": e["id"], "title": e["title"], "content": e["content"]} for e in kb],
    }
    out = llm.chat_json(SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.2, tag="reexamine")
    if not isinstance(out, dict):
        return None
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "claim_id": claim.get("sid"),
        "action": action,
        "reason": reason,
        "revised_text": revised_text,
        "kb_ids": [e["id"] for e in kb],
        "verdict": out.get("verdict"),
        "why": out.get("why"),
        "suggestion": out.get("suggestion"),
        "baseline_note": out.get("baseline_note"),
        "persona_note": out.get("persona_note"),
    }
    paths = SessionPaths(session_id)
    fp = paths.root / "agent" / "reexamine.json"
    items = read_json_optional(fp, default=[]) or []
    items.append(rec)
    write_json(fp, items)
    return rec
