"""S5 反馈闭环：accept / edit / reject，六原因码，回流 exemplar。"""
from __future__ import annotations

import datetime

from .. import ids
from ..core.errors import StageError
from ..core.io import read_json_optional, write_json
from ..core.logging_utils import get_logger
from ..core.paths import SessionPaths, SubjectPaths

log = get_logger(__name__)

ACTIONS = ("accept", "edit", "reject")

# 六个原因码（硬规则 17，不允许自由输入）
REASON_CODES = {
    "insufficient_evidence": "证据不足",
    "over_inference": "从行为跳到心理",
    "wording": "措辞不专业",
    "missing_info": "漏了重要事件",
    "time_misaligned": "定位不对",
    "other": "其他",
}


def list_pending(session_id: str) -> list[dict]:
    claims = read_json_optional(SessionPaths(session_id).claims_json, default=[]) or []
    return [c for c in claims if c.get("state") == "pending"]


def apply_feedback(
    session_id: str,
    claim_id: str,
    action: str,
    reason: str | None = None,
    revised_text: str | None = None,
    subject_id: str | None = None,
) -> dict:
    if action not in ACTIONS:
        raise StageError("s5", f"非法动作 {action}")
    if action == "reject" and reason not in REASON_CODES:
        raise StageError("s5", f"非法原因码 {reason}（必须六选一）")
    if action == "edit" and not revised_text:
        raise StageError("s5", "edit 必须提供 revised_text")

    paths = SessionPaths(session_id)
    claims = read_json_optional(paths.claims_json, default=[]) or []

    target = None
    fb = None
    for c in claims:
        if c.get("sid") != claim_id:
            continue
        target = c
        fb = {
            "feedback_id": ids.feedback_id(session_id, _next_fb_seq(paths)),
            "claim_id": claim_id,
            "action": action,
            "reason": reason,
            "ts": _now_iso(),
        }
        c["state"] = action
        if action == "edit":
            if "original_text" not in c:
                c["original_text"] = c["text"]
            c["text"] = revised_text
            c["revised_text"] = revised_text
        break

    if fb is None or target is None:
        raise KeyError(f"未找到 claim {claim_id}")

    write_json(paths.claims_json, claims)
    _append_feedback(paths, fb)
    _reflow(target, action, revised_text, subject_id)
    log.info("S5 feedback session=%s claim=%s action=%s", session_id, claim_id, action)
    return fb


def _next_fb_seq(paths: SessionPaths) -> int:
    existing = read_json_optional(paths.feedback_json, default=[]) or []
    return len(existing) + 1


def _append_feedback(paths: SessionPaths, fb: dict) -> None:
    existing = read_json_optional(paths.feedback_json, default=[]) or []
    existing.append(fb)
    write_json(paths.feedback_json, existing)


def _reflow(
    claim: dict,
    action: str,
    revised_text: str | None,
    subject_id: str | None,
) -> None:
    """回流：edit 存 exemplar（供 S4 RAG）。"""
    if action != "edit":
        return
    subject = subject_id or claim.get("subject_id") or "unknown"
    sp = SubjectPaths(subject)
    sp.ensure_layout()
    write_json(
        sp.exemplars_dir / f"{claim['sid']}.json",
        {
            "original": claim.get("original_text", claim["text"]),
            "revised": revised_text,
            "event_type": claim.get("event_id", ""),
        },
    )


def stats(session_id: str) -> dict:
    fbs = read_json_optional(SessionPaths(session_id).feedback_json, default=[]) or []
    by_action: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for f in fbs:
        by_action[f["action"]] = by_action.get(f["action"], 0) + 1
        if f.get("reason"):
            by_reason[f["reason"]] = by_reason.get(f["reason"], 0) + 1
    return {"n": len(fbs), "by_action": by_action, "by_reason": by_reason}


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
