"""S5 反馈闭环：accept / edit / reject，六原因码，回流 exemplar。"""
from __future__ import annotations

from pathlib import Path

from . import config
from . import ids, io

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
    p = config.session_dir(session_id) / "agent" / "claims.json"
    if not p.exists():
        return []
    return [c for c in io.read_json(p) if c.get("state") == "pending"]


def apply_feedback(session_id: str, claim_id: str, action: str,
                   reason: str | None = None, revised_text: str | None = None,
                   subject_id: str | None = None) -> dict:
    assert action in ACTIONS, f"非法动作 {action}"
    if action == "reject":
        assert reason in REASON_CODES, f"非法原因码 {reason}（必须六选一）"
    if action == "edit":
        assert revised_text, "edit 必须提供 revised_text"

    wdir = config.session_dir(session_id)
    claims_path = wdir / "agent" / "claims.json"
    claims = io.read_json(claims_path) if claims_path.exists() else []

    fb = None
    for c in claims:
        if c.get("sid") == claim_id:
            fb = {
                "feedback_id": ids.feedback_id(session_id, _next_fb_seq(wdir)),
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

    if fb is None:
        raise KeyError(f"未找到 claim {claim_id}")

    io.write_json(claims_path, claims)
    _append_feedback(wdir, fb)
    _reflow(session_id, c, action, revised_text, subject_id)
    return fb


def _next_fb_seq(wdir: Path) -> int:
    p = wdir / "feedback" / "feedback.json"
    if p.exists():
        return len(io.read_json(p)) + 1
    return 1


def _append_feedback(wdir: Path, fb: dict) -> None:
    p = wdir / "feedback" / "feedback.json"
    existing = io.read_json(p) if p.exists() else []
    existing.append(fb)
    io.write_json(p, existing)


def _reflow(session_id: str, claim: dict, action: str, revised_text: str | None,
            subject_id: str | None) -> None:
    """回流：edit 存 exemplar（供 S4 RAG）。"""
    if action != "edit":
        return
    subject = subject_id or claim.get("subject_id") or "unknown"
    edir = config.SUBJECTS_DIR / subject / "exemplars"
    edir.mkdir(parents=True, exist_ok=True)
    io.write_json(
        edir / f"{claim['sid']}.json",
        {"original": claim.get("original_text", claim["text"]),
         "revised": revised_text, "event_type": claim.get("event_id", "")},
    )


def stats(session_id: str) -> dict:
    p = config.session_dir(session_id) / "feedback" / "feedback.json"
    if not p.exists():
        return {"n": 0, "by_action": {}, "by_reason": {}}
    fbs = io.read_json(p)
    by_action: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for f in fbs:
        by_action[f["action"]] = by_action.get(f["action"], 0) + 1
        if f.get("reason"):
            by_reason[f["reason"]] = by_reason.get(f["reason"], 0) + 1
    return {"n": len(fbs), "by_action": by_action, "by_reason": by_reason}


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
