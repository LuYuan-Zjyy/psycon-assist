"""S3 指标与事件定级：异常 chunk → event + evidence，规则定 severity。

event 与 chunk 一一对应（硬规则 13：每个 event 至少 1 条 evidence）。
severity 只由规则决定，LLM 无权改（硬规则 12）。
"""
from __future__ import annotations

from .. import ids
from ..core.errors import ContractError
from ..core.io import read_json, write_json
from ..core.logging_utils import get_logger
from ..core.models import MediaInfo, StageReport
from ..core.paths import SessionPaths

log = get_logger(__name__)

# severity 规则表（v1.1 §5.3）
SEVERITY = {
    "va_shift": "high",
    "risk": "high",
    "silence": "medium",
    "incongruence": "medium",
    "low_activation": "low",
}

TOOL_VERSION = "psyconassist-0.2.0"


def run(session_id: str, media: MediaInfo, baseline_available: bool = True) -> StageReport:
    del media  # 时间边界已在 S2 chunks 内；保留签名便于编排层统一调用
    paths = SessionPaths(session_id)
    chunks = read_json(paths.chunks_json)

    events: list[dict] = []
    all_evidence: list[dict] = []
    ev_seq = 0
    for c in chunks:
        if not c.get("is_anomaly"):
            continue
        atype = c["anomaly_type"]
        # 基线不可得：保留不依赖跨次基线的类型（短视频/无脸常见）
        if not baseline_available and atype not in ("risk", "silence", "low_activation", "va_shift"):
            continue

        evidence_ids = []
        for ev in _build_evidence(session_id, c, ev_seq):
            ev_seq += 1
            evidence_ids.append(ev["evidence_id"])
            all_evidence.append(ev)

        events.append({
            "event_id": ids.event_id(session_id, len(events) + 1),
            "chunk_id": c["chunk_id"],
            "type": atype,
            "t0": c["t0"],
            "t1": c["t1"],
            "severity": SEVERITY.get(atype, "medium"),
            "trigger": c.get("trigger"),
            "evidence_ids": evidence_ids,
        })

    for e in events:
        if not e["evidence_ids"]:
            raise ContractError(f"event {e['event_id']} 无 evidence")

    write_json(paths.events_json, events)
    write_json(paths.evidence_json, all_evidence)
    by_type = _count_by_type(events)
    log.info("S3 done session=%s events=%s by_type=%s", session_id, len(events), by_type)
    return {"ok": True, "n_events": len(events), "by_type": by_type, "degrade": []}


def _build_evidence(session_id: str, c: dict, start_seq: int) -> list[dict]:
    """为一个异常 chunk 生成 evidence（时间 + 指标 + 值 + 工具版本）。"""
    evs: list[dict] = []
    seq = start_seq

    def add(metric: str, value, t0: float, t1: float) -> None:
        nonlocal seq
        seq += 1
        evs.append({
            "evidence_id": ids.evidence_id(session_id, seq),
            "chunk_id": c["chunk_id"],
            "t0": round(t0, 3),
            "t1": round(t1, 3),
            "metric": metric,
            "value": value,
            "tool_version": TOOL_VERSION,
        })

    va = c.get("va")
    if va and va.get("v") is not None:
        add("va.v", round(va["v"], 4), c["t0"], c["t1"])
    if va and va.get("a") is not None:
        add("va.a", round(va["a"], 4), c["t0"], c["t1"])
    txt = c.get("text") or {}
    if txt.get("polarity") is not None:
        add("polarity", round(txt["polarity"], 4), c["t0"], c["t1"])
    if c.get("all_types"):
        add("anomaly_types", c["all_types"], c["t0"], c["t1"])
    return evs


def _count_by_type(events: list[dict]) -> dict[str, int]:
    d: dict[str, int] = {}
    for e in events:
        d[e["type"]] = d.get(e["type"], 0) + 1
    return d
