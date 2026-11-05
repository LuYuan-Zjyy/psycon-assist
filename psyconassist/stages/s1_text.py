"""S1C 文本：ASR → 角色映射 → 极性 / 主题 / 风险。

ASR 归文本路；极性必须来自模型；风险宁可多报。
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import llm
from ..core.io import read_json_optional, write_json
from ..core.logging_utils import get_logger
from ..core.models import MediaInfo, StageReport
from ..core.paths import SessionPaths
from ..engine import sentiment as senti
from ..engine import whisper as wh

log = get_logger(__name__)

RISK_WORDS = [
    "自杀", "不想活", "活不下去", "想死", "结束生命", "自残", "割腕", "安眠药",
    "伤害自己", "伤害别人", "杀", "绝望", "撑不下去", "没意思", "崩溃",
]
RISK_RE = re.compile("|".join(re.escape(w) for w in RISK_WORDS))
TOPIC_SYSTEM = (
    "你是心理咨询转录的主题抽取器。从输入句子中抽取 1-3 个话题词。"
    '只输出 JSON：{"topics": ["词1", "词2"]}'
)


def run(session_id: str, media: MediaInfo) -> StageReport:
    paths = SessionPaths(session_id)
    paths.ensure_layout()
    wav = Path(media["audio_path"])
    speakers = read_json_optional(paths.speaker_json, default=[]) or []

    asr_ok = False
    raw: list[dict] = []
    if wh.available():
        try:
            raw = wh.transcribe(wav)
            asr_ok = True
        except Exception:
            log.exception("Whisper ASR 失败 session=%s", session_id)
            raw = []
    else:
        log.warning("Whisper 不可用 session=%s", session_id)

    utterances = _build_utterances(session_id, raw, speakers)
    texts = [u["text"] for u in utterances]
    pols = senti.polarity(texts) if texts else []
    topics = _extract_topics(texts)

    text_feat = []
    for i, u in enumerate(utterances):
        text_feat.append({
            "utt_id": u["utt_id"],
            "t0": u["t0"],
            "t1": u["t1"],
            "role": u.get("role"),
            "text": u.get("text"),
            "polarity": pols[i] if i < len(pols) else None,
            "topics": topics[i] if i < len(topics) else [],
            "risk_hits": _risk_scan(u["text"]),
        })

    write_json(paths.transcript_json, utterances)
    write_json(paths.text_feat_json, text_feat)

    log.info("S1C done session=%s utts=%d asr_ok=%s", session_id, len(utterances), asr_ok)
    return {
        "ok": asr_ok,
        "asr_ok": asr_ok,
        "n_utterances": len(utterances),
        "text_unavailable": not asr_ok,
        "degrade": [] if asr_ok else ["asr", "text"],
    }


def _build_utterances(session_id: str, raw: list[dict], speakers: list[dict]) -> list[dict]:
    out = []
    for i, seg in enumerate(raw):
        out.append({
            "utt_id": f"{session_id}-U{i:04d}",
            "t0": seg["t0"],
            "t1": seg["t1"],
            "role": _role_at(seg["t0"], seg["t1"], speakers),
            "text": seg["text"],
            "words": seg.get("words", []),
        })
    return out


def _role_at(t0: float, t1: float, speakers: list[dict]) -> str:
    best, best_role = 0.0, "unknown"
    for s in speakers:
        overlap = max(0.0, min(t1, s["t1"]) - max(t0, s["t0"]))
        if overlap > best:
            best = overlap
            best_role = s.get("role", "unknown")
    return best_role


def _extract_topics(texts: list[str]) -> list[list[str]]:
    if not texts:
        return []
    out = []
    for t in texts:
        r = llm.chat_json(TOPIC_SYSTEM, t)
        if r and isinstance(r, dict) and "topics" in r:
            out.append([str(x) for x in r["topics"]][:3])
        else:
            out.append([])
    return out


def _risk_scan(text: str) -> list[dict]:
    hits = []
    for m in RISK_RE.finditer(text):
        hits.append({
            "word": m.group(0),
            "span": [m.start(), m.end()],
            "fragment": text[max(0, m.start() - 6) : m.end() + 6],
        })
    return hits
