"""阶段间契约模型（TypedDict）。

队友改接口时先改这里，再改读写方；避免魔法 dict key 散落各处。
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


Role = Literal["client", "counselor", "unknown"]
AnomalyType = Literal[
    "normal", "risk", "va_shift", "silence", "incongruence", "low_activation"
]
Severity = Literal["high", "medium", "low"]
ClaimState = Literal["pending", "accept", "edit", "reject"]
FeedbackAction = Literal["accept", "edit", "reject"]


class MediaInfo(TypedDict, total=False):
    """S0 产出；核心字段在写入时必填，short_sample 仅短视频出现。"""
    session_id: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool
    client_voice_sample: list[float]
    normalized_path: str
    audio_path: str
    short_sample: bool


class StageReport(TypedDict, total=False):
    """各 stage.run 的标准返回；pipeline 用它汇总 degrade。"""
    ok: bool
    degrade: list[str]
    reason: str
    # S1A
    visual_unavailable: bool
    face_cov: float
    # S1B
    diarization_ok: bool
    voice_unavailable: bool
    n_speakers: int
    n_segments: int
    # S1C
    asr_ok: bool
    text_unavailable: bool
    n_utterances: int
    # S2
    n_chunks: int
    n_anomaly: int
    baseline_available: bool
    # S3
    n_events: int
    by_type: dict[str, int]
    # S4
    n_claims: int
    llm_ok: bool


class QualityReport(TypedDict):
    face_cov: float | None
    asr_ok: bool
    diarization_ok: bool
    degrade: list[str]


class VAPoint(TypedDict, total=False):
    v: float | None
    a: float | None
    conf: float | None


class Chunk(TypedDict, total=False):
    chunk_id: str
    t0: float
    t1: float
    is_anomaly: bool
    anomaly_type: AnomalyType
    all_types: list[str]
    trigger: Any
    va: VAPoint | None
    text: dict[str, Any]
    missing: list[str]


class Evidence(TypedDict):
    evidence_id: str
    chunk_id: str
    t0: float
    t1: float
    metric: str
    value: Any
    tool_version: str


class Event(TypedDict, total=False):
    event_id: str
    chunk_id: str
    type: AnomalyType
    t0: float
    t1: float
    severity: Severity
    trigger: Any
    evidence_ids: list[str]


class Claim(TypedDict, total=False):
    sid: str
    event_id: str
    event_type: str
    text: str
    original_text: str
    revised_text: str
    evidence_ids: list[str]
    state: ClaimState


class PipelineSummary(TypedDict, total=False):
    session_id: str
    subject_id: str
    s2: StageReport
    s3: StageReport
    s4: dict[str, Any]
    s5: dict[str, Any]
    s6: dict[str, Any]
