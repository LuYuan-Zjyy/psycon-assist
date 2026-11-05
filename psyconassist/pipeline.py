"""流水线编排：S0 → S6。

入口：
- ``run_full``       真实视频
- ``run_synthetic``  合成数据（验证 S2–S6，无需重引擎）
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import ids, synthetic
from .core.io import read_json, write_json
from .core.logging_utils import get_logger
from .core.models import MediaInfo, PipelineSummary, QualityReport
from .core.paths import SessionPaths
from .stages import (
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

log = get_logger(__name__)
ProgressCb = Callable[[str, str], None]


def run_full(
    video: Path,
    client_voice_sample: list[float],
    subject_id: str,
    session_id: str | None = None,
    use_llm: bool = True,
    progress: ProgressCb | None = None,
) -> PipelineSummary:
    """真实视频全流程。``progress(stage, detail)`` 可选。"""
    del use_llm  # 保留签名兼容；LLM 可用性由 llm.available() 决定
    session_id = session_id or ids.session_id(1)
    _emit(progress, "s0", "接入与归一化")
    media = s0_ingest.run(session_id, Path(video), client_voice_sample)
    return _run_stages(session_id, subject_id, media, progress=progress)


def run_synthetic(
    subject_id: str,
    duration: float = 3000.0,
    seed: int = 42,
    session_id: str | None = None,
) -> PipelineSummary:
    """合成数据全流程（S1 用桩 + S2–S6 真实逻辑）。"""
    session_id = session_id or ids.session_id(1)
    synthetic.generate(session_id, duration, seed)
    media: MediaInfo = {
        "session_id": session_id,
        "duration": duration,
        "fps": 5.0,
        "width": 1280,
        "height": 720,
        "has_audio": True,
        "client_voice_sample": [0.0, 3.0],
        "normalized_path": "",
        "audio_path": "",
    }
    write_json(SessionPaths(session_id).media_json, media)
    return _run_stages(session_id, subject_id, media, from_synthetic=True)


def _emit(progress: ProgressCb | None, stage: str, detail: str) -> None:
    log.info("[%s] %s", stage, detail)
    if not progress:
        return
    try:
        progress(stage, detail)
    except Exception:
        log.exception("progress callback failed stage=%s", stage)


def _run_stages(
    session_id: str,
    subject_id: str,
    media: MediaInfo,
    from_synthetic: bool = False,
    progress: ProgressCb | None = None,
) -> PipelineSummary:
    paths = SessionPaths(session_id)
    paths.ensure_layout()
    summary: PipelineSummary = {"session_id": session_id, "subject_id": subject_id}

    _emit(progress, "s1", "三路特征提取")
    if not from_synthetic:
        vis_report = s1_visual.run(session_id, media)
        voice_report = s1_voice.run(session_id, media)
        text_report = s1_text.run(session_id, media)
    else:
        vis_report = {"ok": True, "visual_unavailable": False, "face_cov": 0.9, "degrade": []}
        voice_report = {"ok": True, "diarization_ok": True, "voice_unavailable": False, "degrade": []}
        text_report = {"ok": True, "asr_ok": True, "text_unavailable": False, "degrade": []}

    degrade: list[str] = []
    for report in (vis_report, voice_report, text_report):
        for d in report.get("degrade") or []:
            if d not in degrade:
                degrade.append(d)
    # 兼容旧字段
    if vis_report.get("visual_unavailable") and "visual" not in degrade:
        degrade.append("visual")
    if voice_report.get("voice_unavailable") and "voice" not in degrade:
        degrade.append("voice")
    if text_report.get("text_unavailable") and "text" not in degrade:
        degrade.append("text")

    _emit(progress, "s2", "融合与异常切片")
    s2 = s2_chunk.run(session_id, media, degrade)
    summary["s2"] = s2

    _emit(progress, "s3", "指标与事件")
    s3 = s3_events.run(session_id, media, baseline_available=s2["baseline_available"])
    summary["s3"] = s3

    quality: QualityReport = {
        "face_cov": vis_report.get("face_cov"),
        "asr_ok": bool(text_report.get("asr_ok", True)),
        "diarization_ok": bool(voice_report.get("diarization_ok", True)),
        "degrade": degrade,
    }
    write_json(paths.quality_json, quality)

    _emit(progress, "s4", "Agent 编排（LangGraph）")
    profile = s6_profile.get_profile(subject_id)
    exemplars = s6_profile.load_exemplars(subject_id)
    summary_text = s6_profile.profile_summary(profile)
    s4 = s4_agent.run(session_id, subject_id, exemplars=exemplars, profile_summary=summary_text)
    summary["s4"] = {"n_claims": len(s4.get("claims", [])), "llm_ok": s4.get("llm_ok", False)}

    summary["s5"] = {"pending": s5_feedback.list_pending(session_id)}

    _emit(progress, "s6", "画像更新")
    chunks = read_json(paths.chunks_json)
    profile = s6_profile.update_profile(subject_id, session_id, chunks, quality)
    summary["s6"] = {
        "n_sessions": profile["n_sessions"],
        "comparability": profile["comparability"],
        "exemplars_count": profile["exemplars_count"],
    }

    log.info(
        "pipeline done session=%s events=%s claims=%s degrade=%s",
        session_id,
        s3.get("n_events"),
        summary["s4"].get("n_claims"),
        degrade,
    )
    return summary
