"""流水线编排：S0 → S6 全流程。

两种入口：
- run_full      真实视频（需 ffmpeg / OpenFace / pyannote / Whisper，缺失自动降级）
- run_synthetic 合成数据（无需任何重引擎，验证 S2–S6）
"""
from __future__ import annotations

from pathlib import Path

from . import config
from . import (
    ids,
    io,
    s0_ingest,
    s1_text,
    s1_visual,
    s1_voice,
    s2_chunk,
    s3_events,
    s4_agent,
    s5_feedback,
    s6_profile,
    synthetic,
)


def run_full(video: Path, client_voice_sample: list[float], subject_id: str,
             session_id: str | None = None, use_llm: bool = True,
             progress=None) -> dict:
    """真实视频全流程。progress(stage, detail) 可选回调。"""
    session_id = session_id or ids.session_id(1)
    _emit(progress, "s0", "接入与归一化")
    media = s0_ingest.run(session_id, video, client_voice_sample)
    return _run_stages(session_id, subject_id, media, progress=progress)


def _emit(progress, stage: str, detail: str) -> None:
    if progress:
        try:
            progress(stage, detail)
        except Exception:
            pass


def run_synthetic(subject_id: str, duration: float = 3000.0, seed: int = 42,
                  session_id: str | None = None) -> dict:
    """合成数据全流程（S1 用桩 + S2–S6 真实逻辑）。"""
    session_id = session_id or ids.session_id(1)
    # 合成三路特征（替代 S0/S1 重引擎）
    synthetic.generate(session_id, duration, seed)
    media = {
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
    # 落盘 media.json（与真实流程一致，供前端读取）
    io.write_json(config.session_dir(session_id) / "media.json", media)
    return _run_stages(session_id, subject_id, media, from_synthetic=True)


def _run_stages(session_id: str, subject_id: str, media: dict, from_synthetic: bool = False,
                progress=None) -> dict:
    wdir = config.session_dir(session_id)
    summary = {"session_id": session_id, "subject_id": subject_id}

    # S1 三路（合成模式跳过，特征已生成）
    _emit(progress, "s1", "三路特征提取")
    if not from_synthetic:
        vis_report = s1_visual.run(session_id, media)
        voice_report = s1_voice.run(session_id, media)
        text_report = s1_text.run(session_id, media)
    else:
        vis_report = {"visual_unavailable": False, "face_cov": 0.9}
        voice_report = {"diarization_ok": True, "voice_unavailable": False}
        text_report = {"asr_ok": True, "text_unavailable": False}

    degrade = []
    if vis_report.get("visual_unavailable"):
        degrade.append("visual")
    if voice_report.get("voice_unavailable"):
        degrade.append("voice")
    if text_report.get("text_unavailable"):
        degrade.append("text")

    # S2 切片
    _emit(progress, "s2", "融合与异常切片")
    s2 = s2_chunk.run(session_id, media, degrade)
    summary["s2"] = s2

    # S3 事件
    _emit(progress, "s3", "指标与事件")
    s3 = s3_events.run(session_id, media, baseline_available=s2["baseline_available"])
    summary["s3"] = s3

    # 质量报告
    quality = {
        "face_cov": vis_report.get("face_cov"),
        "asr_ok": text_report.get("asr_ok", True),
        "diarization_ok": voice_report.get("diarization_ok", True),
        "degrade": degrade,
    }
    io.write_json(wdir / "quality.json", quality)

    # S4 Agent（读历史画像 + exemplar）
    _emit(progress, "s4", "Agent 编排（LangGraph）")
    profile = s6_profile.get_profile(subject_id)
    exemplars = s6_profile.load_exemplars(subject_id)
    summary_text = s6_profile.profile_summary(profile)
    s4 = s4_agent.run(session_id, subject_id, exemplars=exemplars, profile_summary=summary_text)
    summary["s4"] = {"n_claims": len(s4.get("claims", [])), "llm_ok": s4.get("llm_ok", False)}

    # S5 反馈（Demo：待确认，不做自动 accept）
    summary["s5"] = {"pending": s5_feedback.list_pending(session_id)}

    # S6 画像更新
    _emit(progress, "s6", "画像更新")
    chunks = io.read_json(wdir / "chunks" / "chunks.json")
    profile = s6_profile.update_profile(subject_id, session_id, chunks, quality)
    summary["s6"] = {
        "n_sessions": profile["n_sessions"],
        "comparability": profile["comparability"],
        "exemplars_count": profile["exemplars_count"],
    }

    return summary
