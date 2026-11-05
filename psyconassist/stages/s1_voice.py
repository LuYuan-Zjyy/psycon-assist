"""S1B 语音：VAD → 分离 → 声学特征（按角色）。

只做声学特征（F0/响度），不做转写；转写归 S1C。
按 client / counselor 分开。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.audio import load_mono
from ..core.io import write_json
from ..core.logging_utils import get_logger
from ..core.models import MediaInfo, StageReport
from ..core.paths import SessionPaths
from ..engine import acoustic as aco
from ..engine import pyannote as pya

log = get_logger(__name__)


def run(session_id: str, media: MediaInfo) -> StageReport:
    paths = SessionPaths(session_id)
    paths.ensure_layout()
    wav = Path(media["audio_path"])
    client_sample = media.get("client_voice_sample") or []
    duration = float(media["duration"])

    vad_failed = False
    try:
        vad = _vad_energy(wav, duration)
    except Exception:
        log.exception("VAD 失败，按整段有声处理 session=%s", session_id)
        vad = [(0.0, duration)]
        vad_failed = True

    diarize_ok = False
    if pya.available():
        try:
            raw_segs = pya.diarize(wav, n_speakers=2)
            diarize_ok = True
        except Exception:
            log.exception("pyannote 分离失败，回退单说话人 session=%s", session_id)
            raw_segs = _fallback_single_speaker(vad)
    else:
        log.warning("pyannote 不可用，回退单说话人 session=%s", session_id)
        raw_segs = _fallback_single_speaker(vad)

    role_map = _map_roles(raw_segs, client_sample)
    segs = []
    for s in raw_segs:
        item = dict(s)
        item["role"] = role_map.get(s.get("speaker_id"), "unknown")
        if not diarize_ok:
            # 没做说话人分离就不该声称知道谁是来访者
            item["role"] = "unknown"
        segs.append(item)

    features = aco.segment_features(wav, segs) if aco.available() else segs

    speaker_out = [
        {
            "seg_id": f"{session_id}-SG{i:04d}",
            "t0": f["t0"],
            "t1": f["t1"],
            "speaker_id": f.get("speaker_id"),
            "role": f.get("role", "unknown"),
        }
        for i, f in enumerate(segs)
    ]
    voice_out = [
        {
            "seg_id": f"{session_id}-SG{i:04d}",
            "t0": f["t0"],
            "t1": f["t1"],
            "role": f.get("role", "unknown"),
            "f0_med": f.get("f0_med"),
            "f0_iqr": f.get("f0_iqr"),
            "loudness_rel": f.get("loudness_rel"),
            "loudness_var": f.get("loudness_var"),
        }
        for i, f in enumerate(features)
    ]
    vad_out = [{"t0": v[0], "t1": v[1], "voiced": True} for v in vad]

    write_json(paths.speaker_json, speaker_out)
    write_json(paths.voice_json, voice_out)
    write_json(paths.vad_json, vad_out)

    degrade = []
    if vad_failed:
        degrade.append("vad")
    if not diarize_ok:
        degrade.append("speaker_diarization")
    voice_unavailable = not aco.available()
    if voice_unavailable:
        degrade.append("voice")

    log.info(
        "S1B done session=%s segs=%d diarize_ok=%s",
        session_id, len(segs), diarize_ok,
    )
    return {
        "ok": True,
        "diarization_ok": diarize_ok,
        "n_speakers": len({s.get("speaker_id") for s in segs}),
        "n_segments": len(segs),
        "voice_unavailable": voice_unavailable,
        "degrade": degrade,
    }


def _vad_energy(
    wav: Path,
    duration: float,
    thresh_db: float = -38.0,
    min_len: float = 0.2,
    min_gap: float = 0.3,
) -> list[tuple[float, float]]:
    sig, sr = load_mono(wav)
    hop = int(0.01 * sr)
    segs: list[tuple[float, float]] = []
    start = None
    for i in range(0, len(sig) - hop + 1, hop):
        frame = sig[i : i + hop]
        rms = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-9)
        voiced = rms > thresh_db
        t = i / sr
        if voiced and start is None:
            start = t
        elif not voiced and start is not None:
            if t - start >= min_len:
                segs.append((round(start, 3), round(t, 3)))
            start = None
    if start is not None and duration - start >= min_len:
        segs.append((round(start, 3), round(duration, 3)))

    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append([s[0], s[1]])
    return [(m[0], m[1]) for m in merged]


def _fallback_single_speaker(vad: list) -> list[dict]:
    return [{"t0": t0, "t1": t1, "speaker_id": "SPK_0"} for t0, t1 in vad]


def _map_roles(segs: list[dict], client_sample: list[float]) -> dict:
    if not client_sample or len(client_sample) < 2:
        return {}
    s0, s1 = client_sample[0], client_sample[1]
    counts: dict = {}
    for s in segs:
        overlap = max(0.0, min(s["t1"], s1) - max(s["t0"], s0))
        if overlap > 0:
            spk = s.get("speaker_id")
            counts[spk] = counts.get(spk, 0.0) + overlap
    if not counts:
        return {}
    client_spk = max(counts, key=counts.get)
    mapping = {client_spk: "client"}
    for spk in {s.get("speaker_id") for s in segs}:
        if spk != client_spk:
            mapping[spk] = "counselor"
    return mapping
