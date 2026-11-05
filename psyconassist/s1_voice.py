"""S1B 语音：VAD → 分离 → 声学特征（按角色）。

只做声学特征（F0/响度），不做转写（硬规则 21）；转写归 S1C。
按 client / counselor 分开（硬规则 22）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from . import io
from .engine import acoustic as aco
from .engine import pyannote as pya


def run(session_id: str, media: dict) -> dict:
    wdir = config.session_dir(session_id)
    wav = Path(media["audio_path"])
    client_sample = media.get("client_voice_sample") or []
    duration = float(media["duration"])

    # 1. VAD（能量阈值）
    vad = _vad_energy(wav, duration)

    # 2. 说话人分离
    diarize_ok = False
    if pya.available():
        try:
            raw_segs = pya.diarize(wav, n_speakers=2)
            diarize_ok = True
        except Exception:
            raw_segs = _fallback_single_speaker(vad)
    else:
        raw_segs = _fallback_single_speaker(vad)

    # 3. 角色映射：client_voice_sample 时间段内占比最高的 speaker → client
    role_map = _map_roles(raw_segs, client_sample)

    segs = []
    for s in raw_segs:
        s = dict(s)
        s["role"] = role_map.get(s.get("speaker_id"), "unknown")
        segs.append(s)

    # 4. 声学特征（按段）
    features = aco.segment_features(wav, segs) if aco.available() else segs

    # 写 speaker.json / voice.json / vad.json
    speaker_out = [
        {"seg_id": f"{session_id}-SG{i:04d}", "t0": f["t0"], "t1": f["t1"],
         "speaker_id": f.get("speaker_id"), "role": f.get("role", "unknown")}
        for i, f in enumerate(segs)
    ]
    voice_out = [
        {"seg_id": f"{session_id}-SG{i:04d}", "t0": f["t0"], "t1": f["t1"],
         "role": f.get("role", "unknown"),
         "f0_med": f.get("f0_med"), "f0_iqr": f.get("f0_iqr"),
         "loudness_rel": f.get("loudness_rel"), "loudness_var": f.get("loudness_var")}
        for i, f in enumerate(features)
    ]
    vad_out = [{"t0": v[0], "t1": v[1], "voiced": True} for v in vad]

    io.write_json(wdir / "speech" / "speaker.json", speaker_out)
    io.write_json(wdir / "speech" / "voice.json", voice_out)
    io.write_json(wdir / "speech" / "vad.json", vad_out)

    return {
        "diarization_ok": diarize_ok,
        "n_speakers": len(set(s.get("speaker_id") for s in segs)),
        "n_segments": len(segs),
        "voice_unavailable": not aco.available(),
        "degrade": [] if diarize_ok else ["speaker_diarization"],
    }


def _vad_energy(wav: Path, duration: float, thresh_db: float = -38.0, min_len: float = 0.2, min_gap: float = 0.3) -> list[tuple[float, float]]:
    """能量阈值 VAD，返回 [(t0, t1)]。"""
    import torchaudio

    waveform, sr = torchaudio.load(str(wav))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    sig = waveform.squeeze(0).numpy()
    hop = int(0.01 * sr)
    segs = []
    start = None
    for i in range(0, len(sig) - hop + 1, hop):
        frame = sig[i:i + hop]
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
    # 合并间隔 < min_gap 的段
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(list(s))
    return [tuple(m) for m in merged]


def _fallback_single_speaker(vad: list) -> list[dict]:
    return [{"t0": t0, "t1": t1, "speaker_id": "SPK_0"} for t0, t1 in vad]


def _map_roles(segs: list[dict], client_sample: list[float]) -> dict:
    """client_voice_sample 时间段内占比最高的 speaker → client。"""
    if not client_sample or len(client_sample) < 2:
        return {}
    s0, s1 = client_sample[0], client_sample[1]
    counts = {}
    for s in segs:
        overlap = max(0.0, min(s["t1"], s1) - max(s["t0"], s0))
        if overlap > 0:
            spk = s.get("speaker_id")
            counts[spk] = counts.get(spk, 0.0) + overlap
    if not counts:
        return {}
    client_spk = max(counts, key=counts.get)
    speakers = set(s.get("speaker_id") for s in segs)
    mapping = {client_spk: "client"}
    for spk in speakers:
        if spk != client_spk:
            mapping[spk] = "counselor"
    return mapping
