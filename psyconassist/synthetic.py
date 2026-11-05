"""合成数据生成器：无重模型引擎时，生成三路特征用于端到端验证 S2–S6。

埋入已知异常以便验证事件检出：va_shift / risk / silence / incongruence / low_activation。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .core import io


def generate(session_id: str, duration: float = 3000.0, seed: int = 42) -> dict:
    """写 au.parquet / speaker.json / voice.json / transcript.json / text_feat.json。"""
    rng = np.random.default_rng(seed)
    wdir = config.session_dir(session_id)
    fps = 5.0

    # ---- 视觉 au.parquet ----
    n = int(duration * fps)
    ts = (np.arange(n) / fps).round(3)
    au12 = 0.4 + 0.15 * np.sin(ts / 60) + rng.normal(0, 0.1, n)
    au04 = 0.3 + 0.1 * np.sin(ts / 90 + 1) + rng.normal(0, 0.1, n)
    # 埋 va_shift：1200–1300s AU04 升高（皱眉）
    mask_shift = (ts >= 1200) & (ts < 1300)
    au04[mask_shift] += 3.0
    # 埋 low_activation：2000–2100s 整体降低
    mask_low = (ts >= 2000) & (ts < 2100)
    au12[mask_low] *= 0.3
    au04[mask_low] *= 0.3
    # 埋视觉缺失：2600–2700s face_missing
    mask_miss = (ts >= 2600) & (ts < 2700)
    conf = np.full(n, 0.9)
    conf[mask_miss] = 0.0
    df = pd.DataFrame({
        "ts": ts, "face_id": 0, "conf": conf,
        "AU01_r": rng.normal(0.2, 0.2, n), "AU02_r": rng.normal(0.2, 0.2, n),
        "AU04_r": au04, "AU12_r": au12, "AU45_r": rng.normal(0.1, 0.2, n),
        "pose_Rx": rng.normal(0, 0.1, n), "pose_Ry": rng.normal(0, 0.1, n),
        "pose_Rz": rng.normal(0, 0.1, n),
        "face_missing": mask_miss,
    })
    io.write_parquet(wdir / "face" / "au.parquet", df)

    # ---- 语音 speaker.json + voice.json ----
    speaker, voice = _synth_speech(session_id, duration, rng)
    io.write_json(wdir / "speech" / "speaker.json", speaker)
    io.write_json(wdir / "speech" / "voice.json", voice)

    # ---- 文本 transcript.json + text_feat.json ----
    utt, feat = _synth_text(session_id, duration, rng)
    io.write_json(wdir / "text" / "transcript.json", utt)
    io.write_json(wdir / "text" / "text_feat.json", feat)

    return {"duration": duration, "n_frames": n}


def _synth_speech(session_id: str, duration: float, rng) -> tuple[list, list]:
    speaker, voice = [], []
    t = 0.0
    idx = 0
    while t < duration - 5:
        # 交替说话
        role = "client" if idx % 2 == 0 else "counselor"
        spk = "SPK_0" if role == "client" else "SPK_1"
        seg_len = float(rng.uniform(2.0, 8.0))
        t0, t1 = round(t, 3), round(min(t + seg_len, duration), 3)
        # 埋 silence：800–900s 来访者不说话（跳过 client 段）
        if role == "client" and 800 <= t0 <= 900:
            t = t1 + 10.0  # 大停顿
            idx += 1
            continue
        speaker.append({"seg_id": f"{session_id}-SG{idx:04d}", "t0": t0, "t1": t1,
                        "speaker_id": spk, "role": role})
        f0 = rng.normal(180 if role == "client" else 140, 20)
        voice.append({"seg_id": f"{session_id}-SG{idx:04d}", "t0": t0, "t1": t1,
                      "role": role, "f0_med": round(f0, 2), "f0_iqr": round(float(rng.uniform(20, 50)), 2),
                      "loudness_rel": round(float(rng.normal(0, 3)), 2),
                      "loudness_var": round(float(rng.uniform(1, 5)), 2)})
        t = t1 + float(rng.uniform(0.3, 1.5))
        idx += 1
    return speaker, voice


def _synth_text(session_id: str, duration: float, rng) -> tuple[list, list]:
    utt, feat = [], []
    t = 0.0
    idx = 0
    base_sentences = [
        "最近工作压力比较大", "睡眠不太好", "和家人沟通有些困难", "这周情绪比较平稳",
        "想起来一些以前的事", "觉得稍微好了一点", "有时候会突然很难过",
    ]
    while t < duration - 10:
        role = "client" if idx % 3 != 0 else "counselor"
        seg_len = float(rng.uniform(3, 10))
        t0, t1 = round(t, 3), round(min(t + seg_len, duration), 3)
        text = base_sentences[idx % len(base_sentences)]
        risk = []
        # 埋 risk：1500–1520s 命中风险词
        if 1500 <= t0 <= 1520:
            text = "有时候真的觉得撑不下去，想过结束生命"
            risk = [{"word": "结束生命", "span": [10, 14], "fragment": text[max(0, 10-6):14+6]}]
        polarity = round(float(rng.normal(-0.2, 0.4)), 4)
        # 埋 incongruence：2400–2420s 文本正但（视觉已埋低激活，见下）
        if 2400 <= t0 <= 2420:
            polarity = 0.6
        utt.append({"utt_id": f"{session_id}-U{idx:04d}", "t0": t0, "t1": t1,
                    "role": role, "text": text, "words": []})
        feat.append({"utt_id": f"{session_id}-U{idx:04d}", "t0": t0, "t1": t1,
                     "polarity": polarity, "topics": ["work"], "risk_hits": risk})
        t = t1 + float(rng.uniform(0.5, 2.0))
        idx += 1
    return utt, feat
