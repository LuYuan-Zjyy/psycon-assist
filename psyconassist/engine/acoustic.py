"""S1B 声学特征：F0（声调）与响度（振幅）。纯 numpy + torchaudio，无额外重依赖。"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def available() -> bool:
    # numpy is enough for the 16 kHz mono PCM written by S0 (see core/audio.py);
    # torchaudio is only a fallback for exotic formats.
    try:
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def _load_wav(wav: Path):
    from ..core.audio import load_mono

    return load_mono(Path(wav))


def _f0_autocorr(signal: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """自相关法估计基频，返回 Hz；无 voiced 返回 NaN。"""
    if len(signal) < sr // fmin:
        return float("nan")
    # 去直流 + 归一化
    x = signal - np.mean(signal)
    if np.std(x) < 1e-6:
        return float("nan")
    x = x / (np.std(x) + 1e-9)
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    if max_lag >= len(x):
        return float("nan")
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac[min_lag:max_lag]
    if len(ac) == 0 or np.max(ac) <= 0:
        return float("nan")
    lag = int(np.argmax(ac)) + min_lag
    return sr / lag


def compute_f0(signal: np.ndarray, sr: int, frame: float = 0.03, hop: float = 0.01) -> list[float]:
    """分帧估计 F0 序列，返回逐帧 Hz（无 voiced 为 NaN）。"""
    fl = int(frame * sr)
    hop_l = int(hop * sr)
    f0s = []
    for i in range(0, len(signal) - fl + 1, hop_l):
        f0s.append(_f0_autocorr(signal[i:i + fl], sr))
    return f0s


def _rms_db(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    return 20 * np.log10(rms + 1e-9)


def segment_features(wav: Path, segments: list[dict]) -> list[dict]:
    """对每个有声段计算声学特征（F0 中位/IQR、相对响度）。segments=[{t0,t1,...}]。"""
    signal, sr = _load_wav(wav)
    # 整场响度均值，用于「相对响度」
    full_rms = _rms_db(signal)

    out = []
    for seg in segments:
        i0 = int(seg["t0"] * sr)
        i1 = int(seg["t1"] * sr)
        i0 = max(0, min(i0, len(signal)))
        i1 = max(0, min(i1, len(signal)))
        if i1 - i0 < int(0.05 * sr):
            out.append({**seg, "f0_med": None, "f0_iqr": None, "loudness_rel": None, "loudness_var": None})
            continue
        seg_sig = signal[i0:i1]
        f0s = compute_f0(seg_sig, sr)
        f0s = np.array([f for f in f0s if not np.isnan(f)])
        if len(f0s) > 0:
            f0_med = float(np.median(f0s))
            f0_iqr = float(np.percentile(f0s, 75) - np.percentile(f0s, 25))
        else:
            f0_med = f0_iqr = None
        rms = _rms_db(seg_sig)
        out.append({
            **seg,
            "f0_med": f0_med,
            "f0_iqr": f0_iqr,
            "loudness_rel": float(rms - full_rms),
            "loudness_var": float(np.std(np.abs(seg_sig))),
        })
    return out
