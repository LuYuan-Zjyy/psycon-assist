"""WAV loading that does not depend on torchcodec.

S0 always writes 16 kHz mono 16-bit PCM, so the stdlib ``wave`` module is
sufficient. torchaudio is kept only as a fallback for exotic formats — on
torchaudio >= 2.9 it needs a working torchcodec build, which is a common
failure mode on Windows.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Return (float32 mono signal in [-1, 1], sample_rate)."""
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
        if width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 1:
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported sample width: {width}")
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data.astype(np.float32), int(sr)
    except Exception:
        pass

    # Fallback: torchaudio (may require torchcodec on torchaudio >= 2.9)
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).numpy().astype(np.float32), int(sr)
