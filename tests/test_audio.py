"""core.audio must decode the PCM WAV that S0 writes, without torchaudio."""
import wave

import numpy as np
import pytest

from psyconassist.core.audio import load_mono


@pytest.fixture
def wav_16k(tmp_path):
    path = tmp_path / "tone.wav"
    sr = 16000
    t = np.arange(sr // 2) / sr
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    pcm = (sig * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return path


def test_load_mono_shape_and_rate(wav_16k):
    data, sr = load_mono(wav_16k)
    assert sr == 16000
    assert data.shape == (8000,)
    assert data.dtype == np.float32
    assert abs(float(np.max(np.abs(data))) - 0.5) < 0.01


def test_stereo_is_downmixed(tmp_path):
    path = tmp_path / "stereo.wav"
    sr = 8000
    left = (0.4 * np.ones(sr // 10) * 32767).astype("<i2")
    right = (-0.4 * np.ones(sr // 10) * 32767).astype("<i2")
    inter = np.empty(left.size * 2, dtype="<i2")
    inter[0::2], inter[1::2] = left, right
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(inter.tobytes())
    data, sr = load_mono(path)
    assert sr == 8000
    assert data.shape == (800,)
    assert abs(float(np.mean(np.abs(data)))) < 0.01  # +0.4 and -0.4 cancel out
