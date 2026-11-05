"""S1C 依赖：Whisper ASR 封装（faster-whisper）。"""
from __future__ import annotations

from pathlib import Path


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def transcribe(wav: Path, model_size: str = "small", language: str = "zh") -> list[dict]:
    """返回 [{t0, t1, text, words:[{w,s,e}]}]，秒单位。"""
    import os
    from faster_whisper import WhisperModel

    download_root = os.environ.get("WHISPER_CACHE", "/public_data/hf_cache/whisper")
    model = WhisperModel(
        model_size,
        device="auto",
        compute_type="int8_float16",
        download_root=download_root,
    )
    segments, _info = model.transcribe(str(wav), language=language)
    out = []
    for seg in segments:
        words = [
            {"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3)}
            for w in (seg.words or [])
        ]
        out.append({
            "t0": round(seg.start, 3),
            "t1": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words,
        })
    return out
