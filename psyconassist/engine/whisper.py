"""S1C 依赖：Whisper ASR 封装（faster-whisper）。"""
from __future__ import annotations

from pathlib import Path

from ..core.logging_utils import get_logger

log = get_logger(__name__)

SAMPLE_RATE = 16000


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_model_path(model_size: str, download_root: str) -> str:
    """模型已在缓存里就直接用快照目录，避免每次跑都要连 Hugging Face。"""
    if "/" in model_size or Path(model_size).exists():
        return model_size
    repo = f"Systran/faster-whisper-{model_size}"
    cache = Path(download_root) / ("models--" + repo.replace("/", "--"))
    snaps = cache / "snapshots"
    if snaps.exists():
        for snap in sorted(snaps.iterdir(), reverse=True):
            if (snap / "model.bin").exists():
                return str(snap)
    return model_size


def _run(model, sig, language):
    """Force the generator so CUDA/library errors surface here (not in the caller)."""
    segments, info = model.transcribe(sig, language=language, word_timestamps=True)
    return list(segments), info


def transcribe(wav: Path, model_size: str = "small", language: str = "zh") -> list[dict]:
    """返回 [{t0, t1, text, words:[{w,s,e}]}]，秒单位。"""
    import os

    import numpy as np
    from faster_whisper import WhisperModel

    from ..core.audio import load_mono

    # faster-whisper normally decodes with PyAV; av>=19 dropped the kwargs it
    # passes (`metadata_errors`), so hand it the PCM we already have instead.
    sig, sr = load_mono(Path(wav))
    if sr != SAMPLE_RATE:
        n = int(round(len(sig) * SAMPLE_RATE / sr))
        sig = np.interp(np.linspace(0, len(sig) - 1, n), np.arange(len(sig)), sig)
        sig = sig.astype(np.float32)
        sr = SAMPLE_RATE

    download_root = os.environ.get("WHISPER_CACHE", "/public_data/hf_cache/whisper")
    resolved = _resolve_model_path(model_size, download_root)
    model = WhisperModel(
        resolved,
        device="auto",
        compute_type="int8_float16",
        download_root=download_root,
    )
    try:
        segments, _info = _run(model, sig, language)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if not any(k in msg for k in ("cublas", "cudnn", "cuda")):
            raise
        log.warning("CUDA 推理不可用（%s），回退 CPU int8", exc)
        model = WhisperModel(
            resolved,
            device="cpu",
            compute_type="int8",
            download_root=download_root,
        )
        segments, _info = _run(model, sig, language)
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
