"""S1B 依赖：pyannote 说话人分离封装。"""
from __future__ import annotations

from pathlib import Path

from .. import config

_pipeline = None


def _patch_torch_load() -> None:
    """Torch 2.6 默认 weights_only=True，会挡 pyannote checkpoint。"""
    import torch

    try:
        from torch.torch_version import TorchVersion

        torch.serialization.add_safe_globals([TorchVersion])
    except Exception:
        pass
    if getattr(torch.load, "_psy_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        # lightning/pyannote 可能显式传 weights_only=True，必须强制关掉
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)

    _load._psy_patched = True  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]


def available() -> bool:
    if not config.HF_TOKEN:
        return False
    try:
        import pyannote.audio  # noqa: F401
        return True
    except Exception:
        return False


def diarize(wav: Path, n_speakers: int = 2) -> list[dict]:
    """返回 [{t0, t1, speaker_id}]，秒单位。"""
    global _pipeline
    from pyannote.audio import Pipeline

    _patch_torch_load()
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.HF_TOKEN,
        )
        try:
            import torch

            if torch.cuda.is_available():
                _pipeline.to(torch.device("cuda"))
        except Exception:
            pass
    diarization = _pipeline(str(wav), num_speakers=n_speakers)
    segs = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segs.append({"t0": round(turn.start, 3), "t1": round(turn.end, 3), "speaker_id": speaker})
    return segs
