"""S0 接入与归一化。

输入任意 mp4/mov（画面=来访者单脸，声音=咨询师+来访者双人），
输出统一时间轴：media.json + norm.mp4 + 16k.wav。

硬条件：有音轨；时长 < 60s 仅标注 short_sample。
"""
from __future__ import annotations

from pathlib import Path

from ..core.errors import StageError
from ..core.io import read_json, write_json
from ..core.logging_utils import get_logger
from ..core.models import MediaInfo
from ..core.paths import SessionPaths
from ..engine import ffmpeg as ff

log = get_logger(__name__)


def run(
    session_id: str,
    video_path: Path,
    client_voice_sample: list[float],
    overwrite: bool = False,
) -> MediaInfo:
    if not ff.available():
        raise StageError(
            "s0",
            "未检测到 ffmpeg/ffprobe。S0 不可跳过，请安装 ffmpeg 后重试。",
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise StageError("s0", f"视频不存在: {video_path}")

    if len(client_voice_sample) < 2 or client_voice_sample[1] <= client_voice_sample[0]:
        raise StageError("s0", "client_voice_sample 必须为 [t0, t1] 且 t1 > t0")

    paths = SessionPaths(session_id)
    paths.ensure_layout()

    if overwrite or not paths.media_json.exists():
        info = ff.probe(video_path)
        if not info["has_audio"]:
            raise StageError("s0", "缺少音轨，无法分析（硬条件不满足，abort）。")
        if info["duration"] < 60.0:
            log.warning("session=%s duration=%.1fs < 60，标记 short_sample", session_id, info["duration"])
            info["short_sample"] = True

        ff.transcode(video_path, paths.norm_mp4)
        ff.extract_audio(video_path, paths.audio_wav)

        media: MediaInfo = {
            "session_id": session_id,
            "duration": round(info["duration"], 3),
            "fps": round(info["fps"], 3),
            "width": info["width"],
            "height": info["height"],
            "has_audio": info["has_audio"],
            "client_voice_sample": list(client_voice_sample),
            "normalized_path": str(paths.norm_mp4),
            "audio_path": str(paths.audio_wav),
        }
        if info.get("short_sample"):
            media["short_sample"] = True
        write_json(paths.media_json, media)
        log.info("S0 done session=%s duration=%.1fs", session_id, media["duration"])
        return media

    return read_json(paths.media_json)
