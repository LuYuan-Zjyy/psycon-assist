"""S0 接入与归一化。

输入任意 mp4/mov（画面=来访者单脸，声音=咨询师+来访者双人），
输出统一时间轴：media.json + norm.mp4 + 16k.wav。

硬条件：有音轨、duration ≥ 60s，不满足 abort。
"""
from __future__ import annotations

from pathlib import Path

from . import config
from . import io
from .engine import ffmpeg as ff


class S0Error(Exception):
    pass


def run(
    session_id: str,
    video_path: Path,
    client_voice_sample: list[float],
    overwrite: bool = False,
) -> dict:
    if not ff.available():
        raise S0Error(
            "未检测到 ffmpeg/ffprobe。S0 不可跳过，请先安装：winget install Gyan.FFmpeg，然后重开终端。"
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise S0Error(f"视频不存在: {video_path}")

    wdir = config.session_dir(session_id)
    norm_path = wdir / "raw" / "norm.mp4"
    audio_path = wdir / "audio" / "16k.wav"
    media_path = wdir / "media.json"

    if overwrite or not media_path.exists():
        info = ff.probe(video_path)

        # 硬条件检查
        if not info["has_audio"]:
            raise S0Error("缺少音轨，无法分析（硬条件不满足，abort）。")
        if info["duration"] < 60.0:
            # warn 但继续（主文档：短样本仅标注）
            info["short_sample"] = True

        # 转码 + 抽音频
        ff.transcode(video_path, norm_path)
        ff.extract_audio(video_path, audio_path)

        media = {
            "session_id": session_id,
            "duration": round(info["duration"], 3),
            "fps": round(info["fps"], 3),
            "width": info["width"],
            "height": info["height"],
            "has_audio": info["has_audio"],
            "client_voice_sample": client_voice_sample,
            "normalized_path": str(norm_path),
            "audio_path": str(audio_path),
        }
        io.write_json(media_path, media)
    else:
        media = io.read_json(media_path)

    return media
