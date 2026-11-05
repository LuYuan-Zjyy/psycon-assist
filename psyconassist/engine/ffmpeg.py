"""S0 依赖：ffmpeg / ffprobe 封装。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .. import config

_FFMPEG = None
_FFPROBE = None


def _candidate_bins() -> list[Path]:
    home = Path(os.environ.get("LOCALAPPDATA", ""))
    return [
        Path(config.FFMPEG_BIN) if config.FFMPEG_BIN else None,
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        home / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        home / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe" / "ffmpeg-9.0.2-full_build" / "bin" / "ffmpeg.exe",
    ]


def _resolve_ffmpeg() -> str | None:
    global _FFMPEG
    if _FFMPEG is not None:
        return _FFMPEG or None
    which = shutil.which("ffmpeg")
    if which:
        _FFMPEG = which
        return _FFMPEG
    for p in _candidate_bins():
        if p and p.exists():
            _FFMPEG = str(p)
            return _FFMPEG
    _FFMPEG = ""
    return None


def _resolve_ffprobe() -> str | None:
    global _FFPROBE
    if _FFPROBE is not None:
        return _FFPROBE or None
    if config.FFPROBE_BIN and Path(config.FFPROBE_BIN).exists():
        _FFPROBE = config.FFPROBE_BIN
        return _FFPROBE
    which = shutil.which("ffprobe")
    if which:
        _FFPROBE = which
        return _FFPROBE
    ff = _resolve_ffmpeg()
    if ff:
        cand = Path(ff).with_name("ffprobe.exe")
        if cand.exists():
            _FFPROBE = str(cand)
            return _FFPROBE
    _FFPROBE = ""
    return None


def available() -> bool:
    return bool(_resolve_ffmpeg() and _resolve_ffprobe())


def probe(video: Path) -> dict:
    """读取时长/帧率/分辨率/音轨信息。失败抛异常。"""
    cmd = [
        _resolve_ffprobe(), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate,width,height,codec_name",
        "-show_entries", "format=duration",
        "-of", "json", str(video),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    vstream = streams[0] if streams else {}
    fr = vstream.get("r_frame_rate", "0/1")
    num, den = (fr.split("/") + ["1"])[:2]
    try:
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(fmt.get("duration") or vstream.get("duration") or 0.0)
    has_audio = _has_audio(video)
    return {
        "duration": duration,
        "fps": fps,
        "width": vstream.get("width"),
        "height": vstream.get("height"),
        "codec": vstream.get("codec_name"),
        "has_audio": has_audio,
    }


def _has_audio(video: Path) -> bool:
    cmd = [
        _resolve_ffprobe(), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "json", str(video),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    data = json.loads(out)
    return bool(data.get("streams"))


def transcode(video: Path, out: Path) -> None:
    """H.264 + GOP 60 + faststart（保证前端 seek 不黑屏）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _resolve_ffmpeg(), "-y", "-i", str(video),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-g", "60", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def extract_audio(video: Path, out_wav: Path) -> None:
    """抽 16kHz 单声道 wav。"""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _resolve_ffmpeg(), "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(out_wav),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
