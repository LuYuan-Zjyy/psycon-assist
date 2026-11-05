"""S1A 依赖：OpenFace 封装（外部程序，非 pip 包）。

配置 OPENFACE_BIN 指向 FeatureExtraction.exe。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .. import config


def available() -> bool:
    return bool(config.OPENFACE_BIN) and Path(config.OPENFACE_BIN).exists()


def extract_frames(video: Path, out_dir: Path, fps: float = 5.0) -> Path:
    """按指定 fps 抽帧（实际由 S1 层负责，这里仅暴露能力）。"""
    # 帧抽取放在 S1 视觉层用 ffmpeg 完成，OpenFace 只处理图片序列
    raise NotImplementedError


def run_openface(images_dir: Path, out_csv: Path) -> None:
    """对图片序列跑 OpenFace，输出逐帧 CSV。"""
    cmd = [
        config.OPENFACE_BIN,
        "-fdir", str(images_dir),
        "-out_dir", str(out_csv.parent),
        "-of", str(out_csv.name),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
