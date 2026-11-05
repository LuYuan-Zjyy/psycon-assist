"""S1A 视觉：唯一人脸 → AU → 头姿 → au.parquet。

画面只有来访者一张脸；出现第二张脸一律忽略（硬规则 8）。
缺失写 NaN，不写 0（硬规则 3）。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import io
from .engine import openface as of

# OpenFace 输出的 AU 列（强度）
AU_COLS = [f"AU{i:02d}_r" for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]]


def run(session_id: str, media: dict) -> dict:
    """返回质量报告；落盘 au.parquet。"""
    wdir = config.session_dir(session_id)
    norm = Path(media["normalized_path"])
    au_path = wdir / "face" / "au.parquet"

    if not of.available():
        # 降级：视觉不可用，写空 parquet（仅表头），face_cov=0
        empty = pd.DataFrame(columns=["ts", "face_id", "conf", *AU_COLS,
                                      "pose_Rx", "pose_Ry", "pose_Rz", "face_missing"])
        io.write_parquet(au_path, empty)
        return {"visual_unavailable": True, "face_cov": 0.0,
                "reason": "OpenFace 未配置（OPENFACE_BIN）"}

    duration = float(media["duration"])
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        frames = tmpdir / "frames"
        frames.mkdir()
        # 1. 粗跑抽帧 5fps
        _extract_frames(norm, frames, fps=5.0)
        csv = tmpdir / "openface.csv"
        of.run_openface(frames, csv)

        df = _parse_openface(csv, fps=5.0)
        # 2. 唯一人脸：conf > 0.7 取面积最大（简化：OpenFace 单脸，直接保留 conf>0.7）
        df = df[df["conf"] > 0.7].copy()
        if df.empty:
            io.write_parquet(au_path, df)
            return {"visual_unavailable": True, "face_cov": 0.0, "reason": "无高置信度人脸"}

        df["face_missing"] = False
        df["face_id"] = 0  # 唯一主脸 id 恒定

        # 3. 精跑（Demo 简化：直接用粗跑结果；TODO 对连续段原帧率重跑）
        # 4. 补缺失帧：构造全时间轴，缺失帧 face_missing=True、AU=NaN
        full_ts = np.arange(0.0, duration, 1.0 / 5.0).round(3)
        df = df.drop_duplicates("ts").set_index("ts")
        full = pd.DataFrame(index=full_ts)
        full.index.name = "ts"
        merged = full.join(df, how="left")
        merged["face_missing"] = merged["conf"].isna()
        merged["face_id"] = 0
        merged = merged.reset_index()
        # 缺失 AU 保持 NaN（不写 0）

    face_cov = float((~merged["face_missing"]).sum()) / max(len(merged), 1)
    io.write_parquet(au_path, merged)
    report = {"visual_unavailable": face_cov < 0.2, "face_cov": round(face_cov, 4)}
    if report["visual_unavailable"]:
        report["reason"] = f"face_cov={face_cov:.2f} < 0.2"
    return report


def _extract_frames(video: Path, out_dir: Path, fps: float) -> None:
    from .engine import ffmpeg as ff
    bin_ = ff._resolve_ffmpeg()
    if not bin_:
        raise RuntimeError("ffmpeg 不可用")
    subprocess.run(
        [bin_, "-y", "-i", str(video), "-vf", f"fps={fps}",
         str(out_dir / "frame_%06d.png")],
        capture_output=True, text=True, check=True,
    )


def _parse_openface(csv: Path, fps: float) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["ts"] = (df.index / fps).round(3)
    df["conf"] = df["confidence"].astype(float)
    keep = ["ts", "conf", *AU_COLS, "pose_Rx", "pose_Ry", "pose_Rz"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()
