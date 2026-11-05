"""S1A 视觉：唯一人脸 → AU → 头姿 → au.parquet。

画面只有来访者一张脸；出现第二张脸一律忽略。
缺失写 NaN，不写 0。
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.io import write_parquet
from ..core.logging_utils import get_logger
from ..core.models import MediaInfo, StageReport
from ..core.paths import SessionPaths
from ..engine import openface as of

log = get_logger(__name__)

AU_COLS = [
    f"AU{i:02d}_r"
    for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
]
POSE_COLS = ["pose_Rx", "pose_Ry", "pose_Rz"]


def run(session_id: str, media: MediaInfo) -> StageReport:
    paths = SessionPaths(session_id)
    paths.ensure_layout()
    au_path = paths.au_parquet
    empty_cols = ["ts", "face_id", "conf", *AU_COLS, *POSE_COLS, "face_missing"]

    if not of.available():
        write_parquet(au_path, pd.DataFrame(columns=empty_cols))
        log.warning("S1A degrade: OpenFace 未配置 session=%s", session_id)
        return {
            "ok": False,
            "visual_unavailable": True,
            "face_cov": 0.0,
            "degrade": ["visual"],
            "reason": "OpenFace 未配置（OPENFACE_BIN）",
        }

    norm = Path(media["normalized_path"])
    duration = float(media["duration"])
    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            frames.mkdir()
            _extract_frames(norm, frames, fps=5.0)
            csv = Path(tmp) / "openface.csv"
            of.run_openface(frames, csv)
            df = _parse_openface(csv, fps=5.0)
            df = df[df["conf"] > 0.7].copy()
            if df.empty:
                write_parquet(au_path, pd.DataFrame(columns=empty_cols))
                return {
                    "ok": False,
                    "visual_unavailable": True,
                    "face_cov": 0.0,
                    "degrade": ["visual"],
                    "reason": "无高置信度人脸",
                }

            df["face_missing"] = False
            df["face_id"] = 0
            full_ts = np.arange(0.0, duration, 1.0 / 5.0).round(3)
            df = df.drop_duplicates("ts").set_index("ts")
            full = pd.DataFrame(index=full_ts)
            full.index.name = "ts"
            merged = full.join(df, how="left")
            merged["face_missing"] = merged["conf"].isna()
            merged["face_id"] = 0
            merged = merged.reset_index()
    except Exception as e:
        log.exception("S1A OpenFace 失败 session=%s", session_id)
        write_parquet(au_path, pd.DataFrame(columns=empty_cols))
        return {
            "ok": False,
            "visual_unavailable": True,
            "face_cov": 0.0,
            "degrade": ["visual"],
            "reason": f"OpenFace 执行失败: {e}",
        }

    face_cov = float((~merged["face_missing"]).sum()) / max(len(merged), 1)
    write_parquet(au_path, merged)
    unavailable = face_cov < 0.2
    report: StageReport = {
        "ok": not unavailable,
        "visual_unavailable": unavailable,
        "face_cov": round(face_cov, 4),
        "degrade": ["visual"] if unavailable else [],
    }
    if unavailable:
        report["reason"] = f"face_cov={face_cov:.2f} < 0.2"
    log.info("S1A done session=%s face_cov=%.3f", session_id, face_cov)
    return report


def _extract_frames(video: Path, out_dir: Path, fps: float) -> None:
    from ..engine import ffmpeg as ff

    bin_ = ff._resolve_ffmpeg()
    if not bin_:
        raise RuntimeError("ffmpeg 不可用")
    subprocess.run(
        [bin_, "-y", "-i", str(video), "-vf", f"fps={fps}", str(out_dir / "frame_%06d.png")],
        capture_output=True,
        text=True,
        check=True,
    )


def _parse_openface(csv: Path, fps: float) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["ts"] = (df.index / fps).round(3)
    df["conf"] = df["confidence"].astype(float)
    keep = ["ts", "conf", *AU_COLS, *POSE_COLS]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()
