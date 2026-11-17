"""S1A 视觉（Linux/GPU）：人脸检测 + EmotiEffLib 直接回归 VA 与离散表情。

替代 Windows 版 OpenFace。依赖（可选，缺则 visual 降级）：
    pip install emotiefflib facenet-pytorch

- 人脸检测用 facenet-pytorch 的 MTCNN（权重内置、无需联网下载；
  insightface 的 buffalo_l 需从 GitHub 下载，在受限网络不可用）。
- VA 用 EmotiEffLib 的 mbf_va_mtl（MobileFaceNet，最轻），直接输出 valence/arousal。
- 离散表情用 enet_b0_8_best_vgaf（8 类）。
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["available", "extract_va"]

VA_MODEL = "mbf_va_mtl"
EXPR_MODEL = "enet_b0_8_best_vgaf"
EMOTIONS = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]


def available() -> bool:
    try:
        import emotiefflib  # noqa: F401
        import facenet_pytorch  # noqa: F401
        return True
    except Exception:
        return False


def extract_va(
    video_path: Path,
    duration: float,
    fps: float = 5.0,
    device: str = "cuda",
) -> pd.DataFrame:
    """逐采样帧估计 VA + 离散表情。

    返回 DataFrame 列：ts, face_missing, conf, va_v, va_a, expr, expr_prob。
    无人脸帧 face_missing=True、其余为 None（不补 0）。
    """
    import torch
    from PIL import Image

    from emotiefflib.facial_analysis import EmotiEffLibRecognizerTorch
    from facenet_pytorch import MTCNN

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    mt = MTCNN(keep_all=True, device=device, post_process=False)
    va_rec = EmotiEffLibRecognizerTorch(model_name=VA_MODEL, device=device)
    expr_rec = EmotiEffLibRecognizerTorch(model_name=EXPR_MODEL, device=device)

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir()
        _extract_frames(video_path, frames_dir, fps)
        files = sorted(frames_dir.glob("frame_*.png"))
        rows = []
        for i, fp in enumerate(files):
            ts = round(i / fps, 3)
            row = {
                "ts": ts, "face_missing": True, "conf": None,
                "va_v": None, "va_a": None, "expr": None, "expr_prob": None,
            }
            try:
                img = np.asarray(Image.open(fp).convert("RGB"))
                boxes, probs = mt.detect(Image.fromarray(img))
                if boxes is not None and len(boxes) > 0:
                    k = int(np.argmax(probs))
                    x0, y0, x1, y1 = [int(v) for v in boxes[k]]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
                    if x1 - x0 > 20 and y1 - y0 > 20:
                        crop = img[y0:y1, x0:x1]
                        feat = va_rec.extract_features(crop)
                        row["va_v"] = float(feat[0])
                        row["va_a"] = float(feat[1])
                        labels, scores = expr_rec.predict_emotions(crop)
                        row["expr"] = labels[0] if isinstance(labels, (list, tuple)) else labels
                        row["expr_prob"] = float(np.max(scores))
                        row["conf"] = float(probs[k])
                        row["face_missing"] = False
            except Exception:
                pass  # 单帧失败仅记 missing
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["ts", "face_missing", "conf", "va_v", "va_a", "expr", "expr_prob"])
    return df


def _extract_frames(video: Path, out_dir: Path, fps: float) -> None:
    from ..engine import ffmpeg as ff

    bin_ = ff._resolve_ffmpeg()
    if not bin_:
        raise RuntimeError("ffmpeg 不可用")
    subprocess.run(
        [bin_, "-y", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "3",
         str(out_dir / "frame_%06d.png")],
        capture_output=True, text=True, check=True,
    )
