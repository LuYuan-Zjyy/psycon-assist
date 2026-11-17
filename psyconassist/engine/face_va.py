"""S1A 视觉（Linux/GPU）：人脸检测 + EmotiEffLib 直接回归 VA 与离散表情。

替代 Windows 版 OpenFace。依赖（可选，缺则 visual 降级）：
    pip install emotiefflib facenet-pytorch

- 人脸检测用 facenet-pytorch 的 MTCNN（权重内置、无需联网下载；
  insightface 的 buffalo_l 需从 GitHub 下载，在受限网络不可用）。
- VA 用 EmotiEffLib 的 mbf_va_mtl（MobileFaceNet，最轻）：predict_emotions 输出
  10 维（8 表情 logits + valence + arousal），取末两位。
- 离散表情用 enet_b0_8_best_vgaf（8 类）。
- 推理用 ONNX 引擎（避免 torch checkpoint 的 mobilefacenet pickle 依赖问题）。
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

    from emotiefflib.facial_analysis import EmotiEffLibRecognizerOnnx
    from facenet_pytorch import MTCNN

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 人脸检测走 MTCNN（GPU 更快）；情感识别走 ONNX：
    # EmotiEffLib 的 torch checkpoint 依赖顶层 mobilefacenet 模块，pip 包未随装，
    # 直接 load 会 ModuleNotFoundError；ONNX 无此问题，且这些模型很小，CPU 推理已足够。
    mt = MTCNN(keep_all=True, device=device, post_process=False)
    # mbf_va_mtl 一次前向同时给出 8 类表情 logits + valence + arousal，
    # 因此不再单独加载第二个表情模型（省一次推理）。
    va_rec = EmotiEffLibRecognizerOnnx(model_name=VA_MODEL)

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
                "x0": None, "y0": None, "x1": None, "y1": None,
                "expr_ent": None, "expr_dist": None,
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
                        # mbf_va_mtl 输出 10 维：前 8 是离散表情 logits，后 2 是 valence/arousal
                        _, va_scores = va_rec.predict_emotions(crop)
                        va_scores = np.asarray(va_scores, dtype=float)
                        row["va_v"] = float(va_scores[0][8])
                        row["va_a"] = float(va_scores[0][9])
                        # 前 8 维是表情 logits：softmax 得到完整分布，而不是只取 argmax
                        logits = va_scores[0][:8]
                        e = np.exp(logits - logits.max())
                        p = e / e.sum()
                        ek = int(np.argmax(p))  # 注意：不能复用 k，k 是人脸框索引
                        row["expr"] = EMOTIONS[ek]
                        row["expr_prob"] = float(p[ek])
                        row["expr_ent"] = float(-(p * np.log(p + 1e-9)).sum())
                        row["expr_dist"] = [round(float(x), 4) for x in p]
                        row["conf"] = float(probs[k])
                        row["face_missing"] = False
                        row["x0"], row["y0"], row["x1"], row["y1"] = x0, y0, x1, y1
            except Exception:
                pass  # 单帧失败仅记 missing
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["ts", "face_missing", "conf", "va_v", "va_a", "expr", "expr_prob",
                                   "expr_ent", "expr_dist", "x0", "y0", "x1", "y1"])
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
