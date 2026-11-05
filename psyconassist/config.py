"""全局配置与路径。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根 = 本文件上两级（xmh/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# 国内机房常无法直连 huggingface.co；未显式配置时走镜像
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if not os.getenv("HF_HOME") and Path("/public_data/hf_cache").exists():
    os.environ["HF_HOME"] = "/public_data/hf_cache"
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/public_data/hf_cache/hub")

DATA_ROOT = Path(os.getenv("PSY_DATA_ROOT", PROJECT_ROOT / "data"))
SUBJECTS_DIR = DATA_ROOT / "subjects"

# ---- 大模型 ----
LLM_API_KEY = os.getenv("DS_API_KEY", "") or os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ---- 重模型引擎 ----
OPENFACE_BIN = os.getenv("OPENFACE_BIN", "")  # FeatureExtraction.exe 绝对路径
HF_TOKEN = os.getenv("HF_TOKEN", "")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "")      # ffmpeg.exe 绝对路径（可空，自动探测）
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "")    # ffprobe.exe 绝对路径（可空，自动探测）

# ---- 参数 ----
MERGE_GAP = 2.0          # 异常段最小间隔
CHUNK_MIN = 4.0          # chunk 最短
CHUNK_MAX = 45.0         # chunk 最长
GRID_SEC = 30.0          # 固定网格
GRID_OVERLAP = 0.15
BASELINE_WINDOW = 8 * 60  # 前 8 分钟
BASELINE_MIN = 3 * 60     # 基线最小可用时长
COMPARABILITY_THRESHOLD = 0.7


def session_dir(session_id: str) -> Path:
    return DATA_ROOT / session_id


def subject_dir(subject_id: str) -> Path:
    return SUBJECTS_DIR / subject_id
