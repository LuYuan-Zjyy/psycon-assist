"""全局配置与路径。"""
from __future__ import annotations

import os
import re
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

# ---- 上传限制 ----
MAX_UPLOAD_MB = int(os.getenv("PSY_MAX_UPLOAD_MB", "2048"))

# ---- ID 校验（防止路径穿越：所有落盘路径都必须经过 session_dir / subject_dir）----
# session id 由代码生成，格式固定；subject id 由用户输入（P-0001 / CI-0001 / subj_01），
# 只限制为安全的路径片段：字母数字开头，仅含字母/数字/下划线/连字符，长度 <= 32，
# 从而杜绝 "/"、"\\"、"."、".."、空格 等穿越字符。
_SESSION_ID_RE = re.compile(r"^S\d{8}-\d{3}$")
_SUBJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def session_dir(session_id: str) -> Path:
    if not _SESSION_ID_RE.match(session_id or ""):
        raise ValueError(f"invalid session_id: {session_id!r}")
    return DATA_ROOT / session_id


def subject_dir(subject_id: str) -> Path:
    if not _SUBJECT_ID_RE.match(subject_id or ""):
        raise ValueError(f"invalid subject_id: {subject_id!r}")
    return SUBJECTS_DIR / subject_id
