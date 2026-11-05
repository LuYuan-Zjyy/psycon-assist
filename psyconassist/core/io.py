"""落盘与读取：阶段产物只通过文件通信。

约定：
- JSON 一律 UTF-8
- 写文件先写临时文件再 replace，避免半截文件
- 缺文件用 read_*_optional，不要到处 .exists() 散落
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from .errors import PsyConError
from .logging_utils import get_logger

log = get_logger(__name__)
T = TypeVar("T")


class IOError_(PsyConError):
    """读写失败。"""


def write_json(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    _atomic_write_text(path, data)
    return path


def read_json(path: Path) -> Any:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise IOError_(f"JSON 不存在: {path}") from e
    except json.JSONDecodeError as e:
        raise IOError_(f"JSON 损坏: {path}: {e}") from e


def read_json_optional(path: Path, default: T | None = None) -> Any | T | None:
    path = Path(path)
    if not path.exists():
        return default
    return read_json(path)


def write_parquet(path: Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # parquet 本身较原子；先写临时再替换更稳
    fd, tmp_name = tempfile.mkstemp(suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except FileNotFoundError as e:
        raise IOError_(f"Parquet 不存在: {path}") from e


def read_parquet_optional(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return read_parquet(path)


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_name).replace(path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
