"""ID 规范（见主文档 §9.1）。

| 对象     | 格式                 | 例                    |
| session  | S + yyyymmdd + -序号  | S20261102-001          |
| chunk    | 会话 + -C + 4 位      | S20261102-001-C0007    |
| event    | 会话 + -E + 3 位      | S20261102-001-E003     |
| evidence | 会话 + -EV + 6 位     | S20261102-001-EV000012 |
| claim    | 会话 + -K + 4 位      | S20261102-001-K0007    |
| feedback | 会话 + -F + 4 位      | S20261102-001-F0021    |
| subject  | P + 4 位              | P-0031                 |
"""
from __future__ import annotations

import datetime


def session_id(seq: int, dt: datetime.datetime | None = None) -> str:
    dt = dt or datetime.datetime.now()
    return f"S{dt.strftime('%Y%m%d')}-{seq:03d}"


def chunk_id(session: str, seq: int) -> str:
    return f"{session}-C{seq:04d}"


def event_id(session: str, seq: int) -> str:
    return f"{session}-E{seq:03d}"


def evidence_id(session: str, seq: int) -> str:
    return f"{session}-EV{seq:06d}"


def claim_id(session: str, seq: int) -> str:
    return f"{session}-K{seq:04d}"


def feedback_id(session: str, seq: int) -> str:
    return f"{session}-F{seq:04d}"


def subject_id(seq: int) -> str:
    return f"P-{seq:04d}"
