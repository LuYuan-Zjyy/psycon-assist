"""S6 画像与跨次：来访者（subject）基线，读 + 写。

comparability < 0.7 不做跨次比较结论（硬规则 24）。
Agent 优化走 Prompt + RAG（不微调）。
"""
from __future__ import annotations

import numpy as np

from ..core.io import read_json, read_json_optional, write_json
from ..core.logging_utils import get_logger
from ..core.models import QualityReport
from ..core.paths import SubjectPaths

log = get_logger(__name__)


def get_profile(subject_id: str) -> dict | None:
    return read_json_optional(SubjectPaths(subject_id).profile_json)


def update_profile(
    subject_id: str,
    session_id: str,
    chunks: list[dict],
    quality: QualityReport | dict,
) -> dict:
    """每场结束（S5 后）更新画像。幂等：同一 session 重复跑不重复计数。"""
    sp = SubjectPaths(subject_id)
    sp.ensure_layout()
    p = get_profile(subject_id) or {
        "subject_id": subject_id,
        "n_sessions": 0,
        "sessions": [],
        "baseline": {"v_mean": None, "v_sd": None, "a_mean": None, "a_sd": None},
        "dynamics": {"inertia_alpha": None},
        "comparability": 0.0,
        "exemplars_count": 0,
    }

    if session_id in p["sessions"]:
        return p  # 幂等：已计入，直接返回

    vs = [c["va"]["v"] for c in chunks if c.get("va") and c["va"].get("v") is not None]
    as_ = [c["va"]["a"] for c in chunks if c.get("va") and c["va"].get("a") is not None]
    new_base = {
        "v_mean": float(np.mean(vs)) if vs else None,
        "v_sd": float(np.std(vs)) if vs else None,
        "a_mean": float(np.mean(as_)) if as_ else None,
        "a_sd": float(np.std(as_)) if as_ else None,
    }

    prev_n = p["n_sessions"]
    if prev_n == 0:
        p["baseline"] = new_base
    else:
        for k in ("v_mean", "v_sd", "a_mean", "a_sd"):
            prev = p["baseline"].get(k)
            cur = new_base.get(k)
            if cur is None:
                continue
            p["baseline"][k] = cur if prev is None else (prev * prev_n + cur) / (prev_n + 1)

    p["n_sessions"] += 1
    p["sessions"].append(session_id)
    p["exemplars_count"] = _count_exemplars(subject_id)
    p["comparability"] = _comparability(p["n_sessions"], quality)
    p["last_updated"] = session_id

    write_json(sp.profile_json, p)
    log.info(
        "S6 profile subject=%s n_sessions=%s comparability=%s",
        subject_id,
        p["n_sessions"],
        p["comparability"],
    )
    return p


def _comparability(n_sessions: int, quality: dict) -> float:
    """质量驱动的可比性；首场无历史，恒为 0。"""
    if n_sessions <= 1:
        return 0.0
    score = 0.0
    n = 0
    if "face_cov" in quality and quality["face_cov"] is not None:
        score += quality["face_cov"]
        n += 1
    if "asr_ok" in quality:
        score += 1.0 if quality["asr_ok"] else 0.0
        n += 1
    if "diarization_ok" in quality:
        score += 1.0 if quality["diarization_ok"] else 0.0
        n += 1
    return round(score / n, 4) if n else 0.0


def _count_exemplars(subject_id: str) -> int:
    edir = SubjectPaths(subject_id).exemplars_dir
    if not edir.exists():
        return 0
    return len(list(edir.glob("*.json")))


def load_exemplars(subject_id: str) -> list[dict]:
    edir = SubjectPaths(subject_id).exemplars_dir
    if not edir.exists():
        return []
    return [read_json(f) for f in sorted(edir.glob("*.json"))]


def profile_summary(profile: dict | None) -> str:
    """画像摘要（短，注入 S4 生成约束）。"""
    if not profile:
        return ""
    b = profile.get("baseline", {})
    return (
        f"该来访者已会谈 {profile.get('n_sessions', 0)} 次，"
        f"历史效价均值 {b.get('v_mean')}，可比性 {profile.get('comparability')}，"
        f"历史采纳/改写示例 {profile.get('exemplars_count')} 条。"
    )
