"""S2 融合与异常切片。

把三路信号融合成逐网格 VA 轨迹，用异常检测切出 chunk（异常段 = 事件候选）。
异常 = 相对来访者自身基线的统计偏差，可解释、无监督。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import config, ids, schemas
from ..core.io import read_json_optional, read_parquet_optional, write_json
from ..core.logging_utils import get_logger
from ..core.models import MediaInfo, StageReport
from ..core.paths import SessionPaths

log = get_logger(__name__)


def run(session_id: str, media: MediaInfo, degrade: list[str] | None = None) -> StageReport:
    paths = SessionPaths(session_id)
    duration = float(media["duration"])
    degrade = degrade or []

    au = read_parquet_optional(paths.au_parquet)
    voice = read_json_optional(paths.voice_json, default=[]) or []
    text = _load_text(paths)

    grid = _build_grid(duration, au, voice, text)
    baseline = _compute_baseline(grid)
    grid = _apply_zscore(grid, baseline)
    grid = _mark_anomalies(grid, text)

    chunks = _grid_to_chunks(session_id, grid, duration)
    schemas.check_covers(chunks, lambda c: c["t0"], lambda c: c["t1"], duration)
    write_json(paths.chunks_json, chunks)

    report: StageReport = {
        "ok": True,
        "n_chunks": len(chunks),
        "n_anomaly": sum(1 for c in chunks if c["is_anomaly"]),
        "baseline_available": baseline is not None,
        "degrade": list(degrade),
    }
    log.info(
        "S2 done session=%s chunks=%s anomaly=%s baseline=%s",
        session_id,
        report["n_chunks"],
        report["n_anomaly"],
        report["baseline_available"],
    )
    return report


def _load_text(paths: SessionPaths) -> list:
    """加载文本特征；兼容旧版缺 t0/t1 的 text_feat，用 transcript 补齐。"""
    feat = read_json_optional(paths.text_feat_json, default=[]) or []
    if not feat:
        return []
    if all(("t0" in x and "t1" in x) for x in feat):
        return feat
    transcript = read_json_optional(paths.transcript_json, default=[]) or []
    by_id = {u.get("utt_id"): u for u in transcript}
    for x in feat:
        u = by_id.get(x.get("utt_id")) or {}
        x.setdefault("t0", u.get("t0", 0.0))
        x.setdefault("t1", u.get("t1", 0.0))
        x.setdefault("role", u.get("role"))
        x.setdefault("text", u.get("text"))
    return feat


def _build_grid(duration: float, au, voice, text) -> list[dict]:
    """网格窗口用于计算特征（30s，重叠 15%）；slot 用于确定 chunk 边界（无重叠）。"""
    step = config.GRID_SEC * (1 - config.GRID_OVERLAP)
    n = max(1, int(math.ceil(duration / step)))
    grid = []
    for i in range(n):
        t0 = round(i * step, 3)
        t1 = round(min(t0 + config.GRID_SEC, duration), 3)
        g = {"t0": t0, "t1": t1, "slot_t0": t0, "slot_t1": round(min((i + 1) * step, duration), 3)}
        fv, fa, fc = _face_va(au, t0, t1)
        vv, va_, vc = _voice_va(voice, t0, t1)
        tv, ta, tc = _text_va(text, t0, t1)
        g["face_v"], g["face_a"], g["conf_face"] = fv, fa, fc
        g["voice_v"], g["voice_a"], g["conf_voice"] = vv, va_, vc
        g["text_v"], g["text_a"], g["conf_text"] = tv, ta, tc
        g.update(_fuse(g))
        grid.append(g)
    return grid


def _face_va(au, t0: float, t1: float):
    if au is None or au.empty:
        return None, None, 0.0
    seg = au[(au["ts"] >= t0) & (au["ts"] < t1)]
    if seg.empty:
        return None, None, 0.0
    cov = float((~seg["face_missing"]).sum()) / max(len(seg), 1)
    valid = seg[~seg["face_missing"]]
    if valid.empty:
        return None, None, cov

    def au_mean(c):
        return valid[c].mean() if c in valid.columns else float("nan")

    au12 = au_mean("AU12_r")
    au04 = au_mean("AU04_r")
    v = None
    if not (math.isnan(au12) or math.isnan(au04)):
        v = max(-1.0, min(1.0, (au12 - au04) / 8.0))
    au_cols = [c for c in valid.columns if c.startswith("AU") and c.endswith("_r")]
    a = None
    if au_cols:
        m = valid[au_cols].abs().mean().mean()
        if not math.isnan(m):
            a = max(0.0, min(1.0, m / 5.0)) * 2 - 1
    return v, a, cov


def _voice_va(voice: list, t0: float, t1: float):
    if not voice:
        return None, None, 0.0
    client = [s for s in voice if s.get("role") == "client" and s["t1"] > t0 and s["t0"] < t1]
    if not client:
        return None, None, 0.0
    cov = sum(min(t1, s["t1"]) - max(t0, s["t0"]) for s in client) / max(t1 - t0, 1e-9)
    cov = max(0.0, min(1.0, cov))
    louds = [s["loudness_rel"] for s in client if s.get("loudness_rel") is not None]
    a = float(np.mean(louds)) if louds else None
    return None, a, cov


def _text_va(text: list, t0: float, t1: float):
    if not text:
        return None, None, 0.0
    seg = [s for s in text if s.get("t1", 0) > t0 and s.get("t0", 0) < t1]
    if not seg:
        return None, None, 0.0
    pols = [s["polarity"] for s in seg if s.get("polarity") is not None]
    cov = min(1.0, len(seg) / max(1, (t1 - t0) / 15))
    if not pols:
        return None, None, cov
    v = float(np.mean(pols))
    a = abs(v) * 2 - 1
    return v, a, cov


def _fuse(g: dict) -> dict:
    w_face = 0.5 * g["conf_face"]
    w_voice = 0.3 * g["conf_voice"]
    w_text = 0.2 * g["conf_text"]
    W = w_face + w_voice + w_text
    # 视觉缺失时放宽，避免文本/语音永远融不进 VA
    min_w = 0.12 if g["conf_face"] < 0.05 else 0.3
    if W < min_w:
        return {"va_v": None, "va_a": None, "va_conf": round(W, 3)}
    v_num = v_den = 0.0
    a_num = a_den = 0.0
    for (vv, aa, w) in [(g["face_v"], g["face_a"], w_face),
                        (g["voice_v"], g["voice_a"], w_voice),
                        (g["text_v"], g["text_a"], w_text)]:
        if vv is not None:
            v_num += w * vv
            v_den += w
        if aa is not None:
            a_num += w * aa
            a_den += w
    v = v_num / v_den if v_den > 0 else None
    a = a_num / a_den if a_den > 0 else None
    return {"va_v": v, "va_a": a, "va_conf": round(W, 3)}


def _compute_baseline(grid: list[dict]) -> dict | None:
    n = len(grid)
    win = max(1, min(n, int(config.BASELINE_WINDOW / config.GRID_SEC)))
    # 短会谈：至少用前 40% 网格，避免 win 被压成 1
    win = max(win, max(2, int(math.ceil(n * 0.4))))
    win = min(win, n)
    seg = grid[:win]
    vs = [g["va_v"] for g in seg if g.get("va_v") is not None]
    min_n = 2 if n <= 12 else max(3, int(config.BASELINE_MIN / config.GRID_SEC))
    if len(vs) < min_n:
        return None
    mean = float(np.mean(vs))
    std = float(np.std(vs))
    if std < 1e-9:
        std = 0.05  # 平坦轨迹也给一个地板，便于相对偏移
    keep = [v for v in vs if abs(v - mean) / std <= 3.0]
    if len(keep) < min_n:
        return None
    return {"v_mean": float(np.mean(keep)), "v_sd": max(float(np.std(keep)), 0.05),
            "window": win, "n": len(keep)}


def _apply_zscore(grid: list[dict], baseline: dict | None) -> list[dict]:
    if baseline is None or baseline["v_sd"] < 1e-9:
        for g in grid:
            g["z_v"] = None
            g["z_a"] = None
        return grid
    for g in grid:
        g["z_v"] = (g["va_v"] - baseline["v_mean"]) / baseline["v_sd"] if g.get("va_v") is not None else None
        g["z_a"] = None
    return grid


def _mark_anomalies(grid: list[dict], text: list) -> list[dict]:
    n = len(grid)
    # 会话内相对激活（无脸/短视频也能出 low_activation）
    acts = [g.get("va_a") for g in grid if g.get("va_a") is not None]
    act_mean = float(np.mean(acts)) if acts else None
    act_sd = float(np.std(acts)) if acts and len(acts) >= 2 else None
    if act_sd is not None and act_sd < 1e-6:
        act_sd = 0.05
    pols = [g.get("text_v") for g in grid if g.get("text_v") is not None]
    pol_mean = float(np.mean(pols)) if pols else None
    pol_sd = float(np.std(pols)) if pols and len(pols) >= 2 else None
    if pol_sd is not None and pol_sd < 1e-6:
        pol_sd = 0.05

    for g in grid:
        g["risk"] = _risk_in_window(text, g["t0"], g["t1"])
        g["anom"] = []
        if g["risk"]:
            g["anom"].append("risk")
        if g.get("z_a") is not None and g["z_a"] < -1.5:
            g["anom"].append("low_activation")
        if act_mean is not None and act_sd is not None and g.get("va_a") is not None:
            if (g["va_a"] - act_mean) / act_sd < -1.2:
                g["anom"].append("low_activation")
        if pol_mean is not None and pol_sd is not None and g.get("text_v") is not None:
            # 相对本场更低落的文本段（哪怕绝对极性仍为正）
            if (g["text_v"] - pol_mean) / pol_sd < -1.2:
                g["anom"].append("va_shift")
        if g.get("text_v") is not None and g.get("va_v") is not None:
            if (g["text_v"] > 0.2 and g["va_v"] < -0.2) or (g["text_v"] < -0.2 and g["va_v"] > 0.2):
                g["anom"].append("incongruence")
    for i in range(n - 1):
        zi = grid[i].get("z_v")
        zj = grid[i + 1].get("z_v")
        if zi is not None and zj is not None and abs(zi) > 2.0 and abs(zj) > 2.0:
            grid[i]["anom"].append("va_shift")
            grid[i + 1]["anom"].append("va_shift")
    for g in grid:
        # 来访者几乎不说话 → silence；无语音覆盖也算
        if g.get("conf_voice") is not None and g["conf_voice"] < 0.12:
            g["anom"].append("silence")
    for g in grid:
        g["anom"] = list(dict.fromkeys(g["anom"]))
    return grid


def _risk_in_window(text: list, t0: float, t1: float) -> bool:
    return any(s.get("risk_hits") and s.get("t0", 0) < t1 and s.get("t1", 0) > t0 for s in text)


# ---------------------------------------------------------------------------
def _grid_to_chunks(session_id: str, grid: list[dict], duration: float) -> list[dict]:
    """把网格的异常标记转成连续、覆盖 [0,duration] 的 chunk（用 slot 边界）。"""
    n = len(grid)
    flags = [bool(g["anom"]) for g in grid]

    # 标记每格类型集合
    type_flags = [set(g["anom"]) for g in grid]

    # 连续异常段
    segs = []  # (start_idx, end_idx_exclusive, types)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            types = set()
            while j < n and flags[j]:
                types.update(type_flags[j])
                j += 1
            segs.append([i, j, types])
            i = j
        else:
            i += 1

    segs = _merge_close(segs, grid)

    # 构造 chunk：异常段 + 之间/两侧的正常段
    raw = []  # (start_idx, end_idx, is_anomaly, types)
    cursor = 0
    for s0, s1, types in segs:
        if s0 > cursor:
            raw.append((cursor, s0, False, set()))
        raw.append((s0, s1, True, types))
        cursor = s1
    if cursor < n:
        raw.append((cursor, n, False, set()))

    # 切分超长段（按 max_len，用 slot 边界切）
    chunks = []
    for s0, s1, is_anom, types in raw:
        t0 = grid[s0]["slot_t0"]
        t1 = grid[s1 - 1]["slot_t1"]
        seg_len = t1 - t0
        if seg_len <= config.CHUNK_MAX:
            chunks.append(_make_chunk(session_id, grid, s0, s1, is_anom, types))
            continue
        # 切分超长：按 max_len 步进，在 slot 边界切
        cur = s0
        cur_t = t0
        while cur < s1:
            target_t = cur_t + config.CHUNK_MAX
            j = cur
            while j < s1 and grid[j]["slot_t1"] <= target_t + 1e-9:
                j += 1
            j = max(j, cur + 1)
            j = min(j, s1)
            chunks.append(_make_chunk(session_id, grid, cur, j, is_anom, types))
            cur_t = grid[j - 1]["slot_t1"]
            cur = j

    # 对齐词边界（单调）+ 强制覆盖 + 重新编号
    chunks = _finalize_chunks(session_id, chunks, duration)
    return chunks


def _merge_close(segs: list, grid: list) -> list:
    if not segs:
        return segs
    out = [list(segs[0])]
    for s in segs[1:]:
        gap = grid[s[0]]["slot_t0"] - grid[out[-1][1] - 1]["slot_t1"]
        if gap < config.MERGE_GAP:
            out[-1][1] = s[1]
            out[-1][2].update(s[2])
        else:
            out.append(list(s))
    return out


def _pick_type(types: set) -> str:
    order = ["risk", "va_shift", "silence", "incongruence", "low_activation"]
    for t in order:
        if t in types:
            return t
    return "normal"


def _make_chunk(session_id, grid, s0, s1, is_anom, types) -> dict:
    t0 = grid[s0]["slot_t0"]
    t1 = grid[s1 - 1]["slot_t1"]
    atype = _pick_type(types) if is_anom else "normal"
    va_v = _mean([g.get("va_v") for g in grid[s0:s1]])
    va_a = _mean([g.get("va_a") for g in grid[s0:s1]])
    conf = _mean([g.get("va_conf") for g in grid[s0:s1]])
    pols = [g.get("text_v") for g in grid[s0:s1] if g.get("text_v") is not None]
    missing = []
    if va_v is None and va_a is None:
        missing.append("va: 融合权重不足或输入缺失")
    va = {"v": va_v, "a": va_a, "conf": round(conf, 3) if conf is not None else None}
    return {
        "chunk_id": f"{session_id}-C{s0:04d}",
        "t0": t0, "t1": t1,
        "is_anomaly": is_anom,
        "anomaly_type": atype,
        "all_types": sorted(types) if types else [],
        "trigger": None,
        "va": va if (va_v is not None or va_a is not None) else None,
        "text": {"polarity": round(float(np.mean(pols)), 4) if pols else None},
        "missing": missing,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _finalize_chunks(session_id: str, chunks: list, duration: float) -> list:
    """内部边界单调对齐词边界（可选，词边界缺省时保持 slot），首尾强制 [0,duration]。"""
    # 用 chunk 之间共享边界作为对齐点
    internal = [c["t1"] for c in chunks[:-1]]
    # 无文本词边界信息时，slot 边界已经连续；这里保持原值即可
    prev = 0.0
    for i, c in enumerate(chunks):
        if i < len(chunks) - 1:
            t1 = internal[i]
        else:
            t1 = duration
        t0 = prev
        if t1 <= t0:
            t1 = t0 + 0.5  # 保底，避免零长度
        c["t0"] = round(t0, 3)
        c["t1"] = round(t1, 3)
        prev = t1
    chunks[0]["t0"] = 0.0
    chunks[-1]["t1"] = round(duration, 3)
    for i, c in enumerate(chunks):
        c["chunk_id"] = ids.chunk_id(session_id, i + 1)
    return chunks
