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
    transcript = read_json_optional(paths.transcript_json, default=[]) or []
    word_ends = sorted({w["e"] for u in transcript for w in (u.get("words") or [])})

    grid = _build_grid(duration, au, voice, text)
    baseline = _compute_baseline(grid)
    grid = _apply_zscore(grid, baseline)
    grid = _mark_anomalies(grid, text)

    chunks = _grid_to_chunks(session_id, grid, duration, word_ends)
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
        g["voice_data_present"] = bool(voice)
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
    # 未做说话人分离时 role=unknown：此时把全部语音当作同一说话人，
    # 否则会误判为整场沉默（claim 措辞仍会标明角色未知）
    client = [s for s in voice if s.get("role") in ("client", "unknown") and s["t1"] > t0 and s["t0"] < t1]
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
    """基线 = 本场前段「非极端」窗口的均值/标准差（效价 + 唤醒）。

    长会谈用固定 8 分钟；短会谈（8 分钟超过会话 40%）退回前 40%，至少 2 格。
    对均值做 3σ 截断，避免把开场异常也算进「正常」。
    """
    n = len(grid)
    win = min(n, int(config.BASELINE_WINDOW / config.GRID_SEC))
    pct_win = min(n, max(2, int(math.ceil(n * 0.4))))
    if pct_win < win:  # 短会谈：8 分钟超过 40%，退回 40%
        win = pct_win
    seg = grid[:win]
    min_n = 2 if n <= 12 else max(3, int(config.BASELINE_MIN / config.GRID_SEC))

    def _stats(values: list) -> tuple[float, float] | None:
        vals = [v for v in values if v is not None]
        if len(vals) < min_n:
            return None
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std < 1e-9:
            std = 0.05
        keep = [v for v in vals if abs(v - mean) / std <= 3.0]
        if len(keep) < min_n:
            return None
        return float(np.mean(keep)), max(float(np.std(keep)), 0.05)

    v = _stats([g.get("va_v") for g in seg])
    a = _stats([g.get("va_a") for g in seg])
    if v is None and a is None:
        return None
    return {
        "v_mean": v[0] if v else None,
        "v_sd": v[1] if v else None,
        "a_mean": a[0] if a else None,
        "a_sd": a[1] if a else None,
        "window": win,
        "n": len(seg),
    }


def _apply_zscore(grid: list[dict], baseline: dict | None) -> list[dict]:
    vsd = baseline.get("v_sd") if baseline else None
    asd = baseline.get("a_sd") if baseline else None
    vmean = baseline.get("v_mean") if baseline else None
    amean = baseline.get("a_mean") if baseline else None
    for g in grid:
        g["z_v"] = (
            (g["va_v"] - vmean) / vsd
            if vsd and vmean is not None and g.get("va_v") is not None
            else None
        )
        g["z_a"] = (
            (g["va_a"] - amean) / asd
            if asd and amean is not None and g.get("va_a") is not None
            else None
        )
    return grid


def _mark_anomalies(grid: list[dict], text: list) -> list[dict]:
    """相对本场基线的统计偏差；类型由规则决定，不调 LLM。

    六类：risk / va_shift / low_activation / text_shift / incongruence / silence。
    - va_shift：效价 z 分连续 |z|>2 的持续偏离（相对基线，不是相邻格都极端）
    - low_activation：唤醒 z 分 < -1.5（无基线时退回会话内相对激活兜底）
    - silence：仅在有语音数据时，来访者发声占比过低才判；无音频不判
    """
    n = len(grid)
    for g in grid:
        g["risk"] = _risk_in_window(text, g["t0"], g["t1"])
        g["anom"] = ["risk"] if g["risk"] else []

    # low_activation：优先用基线 z_a；无基线时退回会话内相对激活
    acts = [g.get("va_a") for g in grid if g.get("va_a") is not None]
    act_mean = float(np.mean(acts)) if acts else None
    act_sd = float(np.std(acts)) if acts and len(acts) >= 2 else None
    if act_sd is not None and act_sd < 1e-6:
        act_sd = 0.05
    for g in grid:
        if g.get("z_a") is not None and g["z_a"] < -1.5:
            g["anom"].append("low_activation")
        elif act_mean is not None and act_sd is not None and g.get("va_a") is not None:
            if (g["va_a"] - act_mean) / act_sd < -1.5:
                g["anom"].append("low_activation")

    # text_shift：文本极性相对本场均值走低（独立于面部，避免无脸时伪造视觉结论）
    pols = [g.get("text_v") for g in grid if g.get("text_v") is not None]
    pol_mean = float(np.mean(pols)) if pols else None
    pol_sd = float(np.std(pols)) if pols and len(pols) >= 2 else None
    if pol_sd is not None and pol_sd < 1e-6:
        pol_sd = 0.05
    for g in grid:
        if pol_mean is not None and pol_sd is not None and g.get("text_v") is not None:
            if (g["text_v"] - pol_mean) / pol_sd < -1.2:
                g["anom"].append("text_shift")

    # incongruence：文本极性与效价符号相反
    for g in grid:
        if g.get("text_v") is not None and g.get("va_v") is not None:
            if (g["text_v"] > 0.2 and g["va_v"] < -0.2) or (g["text_v"] < -0.2 and g["va_v"] > 0.2):
                g["anom"].append("incongruence")

    # va_shift：效价相对基线的持续大幅偏离（连续 |z_v|>2 的 run）
    i = 0
    while i < n:
        if grid[i].get("z_v") is not None and abs(grid[i]["z_v"]) > 2.0:
            j = i
            while j < n and grid[j].get("z_v") is not None and abs(grid[j]["z_v"]) > 2.0:
                j += 1
            for k in range(i, j):
                grid[k]["anom"].append("va_shift")
            i = j
        else:
            i += 1

    # silence：有语音数据但来访者发声占比过低
    for g in grid:
        if g.get("voice_data_present") and g.get("conf_voice") is not None and g["conf_voice"] < 0.12:
            g["anom"].append("silence")

    for g in grid:
        g["anom"] = list(dict.fromkeys(g["anom"]))
    return grid


def _risk_in_window(text: list, t0: float, t1: float) -> bool:
    return any(s.get("risk_hits") and s.get("t0", 0) < t1 and s.get("t1", 0) > t0 for s in text)


# ---------------------------------------------------------------------------
def _grid_to_chunks(session_id: str, grid: list[dict], duration: float, word_ends: list[float] | None = None) -> list[dict]:
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
    chunks = _finalize_chunks(session_id, chunks, duration, word_ends)
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
    order = ["risk", "va_shift", "text_shift", "silence", "incongruence", "low_activation"]
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
    zs = [g.get("z_v") for g in grid[s0:s1] if g.get("z_v") is not None]
    zas = [g.get("z_a") for g in grid[s0:s1] if g.get("z_a") is not None]
    trigger = {
        "z_v": round(float(np.mean(zs)), 3) if zs else None,
        "z_a": round(float(np.mean(zas)), 3) if zas else None,
    }
    va = {"v": va_v, "a": va_a, "conf": round(conf, 3) if conf is not None else None}
    return {
        "chunk_id": f"{session_id}-C{s0:04d}",
        "t0": t0, "t1": t1,
        "is_anomaly": is_anom,
        "anomaly_type": atype,
        "all_types": sorted(types) if types else [],
        "trigger": trigger,
        "va": va if (va_v is not None or va_a is not None) else None,
        "text": {"polarity": round(float(np.mean(pols)), 4) if pols else None},
        "missing": missing,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _finalize_chunks(
    session_id: str,
    chunks: list,
    duration: float,
    word_ends: list[float] | None = None,
) -> list:
    """把内部边界单调对齐到最近词边界，首尾强制 [0, duration]。

    对齐只允许把边界「向前」挪到它之前最近的词尾（间距 <= 半个网格），
    因此绝不会产生重叠或空洞；末块终点固定在 duration。无词边界时保持 slot 边界。
    """
    internal = [c["t1"] for c in chunks[:-1]]
    if word_ends:
        ws = sorted(word_ends)
        max_dist = config.GRID_SEC * 0.5
        snapped: list[float] = []
        prev = 0.0
        for t in internal:
            cands = [w for w in ws if w < t and t - w <= max_dist and w > prev]
            t2 = max(cands) if cands else t
            if t2 <= prev:
                t2 = t
            snapped.append(t2)
            prev = t2
        internal = snapped

    prev = 0.0
    for i, c in enumerate(chunks):
        t1 = internal[i] if i < len(chunks) - 1 else duration
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
