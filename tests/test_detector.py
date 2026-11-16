"""S2 检测器回归测试：基线窗口、z 分、五类异常、覆盖与 trigger。"""
from psyconassist.stages.s2_chunk import (
    _apply_zscore,
    _compute_baseline,
    _grid_to_chunks,
    _mark_anomalies,
    schemas,
)


def _grid(n, va_v=0.0, va_a=0.5, text_v=0.2, conf_voice=0.5, voice_present=True):
    step = 25.5
    out = []
    for i in range(n):
        out.append({
            "t0": round(i * step, 3),
            "t1": round(min((i + 1) * step, n * step), 3),
            "slot_t0": round(i * step, 3),
            "slot_t1": round(min((i + 1) * step, n * step), 3),
            "va_v": va_v(i) if callable(va_v) else va_v,
            "va_a": va_a(i) if callable(va_a) else va_a,
            "va_conf": 1.0,
            "conf_face": 1.0,
            "conf_voice": conf_voice(i) if callable(conf_voice) else conf_voice,
            "conf_text": 1.0,
            "text_v": text_v(i) if callable(text_v) else text_v,
            "voice_data_present": voice_present,
        })
    return out


def test_baseline_window_long_session():
    b = _compute_baseline(_grid(123))  # 52 分钟
    assert b["window"] == 16            # 8 分钟 / 30s，而不是 40%


def test_baseline_window_short_session():
    b = _compute_baseline(_grid(10))
    assert b["window"] == max(2, int(10 * 0.4))


def test_arousal_baseline_and_z():
    g = _grid(40, va_a=lambda i: 0.5 if i < 16 else 0.2)
    b = _compute_baseline(g)
    assert b["a_mean"] is not None and b["a_sd"] is not None
    g = _apply_zscore(g, b)
    assert any(x.get("z_a") is not None for x in g)


def test_va_shift_sustained_drop():
    g = _grid(60, va_v=lambda i: -0.4 if 30 <= i < 40 else 0.0)
    b = _compute_baseline(g)
    g = _apply_zscore(g, b)
    g = _mark_anomalies(g, [])
    flagged = {i for i, x in enumerate(g) if "va_shift" in x["anom"]}
    assert flagged == set(range(30, 40))


def test_low_activation():
    g = _grid(40, va_a=lambda i: 0.5 if i < 16 else 0.1)
    b = _compute_baseline(g)
    g = _apply_zscore(g, b)
    g = _mark_anomalies(g, [])
    flagged = {i for i, x in enumerate(g) if "low_activation" in x["anom"]}
    assert 25 in flagged


def test_silence_needs_voice_data():
    g = _mark_anomalies(_grid(30, conf_voice=0.05, voice_present=False), [])
    assert not any("silence" in x["anom"] for x in g)
    g2 = _mark_anomalies(_grid(30, conf_voice=0.05, voice_present=True), [])
    assert any("silence" in x["anom"] for x in g2)


def test_text_shift():
    g = _mark_anomalies(_grid(40, text_v=lambda i: -0.8 if 20 <= i < 30 else 0.2), [])
    flagged = {i for i, x in enumerate(g) if "text_shift" in x["anom"]}
    assert 25 in flagged


def test_chunks_cover_and_trigger():
    g = _grid(20, va_v=lambda i: -0.5 if 10 <= i < 14 else 0.0)
    b = _compute_baseline(g)
    g = _apply_zscore(g, b)
    g = _mark_anomalies(g, [])
    chunks = _grid_to_chunks("S20261102-001", g, g[-1]["t1"])
    schemas.check_covers(chunks, lambda c: c["t0"], lambda c: c["t1"], g[-1]["t1"])
    anom = [c for c in chunks if c["is_anomaly"]]
    assert anom
    assert all(c["trigger"].get("z_v") is not None for c in anom)
