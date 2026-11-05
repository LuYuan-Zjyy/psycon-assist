"""Claims must only mention modalities that actually produced evidence."""
from psyconassist.stages.s4_agent import _template_claim


def _ev(metric, value=0.0):
    return {"eid": {"metric": metric, "value": value}}


def test_arousal_only_never_mentions_face():
    e = {"type": "va_shift", "t0": 90, "t1": 120, "evidence_ids": ["eid"]}
    text = _template_claim(e, _ev("va.a"), role_known=False)
    assert "面部" not in text and "效价" not in text
    assert "唤醒" in text


def test_face_and_arousal_mention_both():
    e = {"type": "va_shift", "t0": 90, "t1": 120, "evidence_ids": ["eid", "eid2"]}
    evidence = {"eid": {"metric": "va.v", "value": -0.5}, "eid2": {"metric": "va.a", "value": 1.2}}
    text = _template_claim(e, evidence, role_known=True)
    assert "效价" in text and "唤醒" in text


def test_text_shift_wording():
    e = {"type": "text_shift", "t0": 0, "t1": 30, "evidence_ids": ["eid"]}
    text = _template_claim(e, _ev("polarity"), role_known=False)
    assert "文本情绪" in text
    assert "面部" not in text


def test_silence_uses_neutral_wording_without_diarization():
    e = {"type": "silence", "t0": 0, "t1": 30, "evidence_ids": ["eid"]}
    assert "来访者" not in _template_claim(e, _ev("polarity"), role_known=False)
    assert "来访者" in _template_claim(e, _ev("polarity"), role_known=True)
