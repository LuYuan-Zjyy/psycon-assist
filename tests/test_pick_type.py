from psyconassist.stages.s2_chunk import _pick_type


def test_text_shift_is_not_reported_as_va_shift():
    assert _pick_type({"text_shift"}) == "text_shift"


def test_risk_wins_over_everything():
    assert _pick_type({"risk", "text_shift", "low_activation"}) == "risk"


def test_va_shift_still_wins_over_text_shift():
    assert _pick_type({"va_shift", "text_shift"}) == "va_shift"
