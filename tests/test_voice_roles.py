from psyconassist.stages.s2_chunk import _voice_va


def _seg(role, t0, t1, loud=1.0):
    return {"role": role, "t0": t0, "t1": t1, "loudness_rel": loud}


def test_unknown_role_still_counts_as_speech():
    cov_v, a, cov = _voice_va([_seg("unknown", 0, 10)], 0, 10)
    assert cov == 1.0 and a is not None


def test_no_speech_is_silence():
    cov_v, a, cov = _voice_va([], 0, 10)
    assert cov == 0.0


def test_other_roles_are_ignored_when_client_known():
    cov_v, a, cov = _voice_va([_seg("counselor", 0, 10)], 0, 10)
    assert cov == 0.0
