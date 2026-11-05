"""ID validation must accept real ids and reject path traversal."""
import pytest

from psyconassist import config


@pytest.mark.parametrize("sid", ["S20261105-001", "S20260101-999"])
def test_valid_session_ids(sid):
    assert config.session_dir(sid).name == sid


@pytest.mark.parametrize("sid", ["../../etc", "S2026-1", "..\\..\\x", "", "S20261105_001"])
def test_invalid_session_ids(sid):
    with pytest.raises(ValueError):
        config.session_dir(sid)


@pytest.mark.parametrize("subject", ["P-0001", "CI-0001", "subj_01", "A1", "P0001"])
def test_valid_subject_ids(subject):
    assert config.subject_dir(subject).name == subject


@pytest.mark.parametrize(
    "subject",
    ["../../x", "P;rm -rf", "", "..", "a/b", "a.b", "x" * 40, "-lead", "with space", "a\\b"],
)
def test_invalid_subject_ids(subject):
    with pytest.raises(ValueError):
        config.subject_dir(subject)
