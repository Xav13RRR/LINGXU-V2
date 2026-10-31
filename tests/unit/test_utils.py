import pytest

from lingxu.utils import new_ulid


def test_new_ulid_is_stable_with_injected_inputs() -> None:
    value = new_ulid(timestamp_ms=0, random_bytes=lambda length: b"\0" * length)

    assert value == "0" * 26


def test_new_ulid_rejects_wrong_random_length() -> None:
    with pytest.raises(ValueError, match="exactly 10 bytes"):
        new_ulid(timestamp_ms=0, random_bytes=lambda length: b"\0")
