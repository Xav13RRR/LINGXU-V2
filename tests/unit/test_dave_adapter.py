from types import SimpleNamespace

from lingxu.voice.dave_adapter import inspect_dave_status, receive_fork_supports_dave


def test_pinned_receive_fork_exposes_dave_parser() -> None:
    assert receive_fork_supports_dave() is True


def test_dave_status_reports_booleans_without_exposing_code() -> None:
    connection = SimpleNamespace(
        dave_protocol_version=1,
        dave_session=SimpleNamespace(ready=True),
    )
    client = SimpleNamespace(_connection=connection, voice_privacy_code="sensitive")

    status = inspect_dave_status(client)

    assert status.enabled is True
    assert status.ready is True
    assert status.protocol_version == 1
    assert status.privacy_code_available is True
    assert "sensitive" not in repr(status)


def test_dave_status_fails_closed_when_session_is_absent() -> None:
    connection = SimpleNamespace(dave_protocol_version=1, dave_session=None)

    status = inspect_dave_status(SimpleNamespace(_connection=connection, voice_privacy_code=None))

    assert status.enabled is True
    assert status.ready is False
