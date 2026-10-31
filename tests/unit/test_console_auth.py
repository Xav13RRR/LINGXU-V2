from lingxu.console.auth import ConsoleSessionStore


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def test_console_session_uses_constant_secret_exchange_and_expiry() -> None:
    clock = _Clock()
    sessions = ConsoleSessionStore("s" * 32, clock=clock)

    assert sessions.login("wrong") is None
    session = sessions.login("s" * 32)
    assert session is not None
    assert session.id != "s" * 32
    assert session.csrf_token != session.id

    clock.value += 8 * 60 * 60 + 1
    sessions._purge()
    assert session.id not in sessions._sessions
