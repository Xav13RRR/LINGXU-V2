import threading

from lingxu.voice.consent import ConsentRegistry


def test_consent_is_self_contained_and_idempotent() -> None:
    registry = ConsentRegistry()

    assert registry.grant(101) is True
    assert registry.grant(101) is False
    assert registry.snapshot() == frozenset({101})
    assert registry.withdraw(101) is True
    assert registry.withdraw(101) is False


def test_withdrawal_linearises_with_receive_action() -> None:
    registry = ConsentRegistry()
    registry.grant(7)
    entered = threading.Event()
    release = threading.Event()
    accepted: list[int] = []

    def action() -> None:
        entered.set()
        assert release.wait(timeout=2)
        accepted.append(7)

    receive = threading.Thread(target=registry.perform_if_consented, args=(7, action))
    receive.start()
    assert entered.wait(timeout=2)

    withdrew: list[bool] = []
    withdrawal = threading.Thread(target=lambda: withdrew.append(registry.withdraw(7)))
    withdrawal.start()
    assert withdrawal.is_alive()
    release.set()
    receive.join(timeout=2)
    withdrawal.join(timeout=2)

    assert accepted == [7]
    assert withdrew == [True]
    assert registry.perform_if_consented(7, lambda: None) == (False, None)
