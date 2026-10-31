from __future__ import annotations

import pytest

from lingxu.cli import reconcile_publication
from lingxu.cli.reconcile_publication import parse_operator_ids


def test_manual_reconciliation_accepts_only_exact_content_free_ids() -> None:
    assert parse_operator_ids("01HZX0G7A0D8M9N2R4T6V8WXYZ", "8100", "1001") == (
        "01HZX0G7A0D8M9N2R4T6V8WXYZ",
        8100,
        "1001",
    )
    with pytest.raises(ValueError):
        parse_operator_ids("not-a-session", "8100", "1001")
    with pytest.raises(ValueError):
        parse_operator_ids("01HZX0G7A0D8M9N2R4T6V8WXYZ", "0", "1001")
    with pytest.raises(ValueError):
        parse_operator_ids("01HZX0G7A0D8M9N2R4T6V8WXYZ", "8100", "message")


def test_manual_reconciliation_masks_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fail_without_leaking(*args: object) -> None:
        raise RuntimeError(f"private identifiers: {args!r}")

    monkeypatch.setattr(reconcile_publication, "_reconcile", fail_without_leaking)
    monkeypatch.setattr(
        "sys.argv",
        [
            "reconcile_discord_publication.py",
            "--session-id",
            "01HZX0G7A0D8M9N2R4T6V8WXYZ",
            "--channel-id",
            "8100",
            "--message-id",
            "1001",
        ],
    )

    assert reconcile_publication.main() == 2
    assert (
        capsys.readouterr().out
        == '{"status":"manual_reconciliation_required","reason":"internal_error"}\n'
    )


@pytest.mark.parametrize(
    "arguments",
    (
        ("--unknown-secret=do-not-echo",),
        ("--session-id", "sensitive-looking-value", "--channel-id", "not-a-number"),
    ),
)
def test_cli_parse_errors_never_echo_raw_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, ...],
) -> None:
    monkeypatch.setattr("sys.argv", ["reconcile_discord_publication.py", *arguments])

    assert reconcile_publication.main() == 2
    captured = capsys.readouterr()
    assert captured.out == (
        '{"status":"manual_reconciliation_required","reason":"invalid_arguments"}\n'
    )
    assert captured.err == ""
    assert all(argument not in captured.out for argument in arguments)
