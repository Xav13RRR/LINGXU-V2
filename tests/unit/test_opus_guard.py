from lingxu.voice.opus_guard import (
    is_opus_decode_error,
    safe_error_label,
    safe_receive_diagnostics,
)


class FakeOpusError(Exception):
    pass


class FakeClient:
    def get_recv_diagnostics(self) -> dict[str, object]:
        return {
            "opus_decode_ok": 50,
            "opus_decode_err": 1,
            "secret_packet": "do not expose",
            "dave_inner_decrypt_ok": 42,
            "dave_key": 123,
            "bool_value": True,
        }


def test_error_classification_never_returns_message() -> None:
    error = FakeOpusError("raw packet details")

    assert is_opus_decode_error(error) is True
    assert safe_error_label(error) == "FakeOpusError"
    assert "raw packet" not in safe_error_label(error)


def test_receive_diagnostics_are_allowlisted_integers() -> None:
    assert safe_receive_diagnostics(FakeClient()) == {
        "opus_decode_ok": 50,
        "opus_decode_err": 1,
        "dave_inner_decrypt_ok": 42,
    }
