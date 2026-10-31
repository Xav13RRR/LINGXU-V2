from lingxu.logging import redact_known_secret_values, redact_sensitive_fields


def test_sensitive_keys_and_binary_payloads_are_redacted() -> None:
    result = redact_sensitive_fields(
        None,
        "event",
        {
            "discord_token": "token-value",
            "nested": {"privacy_code": "1234"},
            "payload": b"audio",
        },
    )

    assert result["discord_token"] == "[REDACTED]"
    assert result["nested"] == {"privacy_code": "[REDACTED]"}
    assert result["payload"] == "<bytes:5>"


def test_configured_secret_value_is_removed_from_safe_key() -> None:
    result = redact_known_secret_values(
        None,
        "event",
        {"error": "request rejected for abcdef"},
        secrets=("abcdef",),
    )

    assert result["error"] == "request rejected for [REDACTED]"
