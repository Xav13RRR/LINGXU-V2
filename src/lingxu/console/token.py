"""Reveal the local console login token only on an explicit operator command."""

from __future__ import annotations

from lingxu.config import Settings


def main() -> None:
    settings = Settings()
    token = settings.effective_host_console_token
    if not settings.host_console_enabled or token is None:
        raise SystemExit("Host console is not enabled or has no configured token")
    print(token)


if __name__ == "__main__":
    main()
