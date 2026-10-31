"""Content-free health probe for the private CosyVoice service."""

from __future__ import annotations

import urllib.request


def main() -> None:
    with urllib.request.urlopen("http://127.0.0.1:50000/healthz", timeout=5) as response:
        if response.status != 200 or response.read() != b'{"ok":true}':
            raise SystemExit(1)


if __name__ == "__main__":
    main()
