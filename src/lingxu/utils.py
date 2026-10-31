"""Small dependency-free utilities."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid(
    *,
    timestamp_ms: int | None = None,
    random_bytes: Callable[[int], bytes] = os.urandom,
) -> str:
    """Generate a 26-character ULID suitable for probe directories."""

    milliseconds = int(time.time() * 1_000) if timestamp_ms is None else timestamp_ms
    if milliseconds < 0 or milliseconds >= 2**48:
        raise ValueError("timestamp_ms must fit in 48 bits")
    randomness = random_bytes(10)
    if len(randomness) != 10:
        raise ValueError("random_bytes must return exactly 10 bytes")
    value = (milliseconds << 80) | int.from_bytes(randomness, "big")
    encoded = ["0"] * 26
    for index in range(25, -1, -1):
        encoded[index] = _CROCKFORD[value & 0x1F]
        value >>= 5
    return "".join(encoded)
