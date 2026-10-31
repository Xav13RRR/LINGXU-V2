"""Cooperative process shutdown helpers."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Iterable


class ShutdownSignal:
    """Expose SIGINT and SIGTERM as an awaitable event."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def install(self, signals: Iterable[signal.Signals] | None = None) -> None:
        """Register handlers on platforms that support event-loop signal callbacks."""

        loop = asyncio.get_running_loop()
        selected = tuple(signals or (signal.SIGINT, signal.SIGTERM))
        for signum in selected:
            try:
                loop.add_signal_handler(signum, self._event.set)
            except (NotImplementedError, RuntimeError):
                signal.signal(signum, lambda *_: loop.call_soon_threadsafe(self._event.set))

    def request(self) -> None:
        """Request shutdown programmatically."""

        self._event.set()

    async def wait(self) -> None:
        """Wait until shutdown is requested."""

        await self._event.wait()
