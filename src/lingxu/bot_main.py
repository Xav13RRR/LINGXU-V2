"""Primary independent Lingxu Discord Bot entry point."""

from __future__ import annotations

import asyncio
import contextlib

import structlog

from lingxu.bot.client import create_bot
from lingxu.config import Settings
from lingxu.lifecycle import ShutdownSignal
from lingxu.logging import configure_logging
from lingxu.storage.migrate import require_database_current
from lingxu.voice.segment_recorder import cleanup_stale_recording_plaintext


async def run_bot(settings: Settings) -> None:
    """Close active probes and Discord resources on SIGINT or SIGTERM."""

    stale_plaintext_removed = cleanup_stale_recording_plaintext(settings.runtime_tmp)
    if stale_plaintext_removed:
        structlog.get_logger(__name__).info(
            "recording_plaintext_startup_cleanup",
            files_removed=stale_plaintext_removed,
        )
    if not settings.has_discord_credentials:
        raise ValueError("independent Discord credentials are required")
    await asyncio.to_thread(require_database_current, settings.database_url)
    token = settings.discord_token
    assert token is not None
    bot = create_bot(settings)
    shutdown = ShutdownSignal()
    shutdown.install()
    bot_task = asyncio.create_task(
        bot.start(token.get_secret_value()),
        name="discord-bot",
    )
    shutdown_task = asyncio.create_task(shutdown.wait(), name="discord-shutdown")
    try:
        done, _ = await asyncio.wait(
            (bot_task, shutdown_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if bot_task in done:
            await bot_task
        else:
            await bot.close()
            await bot_task
    finally:
        shutdown_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await shutdown_task
        if not bot.is_closed():
            await bot.close()


def main() -> None:
    """Load validated settings and run the official Bot."""

    settings = Settings()
    configure_logging(settings.log_level, secrets=settings.configured_secret_values)
    log = structlog.get_logger(__name__)
    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        log.info("bot_interrupted")
    except ValueError as exc:
        log.error("discord_credentials_missing", error_type=type(exc).__name__)
        raise SystemExit(2) from exc
    except Exception as exc:
        log.error("bot_stopped", error_type=type(exc).__name__)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
