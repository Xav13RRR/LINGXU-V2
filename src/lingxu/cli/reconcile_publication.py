"""Content-safe explicit reconciliation for a Discord publication receipt."""

from __future__ import annotations

import argparse
import asyncio
import re
from typing import Never

from lingxu.config import Settings
from lingxu.storage.database import Database
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.worker_repositories import (
    ArtifactRepository,
    PublicationReconciliationRequired,
)

_ULID = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}\Z")


class _ArgumentFailure(Exception):
    pass


class _ContentSafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise _ArgumentFailure


def parse_operator_ids(session_id: str, channel_id: str, message_id: str) -> tuple[str, int, str]:
    """Validate opaque operator-supplied identifiers without loading meeting content."""

    if _ULID.fullmatch(session_id) is None:
        raise ValueError("session_id must be a ULID")
    if not channel_id.isdecimal() or int(channel_id) <= 0:
        raise ValueError("channel_id must be a positive decimal identifier")
    if not message_id.isdecimal() or int(message_id) <= 0:
        raise ValueError("message_id must be a positive decimal identifier")
    return session_id, int(channel_id), message_id


async def _reconcile(session_id: str, channel_id: int, message_id: str) -> None:
    settings = Settings()
    database = Database.open(settings.database_url)
    try:
        await ArtifactRepository(
            database.sessions, session_files=SessionFiles(settings.data_root)
        ).finalize_discord_publication(
            session_id=session_id,
            selected_channel_id=channel_id,
            message_id=message_id,
        )
    finally:
        await database.close()


def main() -> int:
    parser = _ContentSafeParser(
        description="Finalize one existing PREPARED publication from an operator-verified receipt."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--channel-id", required=True)
    parser.add_argument("--message-id", required=True)
    try:
        args = parser.parse_args()
        session_id, channel_id, message_id = parse_operator_ids(
            args.session_id, args.channel_id, args.message_id
        )
        asyncio.run(_reconcile(session_id, channel_id, message_id))
    except _ArgumentFailure:
        print('{"status":"manual_reconciliation_required","reason":"invalid_arguments"}')
        return 2
    except (ValueError, PublicationReconciliationRequired) as error:
        reason = (
            error.reason if isinstance(error, PublicationReconciliationRequired) else "invalid_ids"
        )
        print(f'{{"status":"manual_reconciliation_required","reason":"{reason}"}}')
        return 2
    except Exception:
        print('{"status":"manual_reconciliation_required","reason":"internal_error"}')
        return 2
    print('{"status":"finalized"}')
    return 0
