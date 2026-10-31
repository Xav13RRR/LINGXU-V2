from pathlib import Path
from types import SimpleNamespace
from typing import Any

import discord
import pytest

from lingxu.bot.client import (
    LingxuBot,
    _mentions_discord_user,
    _read_live_attachment,
)
from lingxu.config import Settings


class _HTTPResponse:
    status = 404
    reason = "Not Found"


def _http_error() -> discord.HTTPException:
    return discord.HTTPException(_HTTPResponse(), {"code": 0, "message": "not found"})  # type: ignore[arg-type]


class _Attachment:
    url = "https://cdn.discord.test/source"
    proxy_url = "https://proxy.discord.test/source"

    def __init__(self, results: list[bytes | Exception]) -> None:
        self._results = iter(results)
        self.calls: list[bool] = []

    async def read(self, *, use_cached: bool = False) -> bytes:
        self.calls.append(use_cached)
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_markdown_import_uses_guild_messages_without_privileged_content(
    tmp_path: Path,
) -> None:
    bot = LingxuBot(
        Settings(
            _env_file=None,
            discord_application_id=123,
            database_url=(f"sqlite+aiosqlite:///{(tmp_path / 'message-intents.db').as_posix()}"),
            data_root=tmp_path / "sessions",
            runtime_tmp=tmp_path / "runtime",
            tts_provider="cosyvoice-sft",
            tts_cache_dir=tmp_path / "tts",
        )
    )
    try:
        assert bot.intents.guild_messages
        assert not bot.intents.message_content
        assert not bot.intents.members
        assert callable(bot.on_message)
        assert bot.tree.allowed_contexts.guild is True
        assert bot.tree.allowed_contexts.dm_channel is False
        assert bot.tree.allowed_contexts.private_channel is False
        assert bot.tree.allowed_installs.guild is True
        assert bot.tree.allowed_installs.user is False
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_owner_approved_resident_assistant_is_wired_into_bot(tmp_path: Path) -> None:
    bot = LingxuBot(
        Settings(
            _env_file=None,
            discord_application_id=123,
            database_url=(f"sqlite+aiosqlite:///{(tmp_path / 'resident-bot.db').as_posix()}"),
            data_root=tmp_path / "sessions",
            runtime_tmp=tmp_path / "runtime",
            tts_provider="cosyvoice-sft",
            tts_cache_dir=tmp_path / "tts",
            resident_assistant_enabled=True,
            meeting_assistant_llm_model_digest="a" * 64,
        )
    )
    try:
        assert bot._resident_assistant is not None
        await bot.setup_hook()
        assert {command.name for command in bot.tree.get_commands()} == {
            "assistant",
            "meeting",
            "wakeup",
        }
    finally:
        await bot.close()


def test_markdown_import_matches_mentions_by_user_id_not_names() -> None:
    renamed_mentions: list[Any] = [
        SimpleNamespace(id=456, name="聆序Lingxu", display_name="频道昵称一"),
    ]

    assert _mentions_discord_user(renamed_mentions, 456)
    renamed_mentions[0].name = "A completely different global name"
    renamed_mentions[0].display_name = "另一个服务器昵称"
    assert _mentions_discord_user(renamed_mentions, 456)
    assert not _mentions_discord_user(renamed_mentions, 789)


@pytest.mark.asyncio
async def test_markdown_import_prefers_live_attachment_url() -> None:
    attachment = _Attachment([b"agenda"])

    payload = await _read_live_attachment(attachment)  # type: ignore[arg-type]

    assert payload == b"agenda"
    assert attachment.calls == [False]


@pytest.mark.asyncio
async def test_markdown_import_falls_back_to_cached_attachment_url() -> None:
    attachment = _Attachment([_http_error(), b"agenda"])

    payload = await _read_live_attachment(attachment)  # type: ignore[arg-type]

    assert payload == b"agenda"
    assert attachment.calls == [False, True]


@pytest.mark.asyncio
async def test_markdown_import_reports_both_attachment_urls_failed() -> None:
    attachment = _Attachment([_http_error(), _http_error()])

    with pytest.raises(discord.HTTPException):
        await _read_live_attachment(attachment)  # type: ignore[arg-type]

    assert attachment.calls == [False, True]
