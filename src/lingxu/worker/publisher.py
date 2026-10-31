"""Discord publication boundary for the independent Worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import aiohttp


class ArtifactPublisher(Protocol):
    async def publish(
        self,
        *,
        guild_id: int,
        channel_id: int | None,
        content: str,
        files: tuple[Path, ...],
    ) -> tuple[int, str]:
        """Publish once and return the selected channel and external message id."""


class DiscordArtifactPublisher:
    """Publish deterministic artifacts through Discord's documented HTTP API."""

    _BASE_URL = "https://discord.com/api/v10"

    def __init__(self, token: str, *, timeout_seconds: float = 30.0) -> None:
        if not token:
            raise ValueError("Discord token is required for Worker publication")
        self._token = token
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def _select_channel(
        self,
        session: aiohttp.ClientSession,
        *,
        guild_id: int,
        requested: int | None,
    ) -> int:
        if requested is not None:
            return requested
        async with session.get(f"{self._BASE_URL}/guilds/{guild_id}") as response:
            response.raise_for_status()
            payload = await response.json()
        system_channel_id = payload.get("system_channel_id")
        if system_channel_id is None:
            raise RuntimeError("Meeting has no publication channel and guild has no system channel")
        return int(system_channel_id)

    async def publish(
        self,
        *,
        guild_id: int,
        channel_id: int | None,
        content: str,
        files: tuple[Path, ...],
    ) -> tuple[int, str]:
        headers = {"Authorization": f"Bot {self._token}"}
        async with aiohttp.ClientSession(headers=headers, timeout=self._timeout) as session:
            selected_channel = await self._select_channel(
                session,
                guild_id=guild_id,
                requested=channel_id,
            )
            form = aiohttp.FormData()
            attachments = [{"id": index, "filename": path.name} for index, path in enumerate(files)]
            form.add_field(
                "payload_json",
                json.dumps(
                    {"content": content[:2000], "attachments": attachments},
                    ensure_ascii=False,
                ),
                content_type="application/json",
            )
            handles = [path.open("rb") for path in files]
            try:
                for index, (path, handle) in enumerate(zip(files, handles, strict=True)):
                    form.add_field(
                        f"files[{index}]",
                        handle,
                        filename=path.name,
                        content_type="application/octet-stream",
                    )
                async with session.post(
                    f"{self._BASE_URL}/channels/{selected_channel}/messages",
                    data=form,
                ) as response:
                    response.raise_for_status()
                    payload = await response.json()
            finally:
                for handle in handles:
                    handle.close()
        return selected_channel, str(payload["id"])
