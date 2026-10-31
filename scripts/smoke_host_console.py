"""Authenticated deployed Host console smoke with synchronous cleanup."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import aiohttp

from lingxu.config import Settings
from lingxu.meeting.factory import create_meeting_application
from lingxu.storage.session_files import SessionFiles

_HOST_ID = 9_300_000_000_000_001


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} response shape is invalid")
    return cast(dict[str, object], value)


async def _request_json(
    session: aiohttp.ClientSession,
    method: str,
    path: str,
    *,
    csrf: str | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    headers = {"Origin": "http://127.0.0.1:8765"}
    if csrf is not None:
        headers["X-Lingxu-CSRF"] = csrf
    async with session.request(method, path, headers=headers, json=payload) as response:
        response.raise_for_status()
        value = await response.json()
        if not isinstance(value, dict):
            raise TypeError("console response must be an object")
        return value


async def smoke() -> dict[str, object]:
    settings = Settings()
    guild_id = settings.effective_host_console_guild_id
    token = settings.effective_host_console_token
    if not settings.host_console_enabled or guild_id is None or token is None:
        raise RuntimeError("deployed Host console is not fully configured")
    application = create_meeting_application(settings)
    existing = await application.sessions.active_for_guild(guild_id)
    if existing is not None:
        await application.database.close()
        raise RuntimeError("Host console smoke requires no active production Meeting")

    created = await application.controller.create(
        guild_id=guild_id,
        host_user_id=_HOST_ID,
    )
    session_id = created.id
    cleaned = False
    try:
        await application.controller.agenda_add(
            guild_id=guild_id,
            actor_user_id=_HOST_ID,
            title="Phase 13 console smoke",
            duration_minutes=1,
        )
        await application.controller.start(
            guild_id=guild_id,
            actor_user_id=_HOST_ID,
        )

        jar = aiohttp.CookieJar(unsafe=True)
        async with aiohttp.ClientSession(
            base_url="http://127.0.0.1:8765",
            cookie_jar=jar,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as client:
            login = await _request_json(
                client,
                "POST",
                "/api/login",
                payload={"token": token},
            )
            csrf = str(login["csrf_token"])
            restored = await _request_json(client, "GET", "/api/session")
            initial = await _request_json(client, "GET", "/api/snapshot")

            paused = await _request_json(
                client,
                "POST",
                "/api/control/pause",
                csrf=csrf,
            )
            resumed = await _request_json(
                client,
                "POST",
                "/api/control/resume",
                csrf=csrf,
            )
            advanced = await _request_json(
                client,
                "POST",
                "/api/control/next",
                csrf=csrf,
            )
            reset = await _request_json(
                client,
                "POST",
                "/api/simulator/reset",
                csrf=csrf,
            )
            simulator = reset["simulator"]
            if not isinstance(simulator, dict):
                raise TypeError("simulator snapshot is absent")
            possible = simulator["possible_events"]
            if not isinstance(possible, list) or not possible:
                raise RuntimeError("simulator has no initial event")
            dispatched = await _request_json(
                client,
                "POST",
                "/api/simulator/dispatch",
                csrf=csrf,
                payload={"event": str(possible[0])},
            )

            async with client.ws_connect(
                "/ws",
                headers={"Origin": "http://127.0.0.1:8765"},
                heartbeat=10,
            ) as websocket:
                websocket_snapshot = await websocket.receive_json(timeout=5)

            pause_snapshot = _mapping(paused["snapshot"], name="pause snapshot")
            resume_snapshot = _mapping(resumed["snapshot"], name="resume snapshot")
            advance_snapshot = _mapping(advanced["snapshot"], name="advance snapshot")
            dispatched_simulator = _mapping(
                dispatched["simulator"],
                name="dispatched simulator",
            )
            websocket_snapshot = _mapping(websocket_snapshot, name="websocket snapshot")

            result = {
                "authenticated": login.get("authenticated") is True,
                "refresh_restored": restored.get("authenticated") is True,
                "guild_scoped": initial.get("guild_id") == str(guild_id)
                and websocket_snapshot.get("guild_id") == str(guild_id),
                "pause_state": pause_snapshot.get("state"),
                "resume_state": resume_snapshot.get("state"),
                "advanced_state": advance_snapshot.get("state"),
                "simulator_sequence": dispatched_simulator.get("sequence"),
                "websocket_authorized": websocket_snapshot.get("session_id") == session_id,
            }
    finally:
        await application.timers.cancel(session_id)
        await application.sessions.mark_deleted(session_id, actor_user_id=_HOST_ID)
        SessionFiles(settings.data_root).delete(session_id)
        await application.sessions.hard_delete(session_id)
        cleaned = (await application.sessions.active_for_guild(guild_id)) is None
        await application.database.close()
    result["cleaned"] = cleaned
    return result


def main() -> None:
    result = asyncio.run(smoke())
    print(json.dumps(result, sort_keys=True))
    if not (
        result["authenticated"]
        and result["refresh_restored"]
        and result["guild_scoped"]
        and result["pause_state"] == "PAUSED"
        and result["resume_state"] != "PAUSED"
        and result["simulator_sequence"] == 1
        and result["websocket_authorized"]
        and result["cleaned"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
