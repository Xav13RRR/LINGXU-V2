from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import CookieJar, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

from lingxu.console.auth import ConsoleSessionStore
from lingxu.console.server import HostConsoleServer
from lingxu.scenario.loader import load_scenario


class _FakeSnapshots:
    guild_id = 42

    def __init__(self) -> None:
        self.loaded = load_scenario(Path("scenarios/fixed_meeting_v1.yaml"))

    async def snapshot(self) -> dict[str, Any]:
        return {
            "active": True,
            "guild_id": "42",
            "generated_at": "2026-07-27T00:00:00+00:00",
            "session_id": "session-42",
            "host_user_id": "99",
            "state": "DISCUSSION",
            "sequence": 7,
            "agenda": [],
            "participants": [],
            "speaker": {"current_user_id": None, "queued_user_ids": []},
            "timers": [],
            "vote": {
                "anonymous": True,
                "options": [{"id": "yes", "label": "Yes", "count": 2}],
            },
            "final_utterances": [],
            "recent_events": [],
            "scenario": [],
        }

    async def active_scenario(self) -> Any:
        return SimpleNamespace(schema=self.loaded.scenario)


class _FakeController:
    def __init__(self) -> None:
        self.actions: list[tuple[str, dict[str, Any]]] = []

    async def active(self, guild_id: int) -> Any:
        assert guild_id == 42
        return SimpleNamespace(creator_user_id=99)

    async def _run(self, action: str, **kwargs: Any) -> None:
        self.actions.append((action, kwargs))

    async def pause(self, **kwargs: Any) -> None:
        await self._run("pause", **kwargs)

    async def resume(self, **kwargs: Any) -> None:
        await self._run("resume", **kwargs)

    async def next(self, **kwargs: Any) -> None:
        await self._run("next", **kwargs)

    async def back(self, **kwargs: Any) -> None:
        await self._run("back", **kwargs)

    async def skip(self, **kwargs: Any) -> None:
        await self._run("skip", **kwargs)

    async def stop(self, **kwargs: Any) -> None:
        await self._run("stop", **kwargs)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_console_auth_refresh_websocket_controls_tts_and_simulator() -> None:
    snapshots = _FakeSnapshots()
    controller = _FakeController()
    spoken: list[str] = []

    async def manual_tts(text: str) -> str:
        spoken.append(text)
        return "voice"

    console = HostConsoleServer(
        bind_host="127.0.0.1",
        port=0,
        session_store=ConsoleSessionStore("t" * 32),
        snapshots=snapshots,  # type: ignore[arg-type]
        controller=controller,  # type: ignore[arg-type]
        manual_tts=manual_tts,
    )
    client = TestClient(TestServer(console.app), cookie_jar=CookieJar(unsafe=True))
    await client.start_server()
    origin = str(client.make_url("/").origin())
    try:
        assert (await client.get("/api/snapshot")).status == 401
        metrics = await client.get("/metrics")
        metrics_text = await metrics.text()
        assert metrics.status == 200
        assert "lingxu_host_console_up 1" in metrics_text
        assert "guild_id" not in metrics_text
        assert "user_id" not in metrics_text
        with pytest.raises(WSServerHandshakeError) as websocket_error:
            await client.ws_connect("/ws", headers={"Origin": origin})
        assert websocket_error.value.status == 401

        invalid = await client.post(
            "/api/login",
            json={"token": "wrong"},
            headers={"Origin": origin},
        )
        assert invalid.status == 401

        login = await client.post(
            "/api/login",
            json={"token": "t" * 32},
            headers={"Origin": origin},
        )
        assert login.status == 200
        login_payload = await login.json()
        csrf = login_payload["csrf_token"]
        assert login_payload["guild_id"] == "42"

        restored = await client.get("/api/session")
        assert restored.status == 200
        assert (await restored.json())["csrf_token"] == csrf

        snapshot_response = await client.get("/api/snapshot")
        snapshot = await snapshot_response.json()
        assert snapshot["guild_id"] == "42"
        assert "user_id" not in snapshot["vote"]["options"][0]
        assert "voter_token" not in snapshot["vote"]["options"][0]

        rejected = await client.post("/api/control/pause", headers={"Origin": origin})
        assert rejected.status == 403
        cross_origin = await client.post(
            "/api/control/pause",
            headers={
                "Origin": "https://attacker.invalid",
                "X-Lingxu-CSRF": csrf,
            },
        )
        assert cross_origin.status == 403

        control = await client.post(
            "/api/control/pause",
            headers={"Origin": origin, "X-Lingxu-CSRF": csrf},
        )
        assert control.status == 200
        assert controller.actions == [
            (
                "pause",
                {
                    "guild_id": 42,
                    "actor_user_id": 99,
                    "source": "HOST_CONSOLE",
                },
            )
        ]

        tts = await client.post(
            "/api/tts",
            json={"text": "  本地   播报  "},
            headers={"Origin": origin, "X-Lingxu-CSRF": csrf},
        )
        assert await tts.json() == {"ok": True, "mode": "voice"}
        assert spoken == ["本地 播报"]

        reset = await client.post(
            "/api/simulator/reset",
            headers={"Origin": origin, "X-Lingxu-CSRF": csrf},
        )
        reset_payload = await reset.json()
        event = reset_payload["simulator"]["possible_events"][0]
        dispatched = await client.post(
            "/api/simulator/dispatch",
            json={"event": event},
            headers={"Origin": origin, "X-Lingxu-CSRF": csrf},
        )
        assert (await dispatched.json())["simulator"]["sequence"] == 1

        websocket = await client.ws_connect("/ws", headers={"Origin": origin})
        websocket_snapshot = await websocket.receive_json(timeout=2)
        assert websocket_snapshot["guild_id"] == "42"
        assert websocket_snapshot["simulator"]["sequence"] == 1
        await websocket.close()

        logout = await client.post(
            "/api/logout",
            headers={"Origin": origin, "X-Lingxu-CSRF": csrf},
        )
        assert logout.status == 200
        assert (await client.get("/api/session")).status == 401
    finally:
        await client.close()
