"""Authenticated local HTTP/WebSocket Host console."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import structlog
from aiohttp import WSMsgType, web
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lingxu.console.auth import COOKIE_NAME, ConsoleSession, ConsoleSessionStore
from lingxu.console.snapshot import ConsoleSnapshotStore
from lingxu.domain.errors import LingxuError
from lingxu.meeting.controller import MeetingController
from lingxu.scenario.permissions import ScenarioRole
from lingxu.scenario.simulator import ScenarioSimulator

_STATIC_ROOT = Path(__file__).with_name("static")
_CONTROL_ACTIONS = frozenset({"pause", "resume", "next", "back", "skip", "stop"})
_SESSION_COOKIE_SECONDS = 8 * 60 * 60


class _LoginPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1, max_length=4_096)


class _TTSPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=300)


class _SimulationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,79}$")


@web.middleware
async def _security_headers(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    response = await handler(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "connect-src 'self' ws://127.0.0.1:* ws://localhost:*; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'; "
        "form-action 'self'"
    )
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


class HostConsoleServer:
    """Expose one configured Guild to a localhost-only authenticated console."""

    def __init__(
        self,
        *,
        bind_host: str,
        port: int,
        session_store: ConsoleSessionStore,
        snapshots: ConsoleSnapshotStore,
        controller: MeetingController,
        manual_tts: Callable[[str], Awaitable[str]],
    ) -> None:
        self._bind_host = bind_host
        self._port = port
        self._sessions = session_store
        self._snapshots = snapshots
        self._controller = controller
        self._manual_tts = manual_tts
        self._simulator: ScenarioSimulator | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._websockets: set[web.WebSocketResponse] = set()
        self._log = structlog.get_logger(__name__)

        app = web.Application(
            middlewares=[_security_headers],
            client_max_size=16 * 1_024,
        )
        app.add_routes(
            [
                web.get("/", self._index),
                web.get("/app.js", self._javascript),
                web.get("/styles.css", self._stylesheet),
                web.get("/healthz", self._health),
                web.get("/metrics", self._metrics),
                web.post("/api/login", self._login),
                web.get("/api/session", self._session_status),
                web.post("/api/logout", self._logout),
                web.get("/api/snapshot", self._snapshot),
                web.post("/api/control/{action}", self._control),
                web.post("/api/tts", self._tts),
                web.post("/api/simulator/reset", self._simulator_reset),
                web.post("/api/simulator/dispatch", self._simulator_dispatch),
                web.get("/ws", self._websocket),
            ]
        )
        self._app = app

    @property
    def app(self) -> web.Application:
        """Expose the aiohttp application for isolated integration tests."""

        return self._app

    async def start(self) -> None:
        if self._runner is not None:
            raise RuntimeError("Host console is already running")
        runner = web.AppRunner(self._app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self._bind_host, self._port)
        await site.start()
        self._runner = runner
        self._site = site
        self._log.info(
            "host_console_started",
            bind_host=self._bind_host,
            port=self._port,
            guild_id=self._snapshots.guild_id,
        )

    async def stop(self) -> None:
        for websocket in tuple(self._websockets):
            await websocket.close(code=1001, message=b"server shutdown")
        self._websockets.clear()
        runner = self._runner
        self._runner = None
        self._site = None
        if runner is not None:
            await runner.cleanup()

    @staticmethod
    async def _parse(request: web.Request, model: type[BaseModel]) -> BaseModel:
        try:
            payload = await request.json()
            return model.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            raise web.HTTPBadRequest(text="invalid request payload") from exc

    def _require_session(
        self,
        request: web.Request,
        *,
        csrf: bool = False,
    ) -> ConsoleSession:
        session = self._sessions.authenticate(request)
        if session is None:
            raise web.HTTPUnauthorized(text="authentication required")
        if csrf:
            self._sessions.require_same_origin(request)
            self._sessions.require_csrf(request, session)
        return session

    @staticmethod
    async def _index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(_STATIC_ROOT / "index.html")

    @staticmethod
    async def _javascript(request: web.Request) -> web.FileResponse:
        return web.FileResponse(_STATIC_ROOT / "app.js")

    @staticmethod
    async def _stylesheet(request: web.Request) -> web.FileResponse:
        return web.FileResponse(_STATIC_ROOT / "styles.css")

    @staticmethod
    async def _health(request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy"})

    async def _metrics(self, request: web.Request) -> web.Response:
        """Expose content-free local operational gauges without Guild/user labels."""

        snapshot = await self._snapshots.snapshot()
        active = 1 if snapshot.get("active") is True else 0
        sequence = snapshot.get("sequence", 0)
        if not isinstance(sequence, int):
            sequence = 0
        payload = "\n".join(
            (
                "# HELP lingxu_host_console_up Host console process readiness.",
                "# TYPE lingxu_host_console_up gauge",
                "lingxu_host_console_up 1",
                "# HELP lingxu_host_console_websockets Authorized live console connections.",
                "# TYPE lingxu_host_console_websockets gauge",
                f"lingxu_host_console_websockets {len(self._websockets)}",
                "# HELP lingxu_meeting_active Whether the authorized Guild has an active Meeting.",
                "# TYPE lingxu_meeting_active gauge",
                f"lingxu_meeting_active {active}",
                "# HELP lingxu_meeting_event_sequence Latest content-free event sequence.",
                "# TYPE lingxu_meeting_event_sequence gauge",
                f"lingxu_meeting_event_sequence {sequence}",
                "",
            )
        )
        return web.Response(
            text=payload,
            content_type="text/plain",
            charset="utf-8",
        )

    async def _login(self, request: web.Request) -> web.Response:
        self._sessions.require_same_origin(request)
        payload = await self._parse(request, _LoginPayload)
        assert isinstance(payload, _LoginPayload)
        session = self._sessions.login(payload.token)
        if session is None:
            raise web.HTTPUnauthorized(text="invalid login token")
        response = web.json_response(
            {
                "authenticated": True,
                "csrf_token": session.csrf_token,
                "guild_id": str(self._snapshots.guild_id),
            }
        )
        response.set_cookie(
            COOKIE_NAME,
            session.id,
            httponly=True,
            max_age=_SESSION_COOKIE_SECONDS,
            path="/",
            samesite="Strict",
        )
        return response

    async def _session_status(self, request: web.Request) -> web.Response:
        session = self._require_session(request)
        return web.json_response(
            {
                "authenticated": True,
                "csrf_token": session.csrf_token,
                "guild_id": str(self._snapshots.guild_id),
            }
        )

    async def _logout(self, request: web.Request) -> web.Response:
        session = self._require_session(request, csrf=True)
        self._sessions.logout(session.id)
        response = web.json_response({"authenticated": False})
        response.del_cookie(COOKIE_NAME, path="/")
        return response

    def _simulator_payload(self) -> dict[str, Any] | None:
        simulator = self._simulator
        if simulator is None:
            return None
        snapshot = simulator.snapshot()
        scenario = simulator.scenario
        possible_events = sorted(
            transition.event
            for transition in scenario.transitions
            if transition.from_state == snapshot.state
        )
        return {
            "scenario_key": snapshot.scenario_key,
            "state": snapshot.state,
            "sequence": snapshot.sequence,
            "terminal": snapshot.terminal,
            "possible_events": possible_events,
            "events": [
                {
                    "sequence": event.sequence,
                    "event": event.event,
                    "state_before": event.state_before,
                    "state_after": event.state_after,
                    "occurred_at": event.occurred_at.isoformat(),
                }
                for event in snapshot.events[-12:]
            ],
        }

    async def _full_snapshot(self) -> dict[str, Any]:
        snapshot = await self._snapshots.snapshot()
        snapshot["simulator"] = self._simulator_payload()
        return snapshot

    async def _snapshot(self, request: web.Request) -> web.Response:
        self._require_session(request)
        return web.json_response(await self._full_snapshot())

    async def _control(self, request: web.Request) -> web.Response:
        self._require_session(request, csrf=True)
        action = request.match_info["action"]
        if action not in _CONTROL_ACTIONS:
            raise web.HTTPNotFound(text="unknown control")
        current = await self._controller.active(self._snapshots.guild_id)
        arguments = {
            "guild_id": self._snapshots.guild_id,
            "actor_user_id": current.creator_user_id,
            "source": "HOST_CONSOLE",
        }
        operation = getattr(self._controller, action)
        try:
            await operation(**arguments)
        except LingxuError as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        return web.json_response(
            {
                "ok": True,
                "action": action,
                "snapshot": await self._full_snapshot(),
            }
        )

    async def _tts(self, request: web.Request) -> web.Response:
        self._require_session(request, csrf=True)
        payload = await self._parse(request, _TTSPayload)
        assert isinstance(payload, _TTSPayload)
        text = " ".join(payload.text.strip().split())
        if not text:
            raise web.HTTPBadRequest(text="TTS text cannot be blank")
        mode = await self._manual_tts(text)
        return web.json_response({"ok": True, "mode": mode})

    async def _simulator_reset(self, request: web.Request) -> web.Response:
        self._require_session(request, csrf=True)
        active = await self._snapshots.active_scenario()
        if active is None:
            raise web.HTTPConflict(text="no published fixed Scenario is available")
        self._simulator = ScenarioSimulator(active.schema)
        return web.json_response({"ok": True, "simulator": self._simulator_payload()})

    async def _simulator_dispatch(self, request: web.Request) -> web.Response:
        self._require_session(request, csrf=True)
        payload = await self._parse(request, _SimulationPayload)
        assert isinstance(payload, _SimulationPayload)
        simulator = self._simulator
        if simulator is None:
            raise web.HTTPConflict(text="reset the simulator first")
        try:
            simulator.dispatch(payload.event, role=ScenarioRole.HOST)
        except (KeyError, ValueError) as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        return web.json_response({"ok": True, "simulator": self._simulator_payload()})

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        self._sessions.require_same_origin(request)
        self._require_session(request)
        websocket = web.WebSocketResponse(
            heartbeat=20,
            max_msg_size=1_024,
            compress=False,
        )
        await websocket.prepare(request)
        self._websockets.add(websocket)
        try:
            await websocket.send_json(await self._full_snapshot())
            while not websocket.closed:
                try:
                    message = await websocket.receive(timeout=1.0)
                except TimeoutError:
                    await websocket.send_json(await self._full_snapshot())
                    continue
                if message.type in {
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSED,
                    WSMsgType.ERROR,
                }:
                    break
                if message.type is WSMsgType.TEXT and message.data == "refresh":
                    await websocket.send_json(await self._full_snapshot())
        except (ConnectionError, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            self._websockets.discard(websocket)
        return websocket
