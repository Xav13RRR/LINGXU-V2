import inspect
from types import SimpleNamespace
from typing import Any, cast

import discord

from lingxu.bot.meeting_commands import MeetingCommands
from lingxu.bot.meeting_panels import MeetingPanelCoordinator
from lingxu.bot.meeting_ui import ConfirmNextView, HostControlView, HostPanelRuntime, VoteView
from lingxu.bot.meeting_voice import MeetingVoiceCoordinator
from lingxu.bot.permissions import missing_meeting_voice_permissions
from lingxu.meeting.controller import MeetingController
from lingxu.meeting.intent_service import MeetingIntentService


def test_phase7_slash_command_surface_and_persistent_button_ids() -> None:
    controller = cast(MeetingController, object())
    intents = cast(MeetingIntentService, object())
    voice = cast(MeetingVoiceCoordinator, object())
    panels = cast(MeetingPanelCoordinator, object())
    panel_runtime = cast(HostPanelRuntime, object())
    cog = MeetingCommands(controller, intents, voice, panels)
    commands = {command.name: command for command in cog.__cog_app_commands__}

    assert set(commands) == {
        "agenda",
        "speaker",
        "vote",
        "intent",
        "pronunciation",
        "create",
        "controls",
        "start",
        "status",
        "pause",
        "resume",
        "next",
        "back",
        "skip",
        "stop",
        "speak",
        "consent",
        "withdraw",
        "retry",
        "delete",
    }
    assert {command.name for command in commands["agenda"].commands} == {
        "add",
        "list",
        "remove",
        "reorder",
        "template",
    }
    assert {command.name for command in commands["speaker"].commands} == {
        "abandon",
        "end",
        "extend",
        "next",
        "finish",
        "followup-end",
        "remove",
        "skip",
    }
    assert {command.name for command in commands["vote"].commands} == {
        "open",
        "cast",
        "close",
        "status",
    }
    assert {command.name for command in commands["intent"].commands} == {
        "resolve",
        "confirm",
        "reject",
    }
    assert {command.name for command in commands["pronunciation"].commands} == {
        "set",
        "show",
        "clear",
    }

    control_ids = {
        cast(Any, item).custom_id for item in HostControlView(controller, panel_runtime).children
    }
    vote_ids = {cast(Any, item).custom_id for item in VoteView(controller).children}
    assert control_ids == {
        "lingxu:host:pause:v1",
        "lingxu:host:resume:v1",
        "lingxu:host:next:v1",
        "lingxu:host:extend:v1",
        "lingxu:host:skip-turn:v1",
        "lingxu:host:end-turn:v1",
        "lingxu:host:end-followup:v1",
        "lingxu:host:abandon:v1",
        "lingxu:host:stop:v1",
        "lingxu:host:intervene-start:v1",
        "lingxu:host:intervene-end:v1",
        "lingxu:host:enforcement-pause:v1",
        "lingxu:host:enforcement-resume:v1",
        "lingxu:host:emergency-release:v1",
        "lingxu:host:exception-grant:v1",
        "lingxu:host:exception-revoke:v1",
    }
    assert vote_ids == {
        "lingxu:vote:yes:v1",
        "lingxu:vote:no:v1",
        "lingxu:vote:abstain:v1",
    }


def test_mute_members_permission_is_required_only_for_requested_strict_agendas() -> None:
    permissions = SimpleNamespace(
        view_channel=True,
        connect=True,
        speak=True,
        mute_members=False,
    )
    channel = cast(
        discord.VoiceChannel,
        SimpleNamespace(
            guild=SimpleNamespace(me=object()),
            permissions_for=lambda _member: permissions,
        ),
    )

    assert missing_meeting_voice_permissions(channel) == ()
    assert missing_meeting_voice_permissions(
        channel,
        require_mute_members=True,
    ) == ("Mute Members",)


def test_latency_sensitive_meeting_interactions_acknowledge_before_io() -> None:
    command_callbacks = {
        "create": (MeetingCommands.create.callback, "await self._controller.create"),
        "status": (MeetingCommands.status.callback, "await self._controller.status"),
        "retry": (MeetingCommands.retry.callback, "await self._controller.retry_session"),
    }
    for name, (callback, first_io) in command_callbacks.items():
        source = inspect.getsource(callback)
        assert "await interaction.response.defer" in source, name
        assert source.index("await interaction.response.defer") < source.index(first_io), name

    confirm_source = inspect.getsource(ConfirmNextView.confirm)
    assert "await interaction.response.defer" in confirm_source
    assert confirm_source.index("await interaction.response.defer") < confirm_source.index(
        "await self._controller.next"
    )
