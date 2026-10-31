from typing import Any, cast

from lingxu.bot.assistant_commands import AssistantCommands, ResidentAssistantControl


def test_resident_assistant_command_surface_and_default_permissions() -> None:
    cog = AssistantCommands(cast(ResidentAssistantControl, object()))
    commands = {command.name: command for command in cog.__cog_app_commands__}

    assert set(commands) == {"join", "status", "stop"}
    assert cast(Any, commands["join"]).default_permissions.manage_guild is True
    assert cast(Any, commands["stop"]).default_permissions.manage_guild is True
