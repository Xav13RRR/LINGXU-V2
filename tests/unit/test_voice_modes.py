import pytest

from lingxu.bot.voice_modes import GuildVoiceMode, GuildVoiceModeRegistry
from lingxu.domain.errors import VoiceModeConflictError


@pytest.mark.asyncio
async def test_voice_mode_claim_is_idempotent_and_conflicts_across_tasks() -> None:
    registry = GuildVoiceModeRegistry()
    meeting = await registry.claim(guild_id=1, mode=GuildVoiceMode.MEETING, owner_id="meeting")
    repeated = await registry.claim(guild_id=1, mode=GuildVoiceMode.MEETING, owner_id="meeting")

    assert repeated == meeting
    with pytest.raises(VoiceModeConflictError, match="MEETING"):
        await registry.claim(
            guild_id=1,
            mode=GuildVoiceMode.RESIDENT_ASSISTANT,
            owner_id="resident",
        )
    assert (
        await registry.release(
            type(meeting)(
                token="wrong",
                guild_id=meeting.guild_id,
                mode=meeting.mode,
                owner_id=meeting.owner_id,
            )
        )
        is False
    )
    assert await registry.release(meeting) is True
    resident = await registry.claim(
        guild_id=1,
        mode=GuildVoiceMode.RESIDENT_ASSISTANT,
        owner_id="resident",
    )
    assert registry.current(1) == resident
