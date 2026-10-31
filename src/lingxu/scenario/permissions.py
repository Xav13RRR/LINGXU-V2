"""Scenario roles, permissions, and deterministic Discord role mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from lingxu.domain.errors import MeetingPermissionError


class ScenarioRole(StrEnum):
    HOST = "HOST"
    FACILITATOR = "FACILITATOR"
    PARTICIPANT = "PARTICIPANT"
    OBSERVER = "OBSERVER"


class ScenarioPermission(StrEnum):
    VIEW_PUBLIC = "VIEW_PUBLIC"
    CONTROL_SCENARIO = "CONTROL_SCENARIO"
    MANAGE_ROLES = "MANAGE_ROLES"
    SPEAK = "SPEAK"
    VOTE = "VOTE"
    VIEW_PRIVATE = "VIEW_PRIVATE"
    MANAGE_PRIVATE = "MANAGE_PRIVATE"


_ROLE_PERMISSIONS = {
    ScenarioRole.HOST: frozenset(ScenarioPermission),
    ScenarioRole.FACILITATOR: frozenset(
        {
            ScenarioPermission.VIEW_PUBLIC,
            ScenarioPermission.CONTROL_SCENARIO,
            ScenarioPermission.SPEAK,
            ScenarioPermission.VOTE,
            ScenarioPermission.VIEW_PRIVATE,
        }
    ),
    ScenarioRole.PARTICIPANT: frozenset(
        {
            ScenarioPermission.VIEW_PUBLIC,
            ScenarioPermission.SPEAK,
            ScenarioPermission.VOTE,
        }
    ),
    ScenarioRole.OBSERVER: frozenset({ScenarioPermission.VIEW_PUBLIC}),
}


class RolePolicy:
    def allows(self, role: ScenarioRole, permission: ScenarioPermission) -> bool:
        return permission in _ROLE_PERMISSIONS[role]

    def require(self, role: ScenarioRole, permission: ScenarioPermission) -> None:
        if not self.allows(role, permission):
            raise MeetingPermissionError(f"Scenario role {role.value} lacks {permission.value}")


@dataclass(frozen=True, slots=True)
class DiscordRoleMapping:
    discord_role_id: int
    scenario_role: ScenarioRole


class DiscordRoleMapper:
    """Map role IDs only; role names never grant authority."""

    _PRIORITY: ClassVar[dict[ScenarioRole, int]] = {
        ScenarioRole.HOST: 4,
        ScenarioRole.FACILITATOR: 3,
        ScenarioRole.PARTICIPANT: 2,
        ScenarioRole.OBSERVER: 1,
    }

    def __init__(self, mappings: tuple[DiscordRoleMapping, ...]) -> None:
        role_ids = [mapping.discord_role_id for mapping in mappings]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("Discord role mapping ids must be unique")
        self._mappings = {mapping.discord_role_id: mapping.scenario_role for mapping in mappings}

    def resolve(
        self,
        *,
        user_id: int,
        session_host_user_id: int,
        discord_role_ids: frozenset[int],
    ) -> ScenarioRole:
        if user_id == session_host_user_id:
            return ScenarioRole.HOST
        roles = [
            self._mappings[role_id] for role_id in discord_role_ids if role_id in self._mappings
        ]
        return max(roles, key=self._PRIORITY.__getitem__) if roles else ScenarioRole.OBSERVER
