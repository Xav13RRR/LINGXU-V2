"""Layered, deterministic entity dictionaries with auditable replacement."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Any

import yaml

from lingxu.config import Settings
from lingxu.transcript.models import CanonicalTranscriptEntry, NormalisationChange

_MAX_GLOSSARY_BYTES = 1_048_576
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")
_FUZZY_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{5,}")
_ASCII_ALIAS = re.compile(r"[A-Za-z0-9_-]+\Z")
_SOURCES = frozenset({"meeting_glossary", "project_glossary", "guild_glossary", "global_glossary"})


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """One reviewed canonical entity and its explicitly approved aliases."""

    canonical: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Glossary:
    """One reviewed entity dictionary from a named priority layer."""

    source: str
    entries: tuple[GlossaryEntry, ...]

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any],
        *,
        source: str = "project_glossary",
    ) -> Glossary:
        """Validate reviewed entities while deliberately ignoring model suggestions."""

        if source not in _SOURCES:
            raise ValueError("glossary source is not recognised")
        unknown_fields = set(mapping) - {"entities", "model_suggestions"}
        if unknown_fields:
            raise ValueError("glossary contains unknown fields")
        entities = mapping.get("entities")
        if not isinstance(entities, list):
            raise ValueError("glossary entities must be a list")
        suggestions = mapping.get("model_suggestions", [])
        if not isinstance(suggestions, list):
            raise ValueError("glossary model_suggestions must be a list")

        parsed: list[GlossaryEntry] = []
        aliases_seen: dict[str, str] = {}
        for value in entities:
            if not isinstance(value, dict) or set(value) != {"canonical", "aliases"}:
                raise ValueError("glossary entity must contain canonical and aliases")
            canonical = _validated_text(value["canonical"], field="canonical")
            aliases_value = value["aliases"]
            if not isinstance(aliases_value, list) or not aliases_value:
                raise ValueError("glossary aliases must be a non-empty list")
            aliases = tuple(
                dict.fromkeys(_validated_text(alias, field="alias") for alias in aliases_value)
            )
            for alias in aliases:
                key = alias.casefold()
                previous = aliases_seen.get(key)
                if previous is not None and previous != canonical:
                    raise ValueError("glossary alias maps to multiple canonical entities")
                aliases_seen[key] = canonical
            parsed.append(GlossaryEntry(canonical=canonical, aliases=aliases))
        return cls(source=source, entries=tuple(parsed))


@dataclass(frozen=True, slots=True)
class _Replacement:
    started_offset: int
    ended_offset: int
    canonical: str
    source: str
    priority: int
    confidence: float


def load_glossaries(
    settings: Settings,
    session_id: str,
    guild_id: int,
) -> tuple[Glossary, ...]:
    """Load fixed Meeting/project/Guild/global files without permitting path escape."""

    if not _SAFE_IDENTIFIER.fullmatch(session_id):
        raise ValueError("unsafe glossary Session identifier")
    if isinstance(guild_id, bool) or guild_id <= 0:
        raise ValueError("unsafe glossary Guild identifier")
    specifications = (
        (
            settings.data_root,
            (session_id, "meeting", "glossary"),
            "meeting_glossary",
        ),
        (settings.glossary_root, ("project",), "project_glossary"),
        (settings.glossary_root, ("guilds", str(guild_id)), "guild_glossary"),
        (settings.glossary_root, ("global",), "global_glossary"),
    )
    loaded: list[Glossary] = []
    for root, stem_parts, source in specifications:
        path = _one_layer_path(root, stem_parts)
        if path is not None:
            loaded.append(Glossary.from_mapping(_load_mapping(path), source=source))
    return tuple(loaded)


def normalise_entities(
    entry: CanonicalTranscriptEntry,
    *,
    glossaries: tuple[Glossary, ...],
) -> CanonicalTranscriptEntry:
    """Apply non-cascading reviewed replacements and append exact offset audits."""

    if not entry.text or not glossaries:
        return entry
    candidates = _exact_replacements(entry.text, glossaries)
    occupied = [(item.started_offset, item.ended_offset) for item in candidates]
    candidates.extend(_fuzzy_replacements(entry.text, glossaries, occupied=occupied))
    selected = _non_overlapping(candidates)
    if not selected:
        return entry

    parts: list[str] = []
    changes: list[NormalisationChange] = []
    cursor = 0
    for replacement in selected:
        parts.append(entry.text[cursor : replacement.started_offset])
        original = entry.text[replacement.started_offset : replacement.ended_offset]
        parts.append(replacement.canonical)
        changes.append(
            NormalisationChange(
                source=replacement.source,
                original_text=original,
                replacement_text=replacement.canonical,
                started_offset=replacement.started_offset,
                ended_offset=replacement.ended_offset,
                confidence=replacement.confidence,
                source_entry_ids=entry.source_entry_ids,
            )
        )
        cursor = replacement.ended_offset
    parts.append(entry.text[cursor:])
    return replace(
        entry,
        text="".join(parts),
        normalisation_changes=entry.normalisation_changes + tuple(changes),
    )


def _validated_text(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"glossary {field} must be text")
    selected = value.strip()
    if not selected or len(selected) > 256 or any(ord(character) < 32 for character in selected):
        raise ValueError(f"glossary {field} is invalid")
    return selected


def _one_layer_path(root: Path, stem_parts: tuple[str, ...]) -> Path | None:
    resolved_root = root.resolve()
    candidates: list[Path] = []
    for suffix in (".json", ".yaml", ".yml"):
        unresolved = root.joinpath(*stem_parts[:-1], f"{stem_parts[-1]}{suffix}")
        resolved = unresolved.resolve()
        if not _resolved_path_is_within(resolved_root, resolved):
            raise ValueError("unsafe glossary path escapes configured root")
        if unresolved.exists():
            candidates.append(resolved)
    if len(candidates) > 1:
        raise ValueError("glossary layer has multiple source files")
    return candidates[0] if candidates else None


def _resolved_path_is_within(root: PurePath, candidate: PurePath) -> bool:
    """Check already-resolved path containment without filesystem capabilities."""

    if type(root) is not type(candidate):
        return False
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) > _MAX_GLOSSARY_BYTES:
        raise ValueError("glossary file exceeds size limit")
    try:
        text = payload.decode("utf-8")
        value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError("glossary source is malformed") from error
    if not isinstance(value, dict):
        raise ValueError("glossary source must contain one mapping")
    return value


def _exact_replacements(
    text: str,
    glossaries: tuple[Glossary, ...],
) -> list[_Replacement]:
    replacements: list[_Replacement] = []
    for priority, glossary in enumerate(glossaries):
        for entity in glossary.entries:
            for alias in entity.aliases:
                escaped = re.escape(alias)
                if _ASCII_ALIAS.fullmatch(alias):
                    escaped = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
                for match in re.finditer(escaped, text, flags=re.IGNORECASE):
                    if match.group(0) == entity.canonical:
                        continue
                    replacements.append(
                        _Replacement(
                            started_offset=match.start(),
                            ended_offset=match.end(),
                            canonical=entity.canonical,
                            source=glossary.source,
                            priority=priority,
                            confidence=1.0,
                        )
                    )
    return replacements


def _fuzzy_replacements(
    text: str,
    glossaries: tuple[Glossary, ...],
    *,
    occupied: list[tuple[int, int]],
) -> list[_Replacement]:
    replacements: list[_Replacement] = []
    reviewed: list[tuple[int, Glossary, GlossaryEntry, str]] = []
    for priority, glossary in enumerate(glossaries):
        for entity in glossary.entries:
            for alias in entity.aliases:
                if _ASCII_ALIAS.fullmatch(alias) and len(alias) >= 6:
                    reviewed.append((priority, glossary, entity, alias))
    for match in _FUZZY_TOKEN.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        token = match.group(0)
        known_values = {
            value.casefold()
            for _, _, entity, alias in reviewed
            for value in (entity.canonical, alias)
        }
        if token.casefold() in known_values:
            continue
        matches = [
            (priority, glossary, entity, _fuzzy_confidence(token, alias))
            for priority, glossary, entity, alias in reviewed
            if _one_edit_or_transposition(token.casefold(), alias.casefold())
        ]
        if not matches:
            continue
        best_priority = min(priority for priority, _, _, _ in matches)
        best = [item for item in matches if item[0] == best_priority]
        canonicals = {entity.canonical for _, _, entity, _ in best}
        if len(canonicals) != 1:
            continue
        _, glossary, entity, confidence = min(
            best,
            key=lambda item: (item[2].canonical.casefold(), -item[3]),
        )
        replacements.append(
            _Replacement(
                started_offset=match.start(),
                ended_offset=match.end(),
                canonical=entity.canonical,
                source=glossary.source,
                priority=best_priority,
                confidence=confidence,
            )
        )
    return replacements


def _fuzzy_confidence(first: str, second: str) -> float:
    """Return the deterministic normalized score for an accepted one-edit match."""

    return 1.0 - (1.0 / max(len(first), len(second)))


def _one_edit_or_transposition(first: str, second: str) -> bool:
    if first == second or abs(len(first) - len(second)) > 1:
        return False
    if len(first) == len(second):
        mismatches = [
            index
            for index, values in enumerate(zip(first, second, strict=True))
            if values[0] != values[1]
        ]
        if len(mismatches) == 1:
            return True
        return (
            len(mismatches) == 2
            and mismatches[1] == mismatches[0] + 1
            and first[mismatches[0]] == second[mismatches[1]]
            and first[mismatches[1]] == second[mismatches[0]]
        )
    shorter, longer = (first, second) if len(first) < len(second) else (second, first)
    short_index = 0
    long_index = 0
    skipped = False
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
        elif skipped:
            return False
        else:
            skipped = True
            long_index += 1
    return True


def _non_overlapping(candidates: list[_Replacement]) -> list[_Replacement]:
    selected: list[_Replacement] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.started_offset,
            item.priority,
            -(item.ended_offset - item.started_offset),
            item.canonical.casefold(),
            item.source,
        ),
    ):
        if selected and candidate.started_offset < selected[-1].ended_offset:
            continue
        selected.append(candidate)
    return selected
