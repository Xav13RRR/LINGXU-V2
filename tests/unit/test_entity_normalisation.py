"""Unit tests for deterministic, audited glossary normalisation."""

from pathlib import Path, PureWindowsPath

import pytest

from lingxu.config import Settings
from lingxu.transcript.glossary import (
    Glossary,
    _resolved_path_is_within,
    load_glossaries,
    normalise_entities,
)
from lingxu.transcript.models import CanonicalTranscript, CanonicalTranscriptEntry


def entry(text: str) -> CanonicalTranscriptEntry:
    """Build one synthetic canonical entry without Meeting content."""

    return CanonicalTranscriptEntry(
        entry_id="entry-1",
        source_entry_ids=("source-1",),
        speaker_user_id=7,
        speaker_name=None,
        started_ms=1_000,
        ended_ms=2_000,
        raw_texts=(text,),
        text=text,
        entry_type="statement",
        language="en",
        confidence=None,
        utterance_group_id="group:7:entry-1",
        normalisation_changes=(),
        asr_tags=(),
        emotion=None,
    )


def test_exact_alias_is_audited_and_unknown_entity_is_unchanged() -> None:
    glossary = Glossary.from_mapping(
        {"entities": [{"canonical": "Example", "aliases": ["exampel"]}]}
    )

    changed = normalise_entities(entry("exampel"), glossaries=(glossary,))
    untouched = normalise_entities(entry("unknown-name"), glossaries=(glossary,))

    assert changed.text == "Example"
    assert changed.normalisation_changes[0].source == "project_glossary"
    assert changed.normalisation_changes[0].original_text == "exampel"
    assert changed.normalisation_changes[0].replacement_text == "Example"
    assert changed.normalisation_changes[0].confidence == 1.0
    assert changed.normalisation_changes[0].source_entry_ids == ("source-1",)
    assert (
        changed.normalisation_changes[0].started_offset,
        changed.normalisation_changes[0].ended_offset,
    ) == (0, 7)
    assert untouched.text == "unknown-name"
    assert untouched.normalisation_changes == ()


def test_exact_alias_with_canonical_case_is_normalised_and_audited() -> None:
    glossary = Glossary.from_mapping({"entities": [{"canonical": "OpenAI", "aliases": ["openai"]}]})

    changed = normalise_entities(entry("openai"), glossaries=(glossary,))

    assert changed.text == "OpenAI"
    assert changed.normalisation_changes[0].original_text == "openai"
    assert changed.normalisation_changes[0].replacement_text == "OpenAI"
    assert normalise_entities(changed, glossaries=(glossary,)) == changed


def test_meeting_glossary_has_priority_over_project_guild_and_global() -> None:
    glossaries = tuple(
        Glossary.from_mapping(
            {"entities": [{"canonical": canonical, "aliases": ["shared-alias"]}]},
            source=source,
        )
        for canonical, source in (
            ("Meeting", "meeting_glossary"),
            ("Project", "project_glossary"),
            ("Guild", "guild_glossary"),
            ("Global", "global_glossary"),
        )
    )

    result = normalise_entities(entry("shared-alias"), glossaries=glossaries)

    assert result.text == "Meeting"
    assert tuple(change.source for change in result.normalisation_changes) == ("meeting_glossary",)


def test_conservative_fuzzy_match_applies_only_one_unambiguous_edit() -> None:
    glossary = Glossary.from_mapping(
        {
            "entities": [
                {"canonical": "Example", "aliases": ["example"]},
                {"canonical": "Samples", "aliases": ["samples"]},
            ]
        }
    )

    changed = normalise_entities(entry("examlpe"), glossaries=(glossary,))
    untouched = normalise_entities(entry("unknown-name"), glossaries=(glossary,))

    assert changed.text == "Example"
    assert changed.normalisation_changes[0].original_text == "examlpe"
    assert changed.normalisation_changes[0].confidence == pytest.approx(6 / 7)
    assert changed.normalisation_changes[0].source_entry_ids == ("source-1",)
    assert untouched.text == "unknown-name"
    assert untouched.normalisation_changes == ()


def test_model_suggestions_are_never_automatically_applied() -> None:
    glossary = Glossary.from_mapping(
        {
            "entities": [],
            "model_suggestions": [
                {"canonical": "Suggested", "aliases": ["suggestd"]},
            ],
        }
    )

    result = normalise_entities(entry("suggestd"), glossaries=(glossary,))

    assert result.text == "suggestd"
    assert result.normalisation_changes == ()


def test_alias_chain_is_non_cascading_with_one_audit_layer() -> None:
    glossary = Glossary.from_mapping(
        {
            "entities": [
                {"canonical": "beta", "aliases": ["alpha"]},
                {"canonical": "gamma", "aliases": ["beta"]},
            ]
        }
    )

    result = normalise_entities(entry("alpha"), glossaries=(glossary,))

    assert result.text == "beta"
    assert len(result.normalisation_changes) == 1
    assert result.normalisation_changes[0].original_text == "alpha"
    assert result.normalisation_changes[0].replacement_text == "beta"


def test_legacy_canonical_audit_deserialises_with_explicit_defaults() -> None:
    legacy = {
        "schema_version": 1,
        "entries": [
            {
                "entry_id": "entry-1",
                "source_entry_ids": ["source-1", "source-2"],
                "speaker_user_id": 7,
                "speaker_name": "participant",
                "started_ms": 1_000,
                "ended_ms": 2_000,
                "raw_texts": ["alpha"],
                "text": "beta",
                "entry_type": "statement",
                "language": "en",
                "confidence": None,
                "utterance_group_id": "group:7:entry-1",
                "normalisation_changes": [
                    {
                        "source": "project_glossary",
                        "original_text": "alpha",
                        "replacement_text": "beta",
                        "started_offset": 0,
                        "ended_offset": 5,
                    }
                ],
                "asr_tags": [],
                "emotion": None,
            }
        ],
        "metrics": {
            "source_entry_count": 2,
            "canonical_entry_count": 1,
            "duplicate_entry_count": 1,
            "backchannel_entry_count": 0,
            "asr_tag_count": 0,
        },
    }

    decoded = CanonicalTranscript.from_dict(legacy)

    assert decoded.schema_version == 1
    assert decoded.entries[0].normalisation_changes[0].confidence == 1.0
    assert decoded.entries[0].normalisation_changes[0].source_entry_ids == (
        "source-1",
        "source-2",
    )


def test_layered_yaml_and_json_loading_uses_fixed_safe_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "sessions"
    glossary_root = tmp_path / "glossaries"
    session_id = "01SYNTHETICSESSION0000000000"
    meeting_dir = data_root / session_id / "meeting"
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "glossary.yaml").write_text(
        "entities:\n  - canonical: Meeting\n    aliases: [alias]\n",
        encoding="utf-8",
    )
    glossary_root.mkdir()
    (glossary_root / "project.json").write_text(
        '{"entities":[{"canonical":"Project","aliases":["alias"]}]}',
        encoding="utf-8",
    )
    (glossary_root / "guilds").mkdir()
    (glossary_root / "guilds" / "81.yaml").write_text(
        "entities:\n  - canonical: Guild\n    aliases: [alias]\n",
        encoding="utf-8",
    )
    (glossary_root / "global.json").write_text(
        '{"entities":[{"canonical":"Global","aliases":["alias"]}]}',
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        data_root=data_root,
        glossary_root=glossary_root,
    )

    glossaries = load_glossaries(settings, session_id=session_id, guild_id=81)

    assert tuple(glossary.source for glossary in glossaries) == (
        "meeting_glossary",
        "project_glossary",
        "guild_glossary",
        "global_glossary",
    )
    assert normalise_entities(entry("alias"), glossaries=glossaries).text == "Meeting"


@pytest.mark.parametrize(
    "payload",
    (
        "entities: not-a-list\n",
        "entities:\n  - canonical: ''\n    aliases: [bad]\n",
        "entities:\n  - canonical: Valid\n    aliases: not-a-list\n",
        "unknown: []\n",
    ),
)
def test_malformed_glossary_dictionary_is_rejected(tmp_path: Path, payload: str) -> None:
    glossary_root = tmp_path / "glossaries"
    glossary_root.mkdir()
    (glossary_root / "project.yaml").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="glossary"):
        load_glossaries(
            Settings(_env_file=None, data_root=tmp_path / "sessions", glossary_root=glossary_root),
            session_id="01SYNTHETICSESSION0000000000",
            guild_id=81,
        )


def test_unsafe_session_path_is_rejected(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_root=tmp_path / "sessions",
        glossary_root=tmp_path / "glossaries",
    )

    with pytest.raises(ValueError, match="unsafe"):
        load_glossaries(settings, session_id="../escape", guild_id=81)


def test_resolved_path_containment_is_capability_independent() -> None:
    root = PureWindowsPath("C:/lingxu/glossaries")

    assert _resolved_path_is_within(root, root / "guilds" / "81.yaml")
    assert not _resolved_path_is_within(root, PureWindowsPath("C:/lingxu/glossaries-escape/a.yaml"))
    assert not _resolved_path_is_within(root, PureWindowsPath("D:/lingxu/glossaries/a.yaml"))


def test_glossary_symlink_escape_is_rejected(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_root=tmp_path / "sessions",
        glossary_root=tmp_path / "glossaries",
    )
    outside = tmp_path / "outside.yaml"
    outside.write_text("entities: []\n", encoding="utf-8")
    project = settings.glossary_root / "project.yaml"
    project.parent.mkdir()
    try:
        project.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(ValueError, match="unsafe"):
        load_glossaries(
            settings,
            session_id="01SYNTHETICSESSION0000000000",
            guild_id=81,
        )
