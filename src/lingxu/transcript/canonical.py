"""Pure deterministic canonicalisation for privacy-eligible transcript entries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from lingxu.domain.models import TranscriptEntryRecord
from lingxu.transcript.models import (
    CanonicalTranscript,
    CanonicalTranscriptEntry,
    CanonicalTranscriptMetrics,
)

OVERLAP_IOU_MIN = 0.70
SIMILARITY_MIN = 0.85
SAME_START_TOLERANCE_MS = 100
SAME_START_SIMILARITY_MIN = 0.78
CROSS_SOURCE_SIMILARITY_MIN = 0.78
DUPLICATE_WINDOW_MS = 30_000
UTTERANCE_GROUP_GAP_MS = 1_500
CONTAINMENT_RATIO_MIN = 0.70
EXCLUSIVE_ALIGNMENT_LENGTH_RATIO_MIN = 0.90

_TAG_PATTERN = re.compile(r"<\|([^<>|]+)\|>")
_BRACKET_TAG_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z _-]{0,31})\]")
_LANGUAGE_TAGS = frozenset({"zh", "en", "yue", "ja", "ko", "nospeech"})
_EMOTION_TAGS = frozenset({"angry", "happy", "neutral", "sad", "unknown"})
_INTERNAL_TAGS = frozenset(
    {
        "speech",
        "withitn",
        "woitn",
        "bgm",
        "applause",
        "laughter",
        "cry",
        "sneeze",
        "breath",
        "cough",
        "music",
        "noise",
    }
)
_BACKCHANNELS = frozenset(
    {
        "啊",
        "哦",
        "噢",
        "嗯",
        "嗯嗯",
        "呃",
        "额",
        "唔",
        "uh",
        "um",
        "hmm",
        "mm",
    }
)
_PRIVACY_CONTROL = re.compile(
    r"撤回.{0,4}同意|停止.{0,4}(录音|记录|转写)|不同意.{0,4}(录音|记录|转写)"
)
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?|[零一二三四五六七八九十百千万亿两]+")
_ACCEPTANCE_MARKERS = ("同意", "接受", "赞成", "可以", "确认")
_NEGATION_PROHIBITION_MARKERS = (
    "不可以",
    "不同意",
    "不得",
    "不能",
    "不许",
    "不要",
    "不行",
    "禁止",
    "拒绝",
    "取消",
    "停止",
    "反对",
    "否决",
    "未",
    "无",
    "否",
    "非",
    "别",
    "勿",
    "莫",
    "不",
    "没",
    "撤回",
)
_INTERROGATIVE_MARKERS = (
    "是否",
    "能否",
    "可否",
    "要不要",
    "有没有",
    "谁",
    "什么",
    "何时",
    "哪里",
    "为什么",
    "怎么",
)
_ACTION_VERBS = (
    "发布",
    "安排",
    "完成",
    "跟进",
    "负责",
    "执行",
    "提交",
    "修复",
    "部署",
    "删除",
    "创建",
    "分配",
    "取消",
)
_SOURCE_PRECEDENCE = {
    "manual": 4,
    "human": 4,
    "segment": 3,
    "utterance": 2,
    "legacy": 1,
    "bot_tts": 0,
}


@dataclass(frozen=True, slots=True)
class _SourceEntry:
    record: TranscriptEntryRecord
    text: str
    comparable_text: str
    tags: tuple[str, ...]
    language: str | None
    emotion: str | None


@dataclass(frozen=True, slots=True)
class _MaterialSignature:
    polarity: str
    is_question: bool
    primary_action: str | None
    numbers: tuple[str, ...]
    privacy_control: str | None


def build_canonical_transcript(
    entries: tuple[TranscriptEntryRecord, ...],
) -> CanonicalTranscript:
    """Resolve bounded same-speaker duplicates without mutating source records."""

    source_entries = (_source_entry(entry) for entry in entries)
    sources = tuple(sorted(source_entries, key=_source_sort_key))
    exclusive_alignment_pairs = _exclusive_cross_source_pairs(sources)
    components = _duplicate_components(
        sources,
        exclusive_alignment_pairs=exclusive_alignment_pairs,
    )
    canonical_entries = tuple(_canonical_entry(component) for component in components)
    grouped_entries = _assign_utterance_groups(canonical_entries)
    metrics = CanonicalTranscriptMetrics(
        source_entry_count=len(sources),
        canonical_entry_count=len(grouped_entries),
        duplicate_entry_count=len(sources) - len(grouped_entries),
        backchannel_entry_count=sum(item.entry_type == "backchannel" for item in grouped_entries),
        asr_tag_count=sum(len(item.asr_tags) for item in grouped_entries),
    )
    return CanonicalTranscript(entries=grouped_entries, metrics=metrics)


def _source_entry(record: TranscriptEntryRecord) -> _SourceEntry:
    raw_text, raw_tags, raw_language, raw_emotion = _strip_asr_tags(record.raw_text)
    display_tags: tuple[str, ...]
    display_language: str | None
    display_emotion: str | None
    if record.normalised_text is None:
        text, display_tags, display_language, display_emotion = (
            raw_text,
            (),
            None,
            None,
        )
    else:
        text, display_tags, display_language, display_emotion = _strip_asr_tags(
            record.normalised_text
        )
    return _SourceEntry(
        record=record,
        text=text,
        comparable_text=_normalised_characters(text),
        tags=tuple(sorted(set(raw_tags) | set(display_tags))),
        language=raw_language or display_language,
        emotion=raw_emotion or display_emotion,
    )


def _source_sort_key(source: _SourceEntry) -> tuple[int, int, int, str]:
    record = source.record
    return (
        record.started_ms,
        record.ended_ms,
        record.user_id if record.user_id is not None else -1,
        record.id,
    )


def _duplicate_components(
    sources: tuple[_SourceEntry, ...],
    *,
    exclusive_alignment_pairs: frozenset[frozenset[str]],
) -> tuple[tuple[_SourceEntry, ...], ...]:
    components: list[list[_SourceEntry]] = []
    for source in sources:
        for component in components:
            if _can_join_component(
                source,
                component,
                exclusive_alignment_pairs=exclusive_alignment_pairs,
            ):
                component.append(source)
                break
        else:
            components.append([source])
    return tuple(tuple(component) for component in components)


def _can_join_component(
    source: _SourceEntry,
    component: list[_SourceEntry],
    *,
    exclusive_alignment_pairs: frozenset[frozenset[str]],
) -> bool:
    earliest_start = min(
        (member.record.started_ms for member in component), default=source.record.started_ms
    )
    latest_start = max(
        (member.record.started_ms for member in component), default=source.record.started_ms
    )
    if (
        max(latest_start, source.record.started_ms) - min(earliest_start, source.record.started_ms)
        > DUPLICATE_WINDOW_MS
    ):
        return False
    return all(
        _is_duplicate(
            source,
            member,
            exclusive_alignment_pairs=exclusive_alignment_pairs,
        )
        for member in component
    )


def _is_duplicate(
    first: _SourceEntry,
    second: _SourceEntry,
    *,
    exclusive_alignment_pairs: frozenset[frozenset[str]] = frozenset(),
) -> bool:
    if first.record.user_id != second.record.user_id:
        return False
    if not _materially_compatible(first, second):
        return False
    started_delta = abs(first.record.started_ms - second.record.started_ms)
    same_start = started_delta <= SAME_START_TOLERANCE_MS
    overlap_iou = _time_overlap_iou(first.record, second.record)
    overlaps = _has_time_overlap(first.record, second.record)
    similarity = SequenceMatcher(None, first.comparable_text, second.comparable_text).ratio()
    text_length_ratio = _text_length_ratio(first.comparable_text, second.comparable_text)
    if first.comparable_text == second.comparable_text and (overlaps or same_start):
        return True
    if overlaps and _is_segment_utterance_pair(first, second):
        if similarity >= CROSS_SOURCE_SIMILARITY_MIN and text_length_ratio >= CONTAINMENT_RATIO_MIN:
            return True
        source_pair = frozenset((first.record.id, second.record.id))
        if (
            source_pair in exclusive_alignment_pairs
            and text_length_ratio >= EXCLUSIVE_ALIGNMENT_LENGTH_RATIO_MIN
        ):
            return True
    if (
        overlaps
        and _containment_ratio(first.comparable_text, second.comparable_text)
        >= CONTAINMENT_RATIO_MIN
    ):
        return True
    if overlap_iou >= OVERLAP_IOU_MIN and similarity >= SIMILARITY_MIN:
        return True
    return (
        same_start
        and similarity >= SAME_START_SIMILARITY_MIN
        and text_length_ratio >= CONTAINMENT_RATIO_MIN
    )


def _exclusive_cross_source_pairs(
    sources: tuple[_SourceEntry, ...],
) -> frozenset[frozenset[str]]:
    """Find reciprocal one-to-one overlapping Segment/utterance alignments."""

    candidates: list[tuple[_SourceEntry, _SourceEntry]] = []
    overlap_counts: dict[str, int] = {}
    for index, first in enumerate(sources):
        for second in sources[index + 1 :]:
            if (
                first.record.user_id != second.record.user_id
                or not _is_segment_utterance_pair(first, second)
                or not _has_time_overlap(first.record, second.record)
            ):
                continue
            candidates.append((first, second))
            for source in (first, second):
                overlap_counts[source.record.id] = overlap_counts.get(source.record.id, 0) + 1
    return frozenset(
        frozenset((first.record.id, second.record.id))
        for first, second in candidates
        if overlap_counts[first.record.id] == overlap_counts[second.record.id] == 1
    )


def _is_segment_utterance_pair(first: _SourceEntry, second: _SourceEntry) -> bool:
    return {_source_kind(first), _source_kind(second)} == {"segment", "utterance"}


def _source_kind(source: _SourceEntry) -> str:
    return source.record.source_key.partition(":")[0].casefold()


def _materially_compatible(first: _SourceEntry, second: _SourceEntry) -> bool:
    first_signature = _material_signature(first.text)
    second_signature = _material_signature(second.text)
    if first_signature.is_question != second_signature.is_question:
        return False
    if first_signature.numbers != second_signature.numbers:
        return False
    if first_signature.privacy_control != second_signature.privacy_control:
        return False
    if (
        first_signature.primary_action is not None
        and second_signature.primary_action is not None
        and first_signature.primary_action != second_signature.primary_action
    ):
        return False
    return (first_signature.polarity == "negated") == (second_signature.polarity == "negated")


def _material_signature(text: str) -> _MaterialSignature:
    comparable = _normalised_characters(text)
    # Compound prohibitions precede their positive substrings (for example, 不可以 before 可以).
    if any(marker in comparable for marker in _NEGATION_PROHIBITION_MARKERS):
        polarity = "negated"
    elif any(marker in comparable for marker in _ACCEPTANCE_MARKERS):
        polarity = "acceptance"
    else:
        polarity = "neutral"
    primary_action = min(
        (verb for verb in _ACTION_VERBS if verb in comparable),
        key=comparable.find,
        default=None,
    )
    privacy_control = next(
        (
            marker
            for marker in ("撤回", "停止", "不同意")
            if marker in comparable and _PRIVACY_CONTROL.search(text)
        ),
        None,
    )
    return _MaterialSignature(
        polarity=polarity,
        is_question=(
            "?" in text
            or "\uff1f" in text
            or text.rstrip().endswith(("吗", "么", "呢"))
            or any(marker in comparable for marker in _INTERROGATIVE_MARKERS)
        ),
        primary_action=primary_action,
        numbers=tuple(_NUMBER_PATTERN.findall(comparable)),
        privacy_control=privacy_control,
    )


def _has_time_overlap(first: TranscriptEntryRecord, second: TranscriptEntryRecord) -> bool:
    return min(first.ended_ms, second.ended_ms) > max(first.started_ms, second.started_ms)


def _time_overlap_iou(first: TranscriptEntryRecord, second: TranscriptEntryRecord) -> float:
    intersection = max(
        0, min(first.ended_ms, second.ended_ms) - max(first.started_ms, second.started_ms)
    )
    union = max(first.ended_ms, second.ended_ms) - min(first.started_ms, second.started_ms)
    return intersection / union if union > 0 else 0.0


def _containment_ratio(first: str, second: str) -> float:
    """Return the fraction of the shorter candidate present in the longer candidate."""

    if not first or not second:
        return 0.0
    shorter, longer = sorted((first, second), key=len)
    return len(shorter) / len(longer) if shorter in longer else 0.0


def _text_length_ratio(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return min(len(first), len(second)) / max(len(first), len(second))


def _canonical_entry(component: tuple[_SourceEntry, ...]) -> CanonicalTranscriptEntry:
    ordered = tuple(sorted(component, key=_source_sort_key))
    representative = min(
        ordered,
        key=lambda source: (
            *_representative_precedence(source),
            -len(source.comparable_text),
            -(source.record.ended_ms - source.record.started_ms),
            source.record.id,
        ),
    )
    source_ids = tuple(sorted(source.record.id for source in ordered))
    tags = tuple(sorted({tag for source in ordered for tag in source.tags}))
    languages = {source.language for source in ordered if source.language is not None}
    emotions = {source.emotion for source in ordered if source.emotion is not None}
    entry_type = _classify_entry(representative.text)
    return CanonicalTranscriptEntry(
        entry_id=representative.record.id,
        source_entry_ids=source_ids,
        speaker_user_id=representative.record.user_id,
        speaker_name=None,
        started_ms=min(source.record.started_ms for source in ordered),
        ended_ms=max(source.record.ended_ms for source in ordered),
        raw_texts=tuple(
            source.record.raw_text for source in sorted(ordered, key=lambda item: item.record.id)
        ),
        text=representative.text,
        entry_type=entry_type,
        language=next(iter(languages)) if len(languages) == 1 else None,
        confidence=None,
        utterance_group_id="",
        normalisation_changes=(),
        asr_tags=tags,
        emotion=next(iter(emotions)) if len(emotions) == 1 else None,
    )


def _representative_precedence(source: _SourceEntry) -> tuple[int, int, int, int]:
    revisions = source.record.revision_json
    revision_number = max(
        (_revision_number(revision) for revision in revisions),
        default=0,
    )
    source_kind = source.record.source_key.partition(":")[0].casefold()
    return (
        -int(bool(revisions)),
        -revision_number,
        -len(revisions),
        -_SOURCE_PRECEDENCE.get(source_kind, 0),
    )


def _revision_number(revision: dict[str, object]) -> int:
    values: list[int] = []
    for field in ("revision", "version", "sequence", "updated_at_ms", "revised_at_ms"):
        value = revision.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
        elif isinstance(value, str) and value.isdecimal():
            values.append(int(value))
    return max(values, default=0)


def _classify_entry(text: str) -> str:
    comparable = _normalised_characters(text)
    if not comparable:
        return "noise"
    if _PRIVACY_CONTROL.search(text):
        return "privacy_control"
    if comparable in _BACKCHANNELS:
        return "backchannel"
    return "statement"


def _assign_utterance_groups(
    entries: tuple[CanonicalTranscriptEntry, ...],
) -> tuple[CanonicalTranscriptEntry, ...]:
    ordered = tuple(sorted(entries, key=_canonical_sort_key))
    latest_by_speaker: dict[int | None, CanonicalTranscriptEntry] = {}
    assigned: list[CanonicalTranscriptEntry] = []
    for entry in ordered:
        previous = latest_by_speaker.get(entry.speaker_user_id)
        group_id = _new_group_id(entry)
        if previous is not None and _can_share_group(previous, entry, ordered):
            group_id = previous.utterance_group_id
        grouped = replace(entry, utterance_group_id=group_id)
        assigned.append(grouped)
        latest_by_speaker[grouped.speaker_user_id] = grouped
    return tuple(assigned)


def _canonical_sort_key(entry: CanonicalTranscriptEntry) -> tuple[int, int, int, str]:
    return (
        entry.started_ms,
        entry.ended_ms,
        entry.speaker_user_id if entry.speaker_user_id is not None else -1,
        entry.entry_id,
    )


def _new_group_id(entry: CanonicalTranscriptEntry) -> str:
    speaker = str(entry.speaker_user_id) if entry.speaker_user_id is not None else "bot"
    return f"group:{speaker}:{entry.entry_id}"


def _can_share_group(
    previous: CanonicalTranscriptEntry,
    current: CanonicalTranscriptEntry,
    entries: tuple[CanonicalTranscriptEntry, ...],
) -> bool:
    if current.started_ms - previous.ended_ms > UTTERANCE_GROUP_GAP_MS:
        return False
    return not any(
        candidate.speaker_user_id != current.speaker_user_id
        and candidate.entry_type not in {"backchannel", "noise", "excluded"}
        and candidate.started_ms <= current.ended_ms
        and candidate.ended_ms >= previous.started_ms
        for candidate in entries
    )


def _strip_asr_tags(text: str) -> tuple[str, tuple[str, ...], str | None, str | None]:
    tags: list[str] = []
    languages: set[str] = set()
    emotions: set[str] = set()

    def remove_angle_tag(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        lowered = token.lower()
        if lowered not in _LANGUAGE_TAGS | _EMOTION_TAGS | _INTERNAL_TAGS:
            return match.group(0)
        tags.append(token)
        if lowered in _LANGUAGE_TAGS:
            languages.add(lowered)
        if lowered in _EMOTION_TAGS:
            emotions.add(token.upper())
        return ""

    def remove_bracket_tag(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        if token.lower() not in _INTERNAL_TAGS:
            return match.group(0)
        tags.append(token)
        return ""

    without_angle_tags = _TAG_PATTERN.sub(remove_angle_tag, text)
    cleaned = _BRACKET_TAG_PATTERN.sub(remove_bracket_tag, without_angle_tags)
    normalised = re.sub(r"\s+", " ", cleaned).strip()
    return (
        normalised,
        tuple(sorted(set(tags))),
        next(iter(languages)) if len(languages) == 1 else None,
        next(iter(emotions)) if len(emotions) == 1 else None,
    )


def _normalised_characters(text: str) -> str:
    normalised = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in normalised
        if not unicodedata.category(character).startswith(("P", "Z"))
    )
