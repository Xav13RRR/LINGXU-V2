import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from lingxu.transcript.models import CanonicalTranscriptEntry
from lingxu.worker.outcome_partitions import (
    MAX_DEEPSEEK_REPAIR_OUTPUT_TOKENS,
    MAX_LLM_WALL_SECONDS_PER_JOB,
    MAX_OUTCOME_OUTPUT_TOKENS,
    MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB,
    MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB,
    LlmActionKind,
    LlmBudgetController,
    OutcomePartitionError,
    UnifiedLlmBudgetJournal,
    digest_task_hash,
    llm_action_hash,
    llm_job_hash,
    outcome_response_schema,
    outcome_response_schema_hash,
    outcome_task_hash,
    parse_partition_outcome_batch,
    parse_partition_outcome_batch_with_rows,
    plan_outcome_partitions,
    serialize_outcome_request,
)
from lingxu.worker.outcomes import (
    MAX_EVIDENCE_IDS_PER_CANDIDATE,
    MAX_MODEL_DEADLINE_CHARS,
    MAX_MODEL_ENTRY_REFS,
    MAX_MODEL_EVENT_REFS,
    MAX_MODEL_OUTCOME_ROWS,
    MAX_MODEL_OUTCOME_TEXT_CHARS,
    MAX_PRD_MODEL_OUTCOME_ROWS,
    OUTCOME_EXTRACTION_CONFIG_VERSION,
    OutcomeCapacitySubtype,
    OutcomeEventFact,
    OutcomeFailureAttempt,
    OutcomeFailureCategory,
    OutcomeFailureRequestKind,
    OutcomeFormatSubtype,
    OutcomeProviderError,
    OutcomeProviderFailure,
    OutcomeRequestKind,
    OutcomeResponseCapacityError,
    OutcomeResponseSchemaError,
    OutcomeSizeBucket,
    OutcomeTaskDisposition,
    OutcomeType,
    TopicOutcomeSet,
    classify_json_decode_failure,
    max_model_outcome_rows,
    outcome_failure_token,
    outcome_json_schema,
    outcome_system_prompt,
    parse_outcome_batch,
)
from lingxu.worker.summary import DeepSeekPostMeetingInferenceProvider
from lingxu.worker.topics import (
    APPROVED_TOPIC_BASE_URL,
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    TopicSegment,
)

DEEPSEEK_BASE_URL = APPROVED_TOPIC_BASE_URL
DEEPSEEK_MODEL = APPROVED_TOPIC_MODEL
DEEPSEEK_MODEL_DIGEST = APPROVED_TOPIC_MODEL_DIGEST


def _entry(index: int, text: str) -> CanonicalTranscriptEntry:
    return CanonicalTranscriptEntry(
        entry_id=f"e{index}",
        source_entry_ids=(f"source-{index}",),
        speaker_user_id=7,
        speaker_name="participant",
        started_ms=index * 1_000,
        ended_ms=index * 1_000 + 800,
        raw_texts=(text,),
        text=text,
        entry_type="question" if text.endswith("?") else "statement",
        language="en",
        confidence=0.95,
        utterance_group_id=f"group-{index}",
        normalisation_changes=(),
        asr_tags=(),
        emotion=None,
    )


def test_fix36_core_and_overlap_aliases_use_independent_contiguous_namespaces() -> None:
    entries = tuple(_entry(index, f"evidence {index}") for index in range(1, 4))
    original = plan_outcome_partitions(
        TopicSegment(
            "topic-fix36-aliases",
            "Fix36 aliases",
            tuple(entry.entry_id for entry in entries),
            1_000,
            3_800,
            "deterministic_fallback",
        ),
        entries,
        (),
    )[0]
    partition = original.model_copy(
        update={
            "core_entries": (original.core_entries[1],),
            "overlap_entries": (original.core_entries[0], original.core_entries[2]),
        }
    )

    payload = json.loads(
        serialize_outcome_request(partition, OutcomeRequestKind.QUESTION).decode("utf-8")
    )

    assert [row[0] for row in payload["c"]] == ["e0"]
    assert [row[0] for row in payload["o"]] == ["o0", "o1"]
    assert outcome_response_schema(partition, OutcomeRequestKind.QUESTION)["properties"][
        "candidates"
    ]["items"]["properties"]["source_entry_refs"]["items"]["enum"] == ["e0"]


def test_fix36_da_headroom_and_text_capacity_remain_globally_bounded() -> None:
    assert max_model_outcome_rows(OutcomeRequestKind.DECISION_AGREEMENT, 6) == 5
    assert max_model_outcome_rows(OutcomeRequestKind.DECISION_AGREEMENT, 7) == 4
    assert max_model_outcome_rows(OutcomeRequestKind.DECISION_AGREEMENT, 8) == 3
    assert MAX_MODEL_OUTCOME_TEXT_CHARS == 120

    rows = [
        {
            "type": "decision" if index % 2 == 0 else "agreement",
            "text": "x" * MAX_MODEL_OUTCOME_TEXT_CHARS,
            "source_entry_refs": [f"e{index}"],
            "source_event_refs": [],
        }
        for index in range(4)
    ]
    assert (
        len(
            parse_outcome_batch(
                json.dumps({"complete": True, "candidates": rows}),
                OutcomeRequestKind.DECISION_AGREEMENT,
                topic_partition_count=6,
            )
        )
        == 4
    )
    with pytest.raises(OutcomeResponseCapacityError) as saturated:
        parse_outcome_batch(
            json.dumps({"complete": True, "candidates": rows}),
            OutcomeRequestKind.DECISION_AGREEMENT,
            topic_partition_count=7,
        )
    assert saturated.value.subtype is OutcomeCapacitySubtype.ROW_SATURATION
    with pytest.raises(OutcomeResponseCapacityError) as overlong:
        parse_outcome_batch(
            json.dumps(
                {
                    "complete": True,
                    "candidates": [
                        {
                            "text": "x" * (MAX_MODEL_OUTCOME_TEXT_CHARS + 1),
                            "source_entry_refs": ["e0"],
                        }
                    ],
                }
            ),
            OutcomeRequestKind.QUESTION,
        )
    assert overlong.value.subtype is OutcomeCapacitySubtype.TEXT_CHARS


def test_fix36_prompt_requires_minimal_evidence_compact_text_and_json_escaping() -> None:
    for kind in OutcomeRequestKind:
        prompt = outcome_system_prompt(kind)
        assert "最少且充分" in prompt
        assert "通常只需1项" in prompt
        assert "text目标不超过80字符" in prompt
        assert "JSON字符串转义" in prompt


def _candidate(
    candidate_id: str,
    outcome_type: str,
    source_entry_id: str,
    *,
    owner_user_id: int | None = None,
    owner_name: str | None = None,
    source_event_ref: str | None = None,
) -> dict[str, Any]:
    numeric_source = source_entry_id.removeprefix("e")
    source_ref = f"e{max(0, int(numeric_source) - 1)}" if numeric_source.isdigit() else "e999"
    return {
        "id": candidate_id,
        "type": outcome_type,
        "text": f"supported {outcome_type}",
        "owner_ref": "s0" if owner_user_id is not None else None,
        "owner_name": owner_name,
        "deadline": None,
        "source_entry_refs": [source_ref],
        "source_event_refs": [source_event_ref] if source_event_ref is not None else [],
        "confidence": 0.9,
    }


async def _server(
    responses: Iterator[dict[str, Any] | str],
) -> tuple[TestServer, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": APPROVED_TOPIC_MODEL,
                        "object": "model",
                        "owned_by": "deepseek",
                    }
                ],
            }
        )

    async def complete(request: web.Request) -> web.Response:
        request_payload = await request.json()
        requests.append(request_payload)
        response = next(responses)
        if isinstance(response, dict) and isinstance(response.get("candidates"), list):
            context = json.loads(request_payload["messages"][1]["content"])
            kind = OutcomeRequestKind(context["k"])
            candidates: list[dict[str, object]] = []
            convertible = True
            for candidate in response["candidates"]:
                if not isinstance(candidate, dict) or not {
                    "type",
                    "text",
                    "source_entry_refs",
                } <= set(candidate):
                    convertible = False
                    break
                if kind is OutcomeRequestKind.DECISION_AGREEMENT:
                    candidates.append(
                        {
                            "type": candidate["type"],
                            "text": candidate["text"],
                            "source_entry_refs": candidate["source_entry_refs"],
                            "source_event_refs": candidate.get("source_event_refs", []),
                        }
                    )
                elif kind is OutcomeRequestKind.ACTION:
                    candidates.append(
                        {
                            "text": candidate["text"],
                            "source_entry_refs": candidate["source_entry_refs"],
                            "owner_ref": candidate.get("owner_ref"),
                            "deadline": candidate.get("deadline"),
                        }
                    )
                elif kind is OutcomeRequestKind.QUESTION:
                    candidates.append(
                        {
                            "text": candidate["text"],
                            "source_entry_refs": candidate["source_entry_refs"],
                        }
                    )
                else:
                    candidates.append(
                        {
                            "type": candidate["type"],
                            "text": candidate["text"],
                            "source_entry_refs": candidate["source_entry_refs"],
                        }
                    )
            if convertible:
                response = {"complete": True, "candidates": candidates}
        content = response if isinstance(response, str) else json.dumps(response)
        return web.json_response(
            {
                "id": "content-free-outcome-test",
                "object": "chat.completion",
                "model": APPROVED_TOPIC_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_get("/models", models)
    app.router.add_post("/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    return server, requests


async def _deepseek_outcome_server(
    responses: Iterator[tuple[dict[str, Any] | str, str | None, str]],
) -> tuple[TestServer, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": DEEPSEEK_MODEL, "object": "model", "owned_by": "deepseek"}],
            }
        )

    async def complete(request: web.Request) -> web.Response:
        requests.append(await request.json())
        response, reasoning_content, finish_reason = next(responses)
        content = response if isinstance(response, str) else json.dumps(response)
        return web.json_response(
            {
                "id": "content-free-outcome-test",
                "object": "chat.completion",
                "model": DEEPSEEK_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": reasoning_content,
                        },
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_get("/models", models)
    app.router.add_post("/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    return server, requests


def _deepseek_outcome_provider(server: TestServer) -> DeepSeekPostMeetingInferenceProvider:
    provider = DeepSeekPostMeetingInferenceProvider(
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        expected_model_digest=DEEPSEEK_MODEL_DIGEST,
        api_key="PRIVATE_DEEPSEEK_TEST_KEY",
        max_output_tokens=32_768,
        allow_remote=True,
    )
    provider._models_endpoint = str(server.make_url("/models"))
    provider._endpoint = str(server.make_url("/chat/completions"))
    return provider


async def _boundary_server(
    *,
    chat_status: int = 200,
    chat_body: bytes | None = None,
    chat_delay_seconds: float = 0,
    chat_abort: bool = False,
    tags_digest: str = APPROVED_TOPIC_MODEL_DIGEST,
    tags_body: bytes | None = None,
    tags_delay_seconds: float = 0,
    tags_status: int = 200,
) -> TestServer:
    async def models(_request: web.Request) -> web.Response:
        if tags_delay_seconds:
            import asyncio

            await asyncio.sleep(tags_delay_seconds)
        if tags_body is not None:
            return web.Response(status=tags_status, body=tags_body, content_type="application/json")
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": (
                            APPROVED_TOPIC_MODEL
                            if tags_digest == APPROVED_TOPIC_MODEL_DIGEST
                            else tags_digest
                        ),
                        "object": "model",
                        "owned_by": "deepseek",
                    }
                ],
            },
            status=tags_status,
        )

    async def complete(_request: web.Request) -> web.Response:
        if chat_abort:
            assert _request.transport is not None
            _request.transport.abort()
            raise ConnectionResetError("closed test transport")
        if chat_delay_seconds:
            import asyncio

            await asyncio.sleep(chat_delay_seconds)
        return web.Response(
            status=chat_status,
            body=chat_body
            if chat_body is not None
            else json.dumps(
                {
                    "id": "content-free-boundary-test",
                    "object": "chat.completion",
                    "model": APPROVED_TOPIC_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"complete":true,"candidates":[]}',
                            },
                        }
                    ],
                }
            ).encode(),
            content_type="application/json",
        )

    app = web.Application()
    app.router.add_get("/models", models)
    app.router.add_post("/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    return server


def _provider(server: TestServer, **kwargs: Any) -> DeepSeekPostMeetingInferenceProvider:
    api_key = kwargs.pop("api_key", "PRIVATE_DEEPSEEK_TEST_KEY")
    allow_remote = kwargs.pop("allow_remote", True)
    provider = DeepSeekPostMeetingInferenceProvider(
        base_url=APPROVED_TOPIC_BASE_URL,
        model=APPROVED_TOPIC_MODEL,
        expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
        api_key=api_key,
        allow_remote=allow_remote,
        **kwargs,
    )
    provider._models_endpoint = str(server.make_url("/models"))
    provider._endpoint = str(server.make_url("/chat/completions"))
    return provider


def _assert_deepseek_repair_payload(
    initial: dict[str, Any],
    repair: dict[str, Any],
) -> None:
    for key in ("model", "stream", "response_format"):
        assert repair[key] == initial[key]
    assert initial["thinking"] == {"type": "disabled"}
    assert initial["temperature"] == 0
    assert initial["max_tokens"] == MAX_OUTCOME_OUTPUT_TOKENS
    assert repair["thinking"] == {"type": "enabled"}
    assert repair["reasoning_effort"] == "high"
    assert repair["max_tokens"] == MAX_DEEPSEEK_REPAIR_OUTPUT_TOKENS
    assert "temperature" not in repair


def _bind_provider(
    provider: DeepSeekPostMeetingInferenceProvider,
    partitions: tuple[Any, ...],
) -> tuple[LlmBudgetController, str]:
    job_hash = llm_job_hash(canonical_hash="a" * 64, pipeline_revision="b" * 64)
    expected = (
        digest_task_hash(job_hash),
        *(
            outcome_task_hash(job_hash, partition, kind)
            for partition in partitions
            for kind in OutcomeRequestKind
        ),
    )
    controller = LlmBudgetController(
        UnifiedLlmBudgetJournal.new(
            job_hash=job_hash,
            expected_task_hashes=expected,
        ),
        lambda _journal: None,
    )
    provider.bind_llm_budget(controller)
    return controller, job_hash


async def _extract_all(
    provider: DeepSeekPostMeetingInferenceProvider,
    *,
    topic: TopicSegment,
    entries: tuple[CanonicalTranscriptEntry, ...],
    events: tuple[OutcomeEventFact, ...],
    allowed_participant_ids: frozenset[int],
) -> TopicOutcomeSet:
    partitions = plan_outcome_partitions(
        topic,
        entries,
        events,
        tuple(sorted(allowed_participant_ids)),
    )
    _bind_provider(provider, partitions)
    collected: list[TopicOutcomeSet] = []
    for partition in partitions:
        for kind in OutcomeRequestKind:
            collected.append(
                await provider.extract_topic_outcome_kind(
                    partition=partition,
                    kind=kind,
                    request_bytes=serialize_outcome_request(partition, kind),
                )
            )
    results = tuple(collected)
    failures = tuple(failure for result in results for failure in result.provider_failures)
    return TopicOutcomeSet(
        topic_id=topic.id,
        candidates=tuple(candidate for result in results for candidate in result.candidates),
        extraction_method=("deepseek" if not failures else "deepseek_partial"),
        failure_types=tuple(
            sorted({failure for result in results for failure in result.failure_types})
        ),
        provider_failures=failures,
    )


async def _request_failure(
    provider: DeepSeekPostMeetingInferenceProvider,
) -> OutcomeProviderFailure:
    job_hash = llm_job_hash(canonical_hash="a" * 64, pipeline_revision="b" * 64)
    task_hash = "c" * 64
    controller = LlmBudgetController(
        UnifiedLlmBudgetJournal.new(
            job_hash=job_hash,
            expected_task_hashes=(task_hash,),
        ).reserve_task(task_hash),
        lambda _journal: None,
    )
    provider.bind_llm_budget(controller)
    with pytest.raises(OutcomeProviderError) as caught:
        await provider._request_outcome_content(
            {},
            kind=OutcomeRequestKind.ACTION,
            attempt=OutcomeFailureAttempt.INITIAL,
            task_hash=task_hash,
        )
    return caught.value.failure


def test_outcome_schemas_are_closed_and_request_specific() -> None:
    schema = outcome_json_schema(OutcomeRequestKind.ACTION)
    decision_schema = outcome_json_schema(OutcomeRequestKind.DECISION_AGREEMENT)

    assert schema["additionalProperties"] is False
    assert set(schema) == {"type", "properties", "required", "additionalProperties"}
    assert schema["required"] == ["complete", "candidates"]
    action_candidate = schema["properties"]["candidates"]["items"]
    decision_candidate = decision_schema["properties"]["candidates"]["items"]
    assert action_candidate["additionalProperties"] is False
    assert set(action_candidate["required"]) == {
        "text",
        "source_entry_refs",
        "owner_ref",
        "deadline",
    }
    assert "type" not in action_candidate["properties"]
    assert decision_candidate["properties"]["type"]["enum"] == ["decision", "agreement"]
    assert schema["properties"]["candidates"]["maxItems"] == MAX_MODEL_OUTCOME_ROWS == 3
    assert (
        action_candidate["properties"]["text"]["maxLength"] == MAX_MODEL_OUTCOME_TEXT_CHARS == 120
    )
    assert (
        action_candidate["properties"]["deadline"]["anyOf"][0]["maxLength"]
        == MAX_MODEL_DEADLINE_CHARS
        == 32
    )
    assert (
        action_candidate["properties"]["source_entry_refs"]["maxItems"]
        == MAX_MODEL_ENTRY_REFS
        == MAX_EVIDENCE_IDS_PER_CANDIDATE
        == 12
    )
    assert (
        decision_candidate["properties"]["source_event_refs"]["maxItems"]
        == MAX_MODEL_EVENT_REFS
        == 4
    )


def test_fix35_wire_evidence_cap_matches_candidate_authority_without_truncation() -> None:
    accepted_refs = [f"e{index}" for index in range(MAX_EVIDENCE_IDS_PER_CANDIDATE)]
    accepted = parse_outcome_batch(
        json.dumps(
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "risk",
                        "text": "Synthetic risk",
                        "source_entry_refs": accepted_refs,
                    }
                ],
            }
        ),
        OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
    )

    assert accepted[0].source_entry_refs == tuple(accepted_refs)
    assert (
        outcome_json_schema(OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT)["properties"][
            "candidates"
        ]["items"]["properties"]["source_entry_refs"]["maxItems"]
        == MAX_EVIDENCE_IDS_PER_CANDIDATE
    )
    prompt = outcome_system_prompt(OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT)
    assert f"每组entry证据引用最多{MAX_MODEL_ENTRY_REFS}项" in prompt
    assert f"event证据引用最多{MAX_MODEL_EVENT_REFS}项" in prompt

    with pytest.raises(OutcomeResponseCapacityError) as caught:
        parse_outcome_batch(
            json.dumps(
                {
                    "complete": True,
                    "candidates": [
                        {
                            "type": "risk",
                            "text": "Synthetic risk",
                            "source_entry_refs": [*accepted_refs, "ez"],
                        }
                    ],
                }
            ),
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
        )
    assert caught.value.subtype is OutcomeCapacitySubtype.ENTRY_REFS
    assert MAX_OUTCOME_OUTPUT_TOKENS == 768
    assert MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB == 600
    assert MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB == 601
    assert MAX_LLM_WALL_SECONDS_PER_JOB == 5_400


@pytest.mark.parametrize("topic_partition_count", range(1, 13))
def test_fix33_partition_aware_headroom_never_exceeds_final_topic_capacity(
    topic_partition_count: int,
) -> None:
    entries = (_entry(0, "supported evidence"),)
    partition = plan_outcome_partitions(
        TopicSegment(
            id="topic-capacity",
            title="Capacity",
            entry_ids=(entries[0].entry_id,),
            started_ms=0,
            ended_ms=entries[0].ended_ms,
            method="deterministic_fallback",
        ),
        entries,
        (),
    )[0].model_copy(update={"topic_partition_count": topic_partition_count})
    accepted = {
        kind: outcome_response_schema(partition, kind)["properties"]["candidates"]["maxItems"] - 1
        for kind in OutcomeRequestKind
    }

    expected = (
        (4, 2, 2, 4)
        if topic_partition_count <= 6
        else (3, 2, 2, 4)
        if topic_partition_count == 7
        else (2, 2, 2, 3)
        if topic_partition_count == 8
        else (2, 2, 1, 3)
        if topic_partition_count <= 10
        else (2, 1, 1, 2)
    )
    assert tuple(accepted[kind] for kind in OutcomeRequestKind) == expected
    assert topic_partition_count * sum(accepted.values()) <= 80


def test_fix33_exact_wire_rows_fold_losslessly_but_raw_headroom_precedes_fold() -> None:
    entries = (_entry(0, "supported evidence"),)
    topic = TopicSegment(
        id="topic-fold",
        title="Fold",
        entry_ids=(entries[0].entry_id,),
        started_ms=0,
        ended_ms=entries[0].ended_ms,
        method="deterministic_fallback",
    )
    partition = plan_outcome_partitions(topic, entries, ())[0]
    row = {"text": "supported question", "source_entry_refs": ["e0"]}
    parsed = parse_partition_outcome_batch_with_rows(
        json.dumps({"complete": True, "candidates": [row, row]}),
        OutcomeRequestKind.QUESTION,
        partition,
    )

    assert len(parsed.candidates) == 1
    assert parsed.input_row_count == 2
    assert parsed.row_groups[0].retained_ordinal == 0
    assert parsed.row_groups[0].contributor_ordinals == (0, 1)

    with pytest.raises(OutcomeResponseCapacityError) as caught:
        parse_partition_outcome_batch_with_rows(
            json.dumps({"complete": True, "candidates": [row, row, row]}),
            OutcomeRequestKind.QUESTION,
            partition,
        )
    assert caught.value.subtype is OutcomeCapacitySubtype.ROW_SATURATION


def test_fix33_wire_divergence_and_canonical_alias_collision_remain_closed() -> None:
    entries = (_entry(0, "supported evidence"),)
    topic = TopicSegment(
        id="topic-divergence",
        title="Divergence",
        entry_ids=(entries[0].entry_id,),
        started_ms=0,
        ended_ms=entries[0].ended_ms,
        method="deterministic_fallback",
    )
    partition = plan_outcome_partitions(topic, entries, ())[0]
    with pytest.raises(OutcomeResponseSchemaError) as divergent:
        parse_partition_outcome_batch_with_rows(
            json.dumps(
                {
                    "complete": True,
                    "candidates": [
                        {"text": "first", "source_entry_refs": ["e0"]},
                        {"text": "second", "source_entry_refs": ["e0"]},
                    ],
                }
            ),
            OutcomeRequestKind.QUESTION,
            partition,
        )
    assert divergent.value.subtype is OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT


@pytest.mark.asyncio
async def test_fix33_provider_preserves_original_ordinals_and_private_fold_provenance() -> None:
    entry = _entry(0, "supported evidence")
    topic = TopicSegment(
        id="topic-provider-fold",
        title="Provider fold",
        entry_ids=(entry.entry_id,),
        started_ms=0,
        ended_ms=entry.ended_ms,
        method="deterministic_fallback",
    )
    duplicate = {
        "type": "decision",
        "text": "supported decision",
        "source_entry_refs": ["e0"],
        "source_event_refs": [],
    }
    distinct = {
        "type": "agreement",
        "text": "supported agreement",
        "source_entry_refs": ["e0"],
        "source_event_refs": [],
    }
    server, requests = await _server(
        iter(({"complete": True, "candidates": [duplicate, duplicate, distinct]},))
    )
    partition = plan_outcome_partitions(topic, (entry,), ())[0]
    try:
        provider = _provider(server)
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.DECISION_AGREEMENT,
            request_bytes=serialize_outcome_request(
                partition, OutcomeRequestKind.DECISION_AGREEMENT
            ),
        )
    finally:
        await server.close()

    assert len(requests) == 1
    assert result.provider_failures == ()
    assert len(result.candidates) == 2
    assert "-01-" in result.candidates[0].id
    assert "-03-" in result.candidates[1].id
    assert result.wire_input_row_count == 3
    assert tuple(item.contributor_ordinals for item in result.wire_row_resolutions) == (
        (0, 1),
        (2,),
    )
    serialized = result.model_dump_json()
    assert "wire_input_row_count" not in serialized
    assert "wire_row_resolutions" not in serialized


def test_fix32_nonempty_alias_items_are_exact_enums_with_equivalent_acceptance() -> None:
    aliases = ("e0", "e1", "ez", "e10")
    item_schema = outcome_json_schema(
        OutcomeRequestKind.QUESTION,
        entry_aliases=aliases,
    )["properties"]["candidates"]["items"]["properties"]["source_entry_refs"]["items"]

    assert item_schema == {"type": "string", "enum": list(aliases)}

    old_schema = {
        "type": "string",
        "pattern": r"^e[0-9a-z]+$",
        "maxLength": 12,
        "enum": list(aliases),
    }
    bounded_values: tuple[object, ...] = (
        None,
        False,
        0,
        "",
        "e0",
        "e1",
        "ez",
        "e10",
        "e2",
        "o0",
        "E0",
        "e_0",
        "e" + "a" * 12,
        ["e0"],
    )

    def accepts(schema: dict[str, object], value: object) -> bool:
        import re

        if schema.get("type") == "string" and not isinstance(value, str):
            return False
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            return False
        maximum = schema.get("maxLength")
        if isinstance(value, str) and isinstance(maximum, int) and len(value) > maximum:
            return False
        pattern = schema.get("pattern")
        if (
            isinstance(value, str)
            and isinstance(pattern, str)
            and re.fullmatch(pattern, value) is None
        ):
            return False
        return True

    assert [accepts(old_schema, value) for value in bounded_values] == [
        accepts(item_schema, value) for value in bounded_values
    ]


def test_fix32_unbounded_alias_schema_keeps_shape_constraints() -> None:
    item_schema = outcome_json_schema(OutcomeRequestKind.QUESTION)["properties"]["candidates"][
        "items"
    ]["properties"]["source_entry_refs"]["items"]

    assert item_schema == {
        "type": "string",
        "pattern": r"^e[0-9a-z]+$",
        "maxLength": 12,
    }


def test_fix32_entry_event_and_owner_authorities_use_exact_enum_items() -> None:
    decision = outcome_json_schema(
        OutcomeRequestKind.DECISION_AGREEMENT,
        entry_aliases=("e0", "e1"),
        event_aliases=("v0",),
    )["properties"]["candidates"]["items"]["properties"]
    action = outcome_json_schema(
        OutcomeRequestKind.ACTION,
        entry_aliases=("e0",),
        owner_aliases=("s0", "s1"),
    )["properties"]["candidates"]["items"]["properties"]

    assert decision["source_entry_refs"]["items"] == {
        "type": "string",
        "enum": ["e0", "e1"],
    }
    assert decision["source_event_refs"]["items"] == {
        "type": "string",
        "enum": ["v0"],
    }
    assert action["owner_ref"]["anyOf"][0] == {
        "type": "string",
        "enum": ["s0", "s1"],
    }


def test_fix32_empty_alias_authority_stays_unsatisfiable() -> None:
    decision_candidate = outcome_json_schema(
        OutcomeRequestKind.DECISION_AGREEMENT,
        entry_aliases=(),
        event_aliases=(),
    )["properties"]["candidates"]["items"]
    action_properties = outcome_json_schema(
        OutcomeRequestKind.ACTION,
        entry_aliases=(),
        owner_aliases=(),
    )["properties"]["candidates"]["items"]["properties"]

    assert decision_candidate["not"] == {}
    assert decision_candidate["properties"]["source_entry_refs"]["maxItems"] == 0
    assert decision_candidate["properties"]["source_event_refs"]["maxItems"] == 0
    assert action_properties["source_entry_refs"]["maxItems"] == 0
    assert action_properties["owner_ref"] == {"type": "null"}


def test_fix32_does_not_add_unsupported_candidate_array_unique_items() -> None:
    for kind in OutcomeRequestKind:
        candidate_array = outcome_json_schema(kind)["properties"]["candidates"]
        assert "uniqueItems" not in candidate_array


def test_fix26_request_kind_row_headroom_and_topic_worst_case_are_exact() -> None:
    candidates = [
        {"type": outcome_type, "text": outcome_type, "source_entry_refs": ["e0"]}
        for outcome_type in ("proposal", "risk", "disagreement")
    ]
    assert (
        len(
            parse_outcome_batch(
                json.dumps({"complete": True, "candidates": candidates}),
                OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            )
        )
        == 3
    )
    with pytest.raises(OutcomeResponseCapacityError) as prd_capacity:
        parse_outcome_batch(
            json.dumps({"complete": True, "candidates": [*candidates, candidates[0]]}),
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
        )
    assert prd_capacity.value.subtype is OutcomeCapacitySubtype.ROW_SATURATION

    question = {"text": "Question", "source_entry_refs": ["e0"]}
    with pytest.raises(OutcomeResponseCapacityError) as question_capacity:
        parse_outcome_batch(
            json.dumps({"complete": True, "candidates": [question] * 3}),
            OutcomeRequestKind.QUESTION,
        )
    assert question_capacity.value.subtype is OutcomeCapacitySubtype.ROW_SATURATION
    assert MAX_MODEL_OUTCOME_ROWS == 3
    assert MAX_PRD_MODEL_OUTCOME_ROWS == 4
    assert 8 * (2 + 2 + 2 + 3) <= 80


def test_fix26_partition_schema_closes_core_event_and_owner_aliases() -> None:
    entries = (_entry(1, "core evidence"), _entry(2, "overlap evidence"))
    topic = TopicSegment(
        "topic-001",
        "Synthetic topic",
        tuple(entry.entry_id for entry in entries),
        1_000,
        2_800,
        "deterministic_fallback",
    )
    event = OutcomeEventFact(
        sequence_number=1,
        topic_id=topic.id,
        subject="Synthetic result",
        result="passed",
        source_entry_ids=(entries[0].entry_id,),
    )
    original = plan_outcome_partitions(
        topic,
        entries,
        (event,),
        allowed_participant_ids=(7, 8),
    )[0]
    partition = original.model_copy(
        update={
            "core_entries": original.core_entries[:1],
            "overlap_entries": original.core_entries[1:],
        }
    )

    action_schema = outcome_response_schema(partition, OutcomeRequestKind.ACTION)
    action = action_schema["properties"]["candidates"]["items"]["properties"]
    decision_schema = outcome_response_schema(partition, OutcomeRequestKind.DECISION_AGREEMENT)
    decision = decision_schema["properties"]["candidates"]["items"]["properties"]

    assert action["source_entry_refs"]["items"]["enum"] == ["e0"]
    assert action["owner_ref"]["anyOf"][0]["enum"] == ["s0"]
    assert decision["source_event_refs"]["items"]["enum"] == ["v0"]
    assert action_schema["properties"]["candidates"]["maxItems"] == 3
    assert (
        outcome_response_schema(partition, OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT)[
            "properties"
        ]["candidates"]["maxItems"]
        == 5
    )

    with pytest.raises(OutcomeResponseSchemaError) as entry_authority:
        parse_partition_outcome_batch(
            json.dumps(
                {
                    "complete": True,
                    "candidates": [{"text": "Question", "source_entry_refs": ["e1"]}],
                }
            ),
            OutcomeRequestKind.QUESTION,
            partition,
        )
    assert entry_authority.value.subtype is OutcomeFormatSubtype.SCHEMA_ENTRY_AUTHORITY
    with pytest.raises(OutcomeResponseSchemaError) as owner_authority:
        parse_partition_outcome_batch(
            json.dumps(
                {
                    "complete": True,
                    "candidates": [
                        {
                            "text": "Action",
                            "source_entry_refs": ["e0"],
                            "owner_ref": "s1",
                            "deadline": None,
                        }
                    ],
                }
            ),
            OutcomeRequestKind.ACTION,
            partition,
        )
    assert owner_authority.value.subtype is OutcomeFormatSubtype.SCHEMA_OWNER_AUTHORITY


@pytest.mark.parametrize(
    ("kind", "candidate", "kwargs", "subtype"),
    (
        (
            OutcomeRequestKind.QUESTION,
            {"text": "Question", "source_entry_refs": ["e1"]},
            {"allowed_entry_aliases": ("e0",)},
            OutcomeFormatSubtype.SCHEMA_ENTRY_AUTHORITY,
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "type": "decision",
                "text": "Decision",
                "source_entry_refs": [],
                "source_event_refs": ["v1"],
            },
            {"allowed_entry_aliases": (), "allowed_event_aliases": ("v0",)},
            OutcomeFormatSubtype.SCHEMA_EVENT_AUTHORITY,
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "text": "Action",
                "source_entry_refs": ["e0"],
                "owner_ref": "s1",
                "deadline": None,
            },
            {"allowed_entry_aliases": ("e0",), "allowed_owner_aliases": ("s0",)},
            OutcomeFormatSubtype.SCHEMA_OWNER_AUTHORITY,
        ),
    ),
)
def test_fix30_authority_subtype_identifies_exact_alias_namespace(
    kind: OutcomeRequestKind,
    candidate: dict[str, object],
    kwargs: dict[str, tuple[str, ...]],
    subtype: OutcomeFormatSubtype,
) -> None:
    with pytest.raises(OutcomeResponseSchemaError) as captured:
        parse_outcome_batch(
            json.dumps({"complete": True, "candidates": [candidate]}),
            kind,
            **kwargs,
        )

    assert captured.value.subtype is subtype


def test_fix26_event_only_partition_accepts_exact_event_authority_without_entry_fabrication() -> (
    None
):
    entry = _entry(1, "event source")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    event = OutcomeEventFact(
        sequence_number=1,
        topic_id=topic.id,
        subject="Synthetic result",
        result="passed",
        source_entry_ids=(entry.entry_id,),
    )
    original = plan_outcome_partitions(topic, (entry,), (event,))[0]
    partition = original.model_copy(
        update={
            "core_entries": (),
            "overlap_entries": original.core_entries,
            "core_events": (event,),
        }
    )
    schema = outcome_response_schema(partition, OutcomeRequestKind.DECISION_AGREEMENT)
    properties = schema["properties"]["candidates"]["items"]["properties"]
    assert properties["source_entry_refs"]["maxItems"] == 0
    assert properties["source_event_refs"]["items"]["enum"] == ["v0"]
    parsed = parse_partition_outcome_batch(
        json.dumps(
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "Synthetic result",
                        "source_entry_refs": [],
                        "source_event_refs": ["v0"],
                    }
                ],
            }
        ),
        OutcomeRequestKind.DECISION_AGREEMENT,
        partition,
    )
    assert len(parsed) == 1
    assert parsed[0].source_entry_refs == ()
    assert parsed[0].source_event_refs == ("v0",)


def test_fix27_empty_authorities_use_static_zero_capacity_and_null_owner_schema() -> None:
    decision = outcome_json_schema(
        OutcomeRequestKind.DECISION_AGREEMENT,
        entry_aliases=(),
        event_aliases=("v0",),
        owner_aliases=(),
    )["properties"]["candidates"]["items"]
    assert "anyOf" not in decision
    assert decision["properties"]["source_entry_refs"]["maxItems"] == 0
    assert decision["properties"]["source_event_refs"]["minItems"] == 1

    action = outcome_json_schema(
        OutcomeRequestKind.ACTION,
        entry_aliases=("e0",),
        event_aliases=(),
        owner_aliases=(),
    )["properties"]["candidates"]["items"]["properties"]
    assert action["owner_ref"] == {"type": "null"}

    impossible = outcome_json_schema(
        OutcomeRequestKind.DECISION_AGREEMENT,
        entry_aliases=(),
        event_aliases=(),
        owner_aliases=(),
    )["properties"]["candidates"]["items"]
    assert impossible["not"] == {}


def test_fix27_unbounded_decision_schema_requires_entry_or_event_evidence() -> None:
    candidate_schema = outcome_json_schema(OutcomeRequestKind.DECISION_AGREEMENT)["properties"][
        "candidates"
    ]["items"]
    assert candidate_schema["anyOf"] == [
        {"properties": {"source_entry_refs": {"minItems": 1}}},
        {"properties": {"source_event_refs": {"minItems": 1}}},
    ]


@pytest.mark.parametrize(
    ("payload", "expected_subtype"),
    (
        ([], OutcomeFormatSubtype.SCHEMA_ROOT),
        (
            {"complete": True, "candidates": [None]},
            OutcomeFormatSubtype.SCHEMA_CANDIDATE_FIELDS,
        ),
        (
            {
                "complete": True,
                "candidates": [{"type": 1, "text": "x", "source_entry_refs": ["e0"]}],
            },
            OutcomeFormatSubtype.SCHEMA_TYPE,
        ),
        (
            {
                "complete": True,
                "candidates": [{"type": "risk", "text": None, "source_entry_refs": ["e0"]}],
            },
            OutcomeFormatSubtype.SCHEMA_TEXT,
        ),
        (
            {
                "complete": True,
                "candidates": [{"type": "risk", "text": "x", "source_entry_refs": "e0"}],
            },
            OutcomeFormatSubtype.SCHEMA_ENTRY_REFS,
        ),
        (
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "x",
                        "source_entry_refs": [],
                        "source_event_refs": "v0",
                    }
                ],
            },
            OutcomeFormatSubtype.SCHEMA_EVENT_REFS,
        ),
        (
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "x",
                        "source_entry_refs": [],
                        "source_event_refs": [],
                    }
                ],
            },
            OutcomeFormatSubtype.SCHEMA_EVIDENCE_EMPTY,
        ),
    ),
)
def test_fix27_schema_shape_branches_have_closed_content_safe_subtypes(
    payload: object,
    expected_subtype: OutcomeFormatSubtype,
) -> None:
    kind = (
        OutcomeRequestKind.DECISION_AGREEMENT
        if expected_subtype
        in {OutcomeFormatSubtype.SCHEMA_EVENT_REFS, OutcomeFormatSubtype.SCHEMA_EVIDENCE_EMPTY}
        else OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT
    )
    with pytest.raises(OutcomeResponseSchemaError) as captured:
        parse_outcome_batch(
            json.dumps(payload),
            kind,
            allowed_entry_aliases=("e0",),
            allowed_event_aliases=("v0",),
            allowed_owner_aliases=(),
        )
    assert captured.value.subtype is expected_subtype
    assert "x" not in str(captured.value)


def test_fix26_dynamic_schema_hash_changes_with_partition_authority() -> None:
    entries = (_entry(1, "one"), _entry(2, "two"))
    topic = TopicSegment(
        "topic-001",
        "Synthetic topic",
        tuple(entry.entry_id for entry in entries),
        1_000,
        2_800,
        "deterministic_fallback",
    )
    full = plan_outcome_partitions(topic, entries, ())[0]
    reduced = full.model_copy(update={"core_entries": full.core_entries[:1]})
    assert outcome_response_schema_hash(full, OutcomeRequestKind.QUESTION) != (
        outcome_response_schema_hash(reduced, OutcomeRequestKind.QUESTION)
    )


@pytest.mark.asyncio
async def test_fix35_real_provider_accepts_four_prd_rows_and_twelve_exact_entry_refs() -> None:
    entries = tuple(_entry(index, f"evidence {index}") for index in range(1, 13))
    topic = TopicSegment(
        "topic-001",
        "Synthetic topic",
        tuple(entry.entry_id for entry in entries),
        1_000,
        12_800,
        "deterministic_fallback",
    )
    aliases = [*(f"e{index}" for index in range(10)), "ea", "eb"]
    response = {
        "complete": True,
        "candidates": [
            {"type": "proposal", "text": "Proposal", "source_entry_refs": aliases},
            {"type": "risk", "text": "Risk", "source_entry_refs": aliases},
            {
                "type": "disagreement",
                "text": "Disagreement",
                "source_entry_refs": aliases,
            },
            {
                "type": "proposal",
                "text": "Second proposal",
                "source_entry_refs": list(reversed(aliases)),
            },
        ],
    }
    server, requests = await _server(iter((response,)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, entries, (), target_context_chars=6_000)[0]
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            request_bytes=serialize_outcome_request(
                partition, OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT
            ),
        )
    finally:
        await server.close()

    assert len(requests) == 1
    assert provider.actual_chat_requests == controller.journal.actual_inference_calls == 1
    assert len(result.candidates) == 4
    assert result.provider_failures == ()
    schema = outcome_response_schema(partition, OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT)
    assert schema["properties"]["candidates"]["maxItems"] == 5
    assert (
        schema["properties"]["candidates"]["items"]["properties"]["source_entry_refs"]["maxItems"]
        == 12
    )
    assert (
        schema["properties"]["candidates"]["items"]["properties"]["source_entry_refs"]["items"][
            "enum"
        ]
        == aliases
    )


@pytest.mark.asyncio
async def test_fix33_five_prd_rows_are_terminal_capacity_after_two_repairs() -> None:
    entry = _entry(1, "evidence")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    candidate = {"type": "risk", "text": "Risk", "source_entry_refs": ["e0"]}
    response = {"complete": True, "candidates": [candidate] * 5}
    server, requests = await _server(iter((response, response, response)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            request_bytes=serialize_outcome_request(
                partition, OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT
            ),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert provider.actual_chat_requests == controller.journal.actual_inference_calls == 3
    assert result.candidates == ()
    assert len(result.provider_failures) == 3
    assert [failure.attempt for failure in result.provider_failures] == [
        OutcomeFailureAttempt.INITIAL,
        OutcomeFailureAttempt.REPAIR,
        OutcomeFailureAttempt.REPAIR,
    ]
    assert all(
        failure.capacity_subtype is OutcomeCapacitySubtype.ROW_SATURATION
        for failure in result.provider_failures
    )


def test_outcome_completion_signal_is_strict_and_saturation_discards_batch() -> None:
    candidate = {"text": "Question", "source_entry_refs": ["e0"]}
    for payload in (
        {"candidates": [candidate]},
        {"complete": None, "candidates": [candidate]},
        {"complete": True, "candidates": [candidate], "unknown": 1},
    ):
        with pytest.raises(ValueError):
            parse_outcome_batch(json.dumps(payload), OutcomeRequestKind.QUESTION)
    with pytest.raises(OutcomeResponseCapacityError):
        parse_outcome_batch(
            json.dumps({"complete": False, "candidates": [candidate]}),
            OutcomeRequestKind.QUESTION,
        )
    with pytest.raises(OutcomeResponseCapacityError):
        parse_outcome_batch(
            json.dumps({"complete": True, "candidates": [candidate] * 3}),
            OutcomeRequestKind.QUESTION,
        )
    for kind, over_capacity_row in (
        (
            OutcomeRequestKind.QUESTION,
            {"text": "x" * (MAX_MODEL_OUTCOME_TEXT_CHARS + 1), "source_entry_refs": ["e0"]},
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "text": "Question",
                "source_entry_refs": [f"e{index}" for index in range(MAX_MODEL_ENTRY_REFS + 1)],
            },
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "text": "Action",
                "source_entry_refs": ["e0"],
                "owner_ref": None,
                "deadline": "d" * 33,
            },
        ),
    ):
        with pytest.raises(OutcomeResponseCapacityError):
            parse_outcome_batch(
                json.dumps({"complete": True, "candidates": [over_capacity_row]}),
                kind,
            )


@pytest.mark.parametrize(
    ("kind", "payload", "expected_subtype"),
    (
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": False,
                "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
            },
            OutcomeCapacitySubtype.COMPLETE_FALSE,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {"text": "Synthetic", "source_entry_refs": ["e0"]},
                    {"text": "Synthetic", "source_entry_refs": ["e0"]},
                    {"text": "Synthetic", "source_entry_refs": ["e0"]},
                ],
            },
            OutcomeCapacitySubtype.ROW_SATURATION,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "x" * (MAX_MODEL_OUTCOME_TEXT_CHARS + 1),
                        "source_entry_refs": ["e0"],
                    }
                ],
            },
            OutcomeCapacitySubtype.TEXT_CHARS,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Synthetic",
                        "source_entry_refs": [
                            f"e{index}" for index in range(MAX_MODEL_ENTRY_REFS + 1)
                        ],
                    }
                ],
            },
            OutcomeCapacitySubtype.ENTRY_REFS,
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "Synthetic",
                        "source_entry_refs": ["e0"],
                        "source_event_refs": ["v0", "v1", "v2", "v3", "v4"],
                    }
                ],
            },
            OutcomeCapacitySubtype.EVENT_REFS,
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Synthetic",
                        "source_entry_refs": ["e0"],
                        "owner_ref": "s123456789abc",
                        "deadline": None,
                    }
                ],
            },
            OutcomeCapacitySubtype.OWNER_REF,
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Synthetic",
                        "source_entry_refs": ["e0"],
                        "owner_ref": None,
                        "deadline": "d" * 33,
                    }
                ],
            },
            OutcomeCapacitySubtype.DEADLINE,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [{"text": "Synthetic", "source_entry_refs": ["ezz"]}],
            },
            None,
        ),
    ),
)
def test_fix24_parser_capacity_subtype_probe(
    kind: OutcomeRequestKind,
    payload: dict[str, object],
    expected_subtype: OutcomeCapacitySubtype | None,
) -> None:
    if expected_subtype is None:
        assert len(parse_outcome_batch(json.dumps(payload), kind)) == 1
        return
    with pytest.raises(OutcomeResponseCapacityError) as caught:
        parse_outcome_batch(json.dumps(payload), kind)
    assert caught.value.subtype is expected_subtype


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "payload", "expected_category", "expected_subtype", "expected_bucket"),
    (
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": False,
                "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
            },
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeCapacitySubtype.COMPLETE_FALSE,
            OutcomeSizeBucket.LTE_1K,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {"text": "Synthetic", "source_entry_refs": ["e0"]},
                    {"text": "Synthetic", "source_entry_refs": ["e0"]},
                    {"text": "Synthetic", "source_entry_refs": ["e0"]},
                ],
            },
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeCapacitySubtype.ROW_SATURATION,
            OutcomeSizeBucket.LTE_1K,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "x" * (MAX_MODEL_OUTCOME_TEXT_CHARS + 1),
                        "source_entry_refs": ["e0"],
                    }
                ],
            },
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeCapacitySubtype.TEXT_CHARS,
            OutcomeSizeBucket.LTE_1K,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Synthetic",
                        "source_entry_refs": ["e0"] * (MAX_MODEL_ENTRY_REFS + 1),
                    }
                ],
            },
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeCapacitySubtype.ENTRY_REFS,
            OutcomeSizeBucket.LTE_1K,
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "Synthetic",
                        "source_entry_refs": ["e0"],
                        "source_event_refs": ["v0", "v1", "v2", "v3", "v4"],
                    }
                ],
            },
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeCapacitySubtype.EVENT_REFS,
            OutcomeSizeBucket.LTE_1K,
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Synthetic",
                        "source_entry_refs": ["e0"],
                        "owner_ref": "s123456789abc",
                        "deadline": None,
                    }
                ],
            },
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeCapacitySubtype.OWNER_REF,
            OutcomeSizeBucket.LTE_1K,
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Synthetic",
                        "source_entry_refs": ["e0"],
                        "owner_ref": None,
                        "deadline": "d" * 33,
                    }
                ],
            },
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeCapacitySubtype.DEADLINE,
            OutcomeSizeBucket.LTE_1K,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [{"text": "Synthetic", "source_entry_refs": ["ezz"]}],
            },
            OutcomeFailureCategory.SCHEMA,
            None,
            OutcomeSizeBucket.LTE_1K,
        ),
    ),
)
async def test_fix24_provider_capacity_subtype_taxonomy_probe(
    kind: OutcomeRequestKind,
    payload: dict[str, object],
    expected_category: OutcomeFailureCategory,
    expected_subtype: OutcomeCapacitySubtype | None,
    expected_bucket: OutcomeSizeBucket,
) -> None:
    entry = _entry(1, "Synthetic source")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    responses = (payload, payload, payload)
    server, requests = await _server(iter(responses))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=kind,
            request_bytes=serialize_outcome_request(partition, kind),
        )
    finally:
        await server.close()

    expected_calls = 3
    assert len(requests) == expected_calls
    assert (
        provider.actual_chat_requests == controller.journal.actual_inference_calls == expected_calls
    )
    assert result.candidates == ()
    assert len(result.provider_failures) == expected_calls
    failure = result.provider_failures[0]
    assert failure.category is expected_category
    assert failure.capacity_subtype is expected_subtype
    assert failure.attempt is OutcomeFailureAttempt.INITIAL
    assert failure.size_bucket is expected_bucket


def test_selected_cjk_semantic_wire_fixtures_record_exact_utf8_sizes() -> None:
    text = "会" * MAX_MODEL_OUTCOME_TEXT_CHARS
    refs = ["eaaaaaaaaaaa", "eaaaaaaaaaab", "eaaaaaaaaaac"]
    candidates = {
        OutcomeRequestKind.DECISION_AGREEMENT: {
            "type": "agreement",
            "text": text,
            "source_entry_refs": refs,
            "source_event_refs": ["vaaaaaaaaaaa", "vaaaaaaaaaab", "vaaaaaaaaaac"],
        },
        OutcomeRequestKind.ACTION: {
            "text": text,
            "source_entry_refs": refs,
            "owner_ref": "s1234567890a",
            "deadline": "会" * MAX_MODEL_DEADLINE_CHARS,
        },
        OutcomeRequestKind.QUESTION: {"text": text, "source_entry_refs": refs},
        OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT: {
            "type": "disagreement",
            "text": text,
            "source_entry_refs": refs,
        },
    }
    observed_sizes = {
        OutcomeRequestKind.DECISION_AGREEMENT: (602, 1082),
        OutcomeRequestKind.ACTION: (576, 1184),
        OutcomeRequestKind.QUESTION: (430, 910),
        OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT: (474, 954),
    }
    for kind, candidate in candidates.items():
        payload = json.dumps(
            {"complete": True, "candidates": [candidate] * (MAX_MODEL_OUTCOME_ROWS - 1)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        assert (len(payload), len(payload.encode("utf-8"))) == observed_sizes[kind]


def test_json_escaped_action_fixture_can_exceed_selected_cjk_utf8_sizes() -> None:
    escaped = "\u0000" * MAX_MODEL_OUTCOME_TEXT_CHARS
    candidate = {
        "text": escaped,
        "source_entry_refs": ["eaaaaaaaaaaa", "eaaaaaaaaaab", "eaaaaaaaaaac"],
        "owner_ref": "s1234567890a",
        "deadline": "\u0000" * MAX_MODEL_DEADLINE_CHARS,
    }
    payload = json.dumps(
        {"complete": True, "candidates": [candidate] * (MAX_MODEL_OUTCOME_ROWS - 1)},
        ensure_ascii=False,
        separators=(",", ":"),
    )

    # The selected CJK byte observations above are not a universal output maximum.
    assert len(payload.encode("utf-8")) > 1_536


@pytest.mark.asyncio
@pytest.mark.parametrize("saturated", ("more", "row_cap"))
async def test_provider_discards_saturated_batch_after_two_repairs(saturated: str) -> None:
    entry = _entry(1, "question?")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    candidate = {"text": "Question", "source_entry_refs": ["e0"]}
    response = {
        "complete": saturated != "more",
        "candidates": [candidate] * (MAX_MODEL_OUTCOME_ROWS if saturated == "row_cap" else 1),
    }
    server, requests = await _server(iter((response, response, response)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()
    assert len(requests) == 3
    assert provider.actual_chat_requests == 3
    assert controller.journal.actual_inference_calls == 3
    assert result.candidates == ()
    assert [(item.category, item.attempt) for item in result.provider_failures] == [
        (OutcomeFailureCategory.OUTPUT_CAPACITY, OutcomeFailureAttempt.INITIAL),
        (OutcomeFailureCategory.OUTPUT_CAPACITY, OutcomeFailureAttempt.REPAIR),
        (OutcomeFailureCategory.OUTPUT_CAPACITY, OutcomeFailureAttempt.REPAIR),
    ]
    assert all(
        failure.capacity_subtype
        is (
            OutcomeCapacitySubtype.COMPLETE_FALSE
            if saturated == "more"
            else OutcomeCapacitySubtype.ROW_SATURATION
        )
        for failure in result.provider_failures
    )


@pytest.mark.asyncio
async def test_fix39_complete_false_gets_two_content_safe_repairs_then_success() -> None:
    entry = _entry(1, "Synthetic question?")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    incomplete = {
        "complete": False,
        "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    valid = {
        "complete": True,
        "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    server, requests = await _server(iter((incomplete, incomplete, valid)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        controller, job_hash = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert len(result.candidates) == 1
    assert result.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS
    assert [
        (failure.category, failure.attempt, failure.capacity_subtype)
        for failure in result.provider_failures
    ] == [
        (
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeFailureAttempt.INITIAL,
            OutcomeCapacitySubtype.COMPLETE_FALSE,
        ),
        (
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeFailureAttempt.REPAIR,
            OutcomeCapacitySubtype.COMPLETE_FALSE,
        ),
    ]
    task_hash = outcome_task_hash(job_hash, partition, OutcomeRequestKind.QUESTION)
    task_actions = tuple(
        action for action in controller.journal.actions if action.task_hash == task_hash
    )
    assert [(action.kind, action.action_hash) for action in task_actions] == [
        (LlmActionKind.OUTCOME_INITIAL, llm_action_hash(task_hash, LlmActionKind.OUTCOME_INITIAL)),
        (LlmActionKind.OUTCOME_REPAIR, llm_action_hash(task_hash, LlmActionKind.OUTCOME_REPAIR, 0)),
        (LlmActionKind.OUTCOME_REPAIR, llm_action_hash(task_hash, LlmActionKind.OUTCOME_REPAIR, 1)),
    ]
    for request in requests[1:]:
        assert request["messages"][:2] == requests[0]["messages"]
        assert len(request["messages"]) == 3
        assert "output_capacity:complete_false" in request["messages"][-1]["content"]
        assert "Synthetic" not in request["messages"][-1]["content"]
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["max_tokens"] == 768
    assert all(request["thinking"] == {"type": "enabled"} for request in requests[1:])
    assert all(
        request["max_tokens"] == MAX_DEEPSEEK_REPAIR_OUTPUT_TOKENS for request in requests[1:]
    )


@pytest.mark.asyncio
async def test_fix40_deepseek_repairs_use_thinking_without_replaying_reasoning() -> None:
    entry = _entry(1, "Synthetic question?")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    incomplete = {
        "complete": False,
        "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    valid = {
        "complete": True,
        "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    sentinels = ("PRIVATE_REASONING_ONE", "PRIVATE_REASONING_TWO")
    server, requests = await _deepseek_outcome_server(
        iter(
            (
                (incomplete, None, "stop"),
                (incomplete, sentinels[0], "stop"),
                (valid, sentinels[1], "stop"),
            )
        )
    )
    try:
        provider = _deepseek_outcome_provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["temperature"] == 0
    assert requests[0]["max_tokens"] == 768
    assert "reasoning_effort" not in requests[0]
    for request in requests[1:]:
        assert request["thinking"] == {"type": "enabled"}
        assert request["reasoning_effort"] == "high"
        assert "temperature" not in request
        assert request["response_format"] == {"type": "json_object"}
        assert request["max_tokens"] == MAX_DEEPSEEK_REPAIR_OUTPUT_TOKENS
    request_text = json.dumps(requests, ensure_ascii=False)
    result_text = result.model_dump_json()
    journal_text = controller.journal.model_dump_json()
    assert all(sentinel not in request_text for sentinel in sentinels)
    assert all(sentinel not in result_text for sentinel in sentinels)
    assert all(sentinel not in journal_text for sentinel in sentinels)
    assert result.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS


@pytest.mark.asyncio
async def test_fix43_deepseek_output_limit_gets_second_bounded_repair() -> None:
    entry = _entry(1, "Synthetic PRD evidence")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    incomplete = {
        "complete": False,
        "candidates": [{"type": "risk", "text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    valid = {
        "complete": True,
        "candidates": [{"type": "risk", "text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    private_reasoning = "PRIVATE_TRUNCATED_REASONING_MUST_NOT_ESCAPE"
    server, requests = await _deepseek_outcome_server(
        iter(
            (
                (incomplete, None, "stop"),
                ("", private_reasoning, "length"),
                (valid, None, "stop"),
            )
        )
    )
    try:
        provider = _deepseek_outcome_provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            request_bytes=serialize_outcome_request(
                partition, OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT
            ),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert requests[1]["max_tokens"] == requests[2]["max_tokens"] == 32_768
    assert result.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.OUTPUT_CAPACITY, OutcomeFailureAttempt.INITIAL),
        (OutcomeFailureCategory.OUTPUT_LIMIT, OutcomeFailureAttempt.REPAIR),
    ]
    assert private_reasoning not in json.dumps(requests, ensure_ascii=False)
    assert private_reasoning not in result.model_dump_json()
    assert private_reasoning not in controller.journal.model_dump_json()


@pytest.mark.asyncio
async def test_fix39_duplicate_then_complete_false_gets_second_repair() -> None:
    entry = _entry(1, "Synthetic evidence")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    first = _candidate("private-a", "open_question", "e1")
    second = {**first, "id": "private-b", "text": "different supported wording"}
    incomplete = {
        "complete": False,
        "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    valid = {
        "complete": True,
        "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    server, requests = await _server(iter(({"candidates": [first, second]}, incomplete, valid)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert len(result.candidates) == 1
    assert [
        (failure.category, failure.attempt, failure.format_subtype, failure.capacity_subtype)
        for failure in result.provider_failures
    ] == [
        (
            OutcomeFailureCategory.SCHEMA,
            OutcomeFailureAttempt.INITIAL,
            OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
            None,
        ),
        (
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeFailureAttempt.REPAIR,
            None,
            OutcomeCapacitySubtype.COMPLETE_FALSE,
        ),
    ]
    assert "schema:duplicate_binding_wire_divergent" in requests[1]["messages"][-1]["content"]
    assert "output_capacity:complete_false" in requests[2]["messages"][-1]["content"]
    assert all("private-" not in request["messages"][-1]["content"] for request in requests[1:])


@pytest.mark.asyncio
async def test_fix39_two_failed_repairs_are_terminal_without_a_fourth_call() -> None:
    entry = _entry(1, "Synthetic question?")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    incomplete = {
        "complete": False,
        "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    server, requests = await _server(iter((incomplete, incomplete, incomplete)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert result.candidates == ()
    assert result.task_disposition is OutcomeTaskDisposition.TERMINAL_FAILURE
    assert [failure.attempt for failure in result.provider_failures] == [
        OutcomeFailureAttempt.INITIAL,
        OutcomeFailureAttempt.REPAIR,
        OutcomeFailureAttempt.REPAIR,
    ]


@pytest.mark.asyncio
async def test_fix24_second_repair_capacity_subtype_is_terminal_and_retains_failures() -> None:
    entry = _entry(1, "Synthetic question?")
    topic = TopicSegment(
        "topic-001", "Synthetic topic", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    repair = {
        "complete": False,
        "candidates": [{"text": "Synthetic", "source_entry_refs": ["e0"]}],
    }
    server, requests = await _server(iter(("not-json", repair, repair)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert provider.actual_chat_requests == controller.journal.actual_inference_calls == 3
    assert result.candidates == ()
    assert [
        (failure.category, failure.attempt, failure.capacity_subtype)
        for failure in result.provider_failures
    ] == [
        (OutcomeFailureCategory.JSON, OutcomeFailureAttempt.INITIAL, None),
        (
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeFailureAttempt.REPAIR,
            OutcomeCapacitySubtype.COMPLETE_FALSE,
        ),
        (
            OutcomeFailureCategory.OUTPUT_CAPACITY,
            OutcomeFailureAttempt.REPAIR,
            OutcomeCapacitySubtype.COMPLETE_FALSE,
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "response"),
    (
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "x" * (MAX_MODEL_OUTCOME_TEXT_CHARS + 1),
                        "source_entry_refs": ["e0"],
                    }
                ],
            },
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Question",
                        "source_entry_refs": ["e0"] * (MAX_MODEL_ENTRY_REFS + 1),
                    }
                ],
            },
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Action",
                        "source_entry_refs": ["e0"],
                        "owner_ref": None,
                        "deadline": "d" * 33,
                    }
                ],
            },
        ),
    ),
)
async def test_provider_treats_over_capacity_fields_as_terminal_after_two_repairs(
    kind: OutcomeRequestKind,
    response: dict[str, object],
) -> None:
    entry = _entry(1, "question?")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    server, requests = await _server(iter((response, response, response)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=kind,
            request_bytes=serialize_outcome_request(partition, kind),
        )
    finally:
        await server.close()
    assert len(requests) == 3
    assert provider.actual_chat_requests == 3
    assert controller.journal.actual_inference_calls == 3
    assert result.candidates == ()
    assert all(
        failure.category is OutcomeFailureCategory.OUTPUT_CAPACITY
        for failure in result.provider_failures
    )
    assert [failure.attempt for failure in result.provider_failures] == [
        OutcomeFailureAttempt.INITIAL,
        OutcomeFailureAttempt.REPAIR,
        OutcomeFailureAttempt.REPAIR,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("subject_length", (MAX_MODEL_OUTCOME_TEXT_CHARS + 1, 500))
async def test_authoritative_event_subject_capacity_fails_before_chat(
    subject_length: int,
) -> None:
    entry = _entry(1, "vote")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    event = OutcomeEventFact(
        sequence_number=1,
        topic_id=topic.id,
        subject="x" * subject_length,
        result="passed",
        source_entry_ids=(entry.entry_id,),
    )
    server, requests = await _server(iter(()))
    try:
        provider = _provider(server)
        partition = next(
            item for item in plan_outcome_partitions(topic, (entry,), (event,)) if item.core_events
        )
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.DECISION_AGREEMENT,
            request_bytes=serialize_outcome_request(
                partition, OutcomeRequestKind.DECISION_AGREEMENT
            ),
        )
    finally:
        await server.close()
    assert requests == []
    assert provider.actual_chat_requests == 0
    assert controller.journal.actual_inference_calls == 0
    assert result.provider_failures == (
        OutcomeProviderFailure(
            category=OutcomeFailureCategory.OUTPUT_CAPACITY,
            request_kind=OutcomeFailureRequestKind.DECISION_AGREEMENT,
            attempt=OutcomeFailureAttempt.PREFLIGHT,
            count=1,
            size_bucket=OutcomeSizeBucket.NOT_OBSERVED,
            capacity_subtype=OutcomeCapacitySubtype.EVENT_SUBJECT,
        ),
    )


@pytest.mark.asyncio
async def test_authoritative_event_reference_capacity_fails_before_chat() -> None:
    entries = tuple(_entry(index, f"source {index}") for index in range(1, 6))
    topic = TopicSegment(
        "topic-001",
        "Topic 1",
        tuple(entry.entry_id for entry in entries),
        1_000,
        5_800,
        "deterministic_fallback",
    )
    event = OutcomeEventFact(
        sequence_number=1,
        topic_id=topic.id,
        subject="accepted",
        result="passed",
        source_entry_ids=tuple(entry.entry_id for entry in entries),
    )
    server, requests = await _server(iter(()))
    try:
        provider = _provider(server)
        partition = next(
            item for item in plan_outcome_partitions(topic, entries, (event,)) if item.core_events
        )
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.DECISION_AGREEMENT,
            request_bytes=serialize_outcome_request(
                partition, OutcomeRequestKind.DECISION_AGREEMENT
            ),
        )
    finally:
        await server.close()
    assert requests == []
    assert controller.journal.actual_inference_calls == 0
    assert result.provider_failures[0].category is OutcomeFailureCategory.OUTPUT_CAPACITY
    assert result.provider_failures[0].attempt is OutcomeFailureAttempt.PREFLIGHT
    assert (
        result.provider_failures[0].capacity_subtype
        is OutcomeCapacitySubtype.AUTHORITATIVE_ENTRY_REFS
    )


@pytest.mark.asyncio
async def test_authoritative_event_row_capacity_fails_before_chat() -> None:
    entry = _entry(1, "votes")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    events = tuple(
        OutcomeEventFact(
            sequence_number=index,
            topic_id=topic.id,
            subject=f"accepted-{index}",
            result="passed",
            source_entry_ids=(entry.entry_id,),
        )
        for index in range(1, MAX_MODEL_OUTCOME_ROWS + 1)
    )
    server, requests = await _server(iter(()))
    try:
        provider = _provider(server)
        partition = next(
            item
            for item in plan_outcome_partitions(topic, (entry,), events[:2])
            if item.core_events
        ).model_copy(update={"core_events": events})
        controller, _ = _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.DECISION_AGREEMENT,
            request_bytes=serialize_outcome_request(
                partition, OutcomeRequestKind.DECISION_AGREEMENT
            ),
        )
    finally:
        await server.close()
    assert requests == []
    assert provider.actual_chat_requests == 0
    assert controller.journal.actual_inference_calls == 0
    assert result.provider_failures[0].category is OutcomeFailureCategory.OUTPUT_CAPACITY
    assert result.provider_failures[0].attempt is OutcomeFailureAttempt.PREFLIGHT
    assert result.provider_failures[0].capacity_subtype is OutcomeCapacitySubtype.EVENT_ROWS


@pytest.mark.parametrize(
    ("kind", "candidate", "expected_fields"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "type": "decision",
                "text": "The proposal was accepted.",
                "source_entry_refs": ["e0"],
                "source_event_refs": ["v0"],
            },
            {"type", "text", "source_entry_refs", "source_event_refs"},
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "text": "Prepare the release notes.",
                "source_entry_refs": ["e0"],
                "owner_ref": None,
                "deadline": None,
            },
            {"text", "source_entry_refs", "owner_ref", "deadline"},
        ),
        (
            OutcomeRequestKind.QUESTION,
            {"text": "When is the release?", "source_entry_refs": ["e0"]},
            {"text", "source_entry_refs"},
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            {
                "type": "risk",
                "text": "Capacity remains constrained.",
                "source_entry_refs": ["e0"],
            },
            {"type", "text", "source_entry_refs"},
        ),
    ),
)
def test_strict_per_kind_minimal_output_dto_is_parseable_without_irrelevant_fields(
    kind: OutcomeRequestKind,
    candidate: dict[str, object],
    expected_fields: set[str],
) -> None:
    """Each call kind exposes only semantic fields it can use as evidence."""

    parsed = parse_outcome_batch(
        json.dumps({"complete": True, "candidates": [candidate]}),
        kind,
    )
    assert len(parsed) == 1
    assert parsed[0].owner_name is None
    assert parsed[0].confidence == 0.0
    candidate_schema = outcome_json_schema(kind)["properties"]["candidates"]["items"]
    assert candidate_schema["additionalProperties"] is False
    assert set(candidate_schema["required"]) == expected_fields


@pytest.mark.parametrize(
    ("kind", "historical_category_clause", "current_shape_clause"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            "本次只提取: 决议或明确一致意见",
            "candidate={type:decision或agreement,text,source_entry_refs,source_event_refs}",
        ),
        (
            OutcomeRequestKind.ACTION,
            "本次只提取: 行动项",
            "candidate={text,source_entry_refs,owner_ref或null,deadline或null}",
        ),
        (
            OutcomeRequestKind.QUESTION,
            "本次只提取: 未决问题",
            "candidate={text,source_entry_refs}",
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            "本次只提取: 提议、风险或分歧",
            "candidate={type:proposal或risk或disagreement,text,source_entry_refs}",
        ),
    ),
)
def test_fix23_prompt_restores_natural_category_obligation_without_pseudo_examples(
    kind: OutcomeRequestKind,
    historical_category_clause: str,
    current_shape_clause: str,
) -> None:
    """Snapshot only synthetic protocol wording; it does not invoke a provider."""

    prompt = outcome_system_prompt(kind)

    assert "每次只提取指定类别" in prompt
    assert historical_category_clause in prompt
    assert current_shape_clause not in prompt
    assert "必须返回core证据明确支持的每一个本次类别候选" in prompt
    assert "不得编造、补全或推断证据中不存在的事实" in prompt
    assert "只有没有任何core证据明确支持的本次类别候选时才返回空candidates" in prompt
    assert "若仍有任何有证据支持但未返回的候选,complete必须为false" in prompt
    assert '无结果时只能返回{"complete":true,"candidates":[]}' not in prompt
    assert "candidate={" not in prompt
    assert "示例" not in prompt
    for forbidden_authority_field in (
        "owner_name",
        "confidence",
        "requires_confirmation",
        "verification_status",
    ):
        assert forbidden_authority_field not in prompt


@pytest.mark.parametrize(
    ("kind", "expected_chars", "expected_bytes", "expected_sha256"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            1958,
            3070,
            "e2d245c2c0277ee850a0de1b45b4ed7ade0a64aa4e2e48879c4d6c5f9bc918bb",
        ),
        (
            OutcomeRequestKind.ACTION,
            1737,
            2799,
            "a2bb35e132676faf0e41d7f19a2c9d92e9a1609932b321ed4abf13a4c1fe3e6b",
        ),
        (
            OutcomeRequestKind.QUESTION,
            1512,
            2576,
            "abca2507feece03688669916e59e49499d4809cd410f3f0c6a9e5775aa20f84f",
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            1595,
            2667,
            "4c3262620130344822ed2d8d9555dc22dfe08ee08ad5dbc8a33b20ae804b288f",
        ),
    ),
)
def test_fix36_prompt_exactly_binds_compact_json_protocol_bytes(
    kind: OutcomeRequestKind,
    expected_chars: int,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    prompt = outcome_system_prompt(kind)

    assert len(prompt) == expected_chars
    assert len(prompt.encode("utf-8")) == expected_bytes
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == expected_sha256
    assert "source_entry_refs只能引用c中的e ref" in prompt
    assert "source_event_refs只能引用x中的ref" in prompt
    assert "不要输出解释或思考过程" in prompt
    assert "动态别名约束:" not in prompt
    assert "输出必须是一个JSON object" in prompt
    assert "通常只需1项" in prompt
    assert "硬上限120字符" in prompt
    assert "按JSON字符串转义" in prompt
    assert "candidate={" not in prompt
    assert "示例" not in prompt


def test_fix36_dynamic_response_schema_hashes_bind_bounded_headroom() -> None:
    entry = _entry(1, "Synthetic question?")
    topic = TopicSegment(
        "topic-001",
        "Topic 1",
        (entry.entry_id,),
        1_000,
        1_800,
        "deterministic_fallback",
    )
    partition = plan_outcome_partitions(
        topic,
        (entry,),
        (),
        allowed_participant_ids=(7,),
    )[0]
    expected_hashes = {
        OutcomeRequestKind.DECISION_AGREEMENT: (
            "cbd0bbc275245f1f6461b265e2bfb16c3fc22b319355114aafa82071c878cd98"
        ),
        OutcomeRequestKind.ACTION: (
            "70cb3c01021dd48ab3a75fc1e45f567c21d59890e24cc545a3e94b94e7e6f082"
        ),
        OutcomeRequestKind.QUESTION: (
            "680223efb1a9243500283cb53630cb3382d8c9c15005454084727072d61f6dc5"
        ),
        OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT: (
            "96a7d85c4608b8b815b2a779ff4e56c0e0568a1a4548a5bb86f17417e4fae2a0"
        ),
    }

    assert {
        kind: hashlib.sha256(
            json.dumps(
                outcome_response_schema(partition, kind),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for kind in OutcomeRequestKind
    } == expected_hashes


@pytest.mark.parametrize(
    ("content", "subtype"),
    (
        ("", OutcomeFormatSubtype.JSON_EMPTY),
        ('  {"ok":true} trailing', OutcomeFormatSubtype.JSON_EXTRA_DATA),
        ('\u00a0{"ok":true} trailing', OutcomeFormatSubtype.JSON_SYNTAX_OTHER),
        ('\u2003{"ok":true} trailing', OutcomeFormatSubtype.JSON_SYNTAX_OTHER),
        ('{"ok":true}\u00a0', OutcomeFormatSubtype.JSON_EXTRA_DATA),
        ('{"ok":true}\u2003', OutcomeFormatSubtype.JSON_EXTRA_DATA),
        ('{"value":"unterminated', OutcomeFormatSubtype.JSON_UNTERMINATED_STRING),
        ("{", OutcomeFormatSubtype.JSON_SYNTAX_OTHER),
    ),
)
def test_fix30_json_classifier_does_not_depend_on_exception_message(
    content: str,
    subtype: OutcomeFormatSubtype,
) -> None:
    error = json.JSONDecodeError("PRIVATE_LOCALIZED_MESSAGE_SENTINEL", content, 0)

    assert classify_json_decode_failure(content, error) is subtype


def test_fix30_legal_json_whitespace_suffix_is_valid_without_classification() -> None:
    assert json.loads('{"ok":true} \t\r\n') == {"ok": True}


def test_fix36_all_four_closed_base_schema_snapshots_bind_text_capacity() -> None:
    assert OUTCOME_EXTRACTION_CONFIG_VERSION == 32
    expected_hashes = {
        OutcomeRequestKind.DECISION_AGREEMENT: (
            "fe815d2158cc74f277e497816d3a2a4575dcdd87a67fba8cc11165a6aabd1353"
        ),
        OutcomeRequestKind.ACTION: (
            "4fbcf5ec7f2542620ada6ecb3a1dd2366c386343cbe9f86f7833099df602014b"
        ),
        OutcomeRequestKind.QUESTION: (
            "709d84ac1b2303b7cad5303e803469b4dde586352b7c19c1e856ba6a536a6d65"
        ),
        OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT: (
            "0321f23103fcb89efa3168958e7ba8536f883946db6f39e7949e5987f4237b1f"
        ),
    }

    observed_hashes = {
        kind: hashlib.sha256(
            json.dumps(
                outcome_json_schema(kind),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for kind in OutcomeRequestKind
    }

    assert observed_hashes == expected_hashes


@pytest.mark.parametrize(
    ("kind", "candidate"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "type": "decision",
                "text": "Synthetic decision",
                "source_entry_refs": ["e0"],
                "source_event_refs": [],
            },
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "text": "Synthetic action",
                "source_entry_refs": ["e0"],
                "owner_ref": None,
                "deadline": None,
            },
        ),
        (
            OutcomeRequestKind.QUESTION,
            {"text": "Synthetic question", "source_entry_refs": ["e0"]},
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            {
                "type": "proposal",
                "text": "Synthetic proposal",
                "source_entry_refs": ["e0"],
            },
        ),
    ),
)
def test_fix23_current_protocol_accepts_nonempty_semantic_example_and_valid_empty_batch(
    kind: OutcomeRequestKind,
    candidate: dict[str, object],
) -> None:
    """Both nonempty and empty roots are valid under the current closed protocol."""

    nonempty = parse_outcome_batch(
        json.dumps({"complete": True, "candidates": [candidate]}),
        kind,
    )
    empty = parse_outcome_batch(
        json.dumps({"complete": True, "candidates": []}),
        kind,
    )

    assert len(nonempty) == 1
    assert empty == ()
    with pytest.raises(ValueError):
        parse_outcome_batch(json.dumps({"candidates": [candidate]}), kind)


@pytest.mark.parametrize(
    ("kind", "payload"),
    (
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [{"text": "Question", "source_entry_refs": ["e0"], "extra": None}],
            },
        ),
        (
            OutcomeRequestKind.QUESTION,
            {"complete": True, "candidates": [{"text": "Question", "source_entry_refs": []}]},
        ),
        (
            OutcomeRequestKind.QUESTION,
            {"complete": True, "candidates": [{"text": "Question", "source_entry_refs": ["E0"]}]},
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [{"text": "Question", "source_entry_refs": ["e0", "e0"]}],
            },
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [{"text": "Question", "source_entry_refs": ["e123456789abc"]}],
            },
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Action",
                        "source_entry_refs": ["e0"],
                        "owner_ref": "S0",
                        "deadline": None,
                    }
                ],
            },
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Action",
                        "source_entry_refs": ["e0"],
                        "owner_ref": None,
                        "deadline": "",
                    }
                ],
            },
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "proposal",
                        "text": "Text",
                        "source_entry_refs": ["e0"],
                        "source_event_refs": [],
                    }
                ],
            },
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            {
                "complete": True,
                "candidates": [{"type": "decision", "text": "Text", "source_entry_refs": ["e0"]}],
            },
        ),
        (OutcomeRequestKind.QUESTION, {"complete": 1, "candidates": []}),
        (OutcomeRequestKind.QUESTION, {"complete": True, "candidates": [], "extra": None}),
    ),
)
def test_semantic_outcome_dto_rejects_every_noncanonical_shape(
    kind: OutcomeRequestKind,
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        parse_outcome_batch(json.dumps(payload), kind)


@pytest.mark.parametrize("wrong_type", ([], {}, True, 1, None))
@pytest.mark.parametrize(
    ("kind", "candidate"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {"text": "Text", "source_entry_refs": ["e0"], "source_event_refs": []},
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            {"text": "Text", "source_entry_refs": ["e0"]},
        ),
    ),
)
def test_semantic_type_wrong_json_type_is_closed_schema_failure(
    kind: OutcomeRequestKind,
    candidate: dict[str, object],
    wrong_type: object,
) -> None:
    with pytest.raises(ValueError):
        parse_outcome_batch(
            json.dumps(
                {
                    "complete": True,
                    "candidates": [{"type": wrong_type, **candidate}],
                }
            ),
            kind,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_type", ([], {}, True, 1, None))
@pytest.mark.parametrize(
    ("kind", "candidate"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {"text": "Text", "source_entry_refs": ["e0"], "source_event_refs": []},
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            {"text": "Text", "source_entry_refs": ["e0"]},
        ),
    ),
)
async def test_real_provider_repairs_wrong_semantic_type_once_and_retains_initial_failure(
    kind: OutcomeRequestKind,
    candidate: dict[str, object],
    wrong_type: object,
) -> None:
    entry = _entry(1, "we discussed an outcome")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    valid_candidate = (
        {
            "type": "decision",
            "text": "Text",
            "source_entry_refs": ["e0"],
            "source_event_refs": [],
        }
        if kind is OutcomeRequestKind.DECISION_AGREEMENT
        else {"type": "proposal", "text": "Text", "source_entry_refs": ["e0"]}
    )
    server, requests = await _server(
        iter(
            (
                {
                    "complete": True,
                    "candidates": [{"type": wrong_type, **candidate}],
                },
                {"complete": True, "candidates": [valid_candidate]},
            )
        )
    )
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=kind,
            request_bytes=serialize_outcome_request(partition, kind),
        )
    finally:
        await server.close()

    assert len(requests) == 2
    assert len(result.candidates) == 1
    assert result.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.INITIAL)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "invalid_candidate"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "type": [],
                "text": "Text",
                "source_entry_refs": ["e0"],
                "source_event_refs": [],
            },
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            {"type": {}, "text": "Text", "source_entry_refs": ["e0"]},
        ),
    ),
)
async def test_real_provider_wrong_semantic_type_second_repair_failure_is_terminal_schema(
    kind: OutcomeRequestKind,
    invalid_candidate: dict[str, object],
) -> None:
    entry = _entry(1, "we discussed an outcome")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    server, requests = await _server(
        iter(
            (
                {"complete": True, "candidates": [invalid_candidate]},
                {"complete": True, "candidates": [invalid_candidate]},
                {"complete": True, "candidates": [invalid_candidate]},
            )
        )
    )
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=kind,
            request_bytes=serialize_outcome_request(partition, kind),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert result.candidates == ()
    assert result.task_disposition is OutcomeTaskDisposition.TERMINAL_FAILURE
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.INITIAL),
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.REPAIR),
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.REPAIR),
    ]
    assert [failure.format_subtype for failure in result.provider_failures] == [
        OutcomeFormatSubtype.SCHEMA_TYPE,
        OutcomeFormatSubtype.SCHEMA_TYPE,
        OutcomeFormatSubtype.SCHEMA_TYPE,
    ]


@pytest.mark.parametrize("legacy_raw_id", (None, "", "x" * 1_000, 42))
def test_legacy_object_candidate_id_cannot_bypass_semantic_schema(
    legacy_raw_id: object,
) -> None:
    candidate = _candidate("unused", "action_item", "e1")
    if legacy_raw_id is None:
        candidate.pop("id")
    else:
        candidate["id"] = legacy_raw_id

    with pytest.raises(ValueError):
        parse_outcome_batch(
            json.dumps({"complete": True, "candidates": [candidate]}),
            OutcomeRequestKind.ACTION,
        )


@pytest.mark.asyncio
async def test_provider_issues_four_bounded_closed_requests_per_topic() -> None:
    responses = iter(
        (
            {"candidates": [_candidate("decision-1", "decision", "e1")]},
            {
                "candidates": [
                    _candidate(
                        "action-1",
                        "action_item",
                        "e2",
                        owner_user_id=7,
                        owner_name="participant",
                    )
                ]
            },
            {"candidates": [_candidate("question-1", "open_question", "e3")]},
            {"candidates": [_candidate("risk-1", "risk", "e4")]},
        )
    )
    server, requests = await _server(responses)
    entries = (
        _entry(1, "we decided to release"),
        _entry(2, "I will prepare release"),
        _entry(3, "when will we release?"),
        _entry(4, "capacity is a risk"),
    )
    topic = TopicSegment(
        "topic-001", "Topic 1", tuple(e.entry_id for e in entries), 1_000, 4_800, "model_confirmed"
    )
    vote = OutcomeEventFact(
        sequence_number=9,
        topic_id=topic.id,
        subject="release",
        result="accepted",
        source_entry_ids=("e1",),
    )
    try:
        provider = _provider(server)
        result = await _extract_all(
            provider,
            topic=topic,
            entries=entries,
            events=(vote,),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert [candidate.type for candidate in result.candidates] == [
        OutcomeType.DECISION,
        OutcomeType.ACTION_ITEM,
        OutcomeType.OPEN_QUESTION,
        OutcomeType.RISK,
    ]
    action = next(
        candidate for candidate in result.candidates if candidate.type is OutcomeType.ACTION_ITEM
    )
    assert action.owner_user_id == 7
    assert action.owner_name == "participant"
    assert len(requests) == 4
    assert all(request["thinking"] == {"type": "disabled"} for request in requests)
    assert all(request["temperature"] == 0 for request in requests)
    assert all(request["max_tokens"] <= 768 for request in requests)
    assert all(request["response_format"] == {"type": "json_object"} for request in requests)
    contexts = [json.loads(request["messages"][1]["content"]) for request in requests]
    assert {context["k"] for context in contexts} == {
        "decision_agreement",
        "action",
        "question",
        "proposal_risk_disagreement",
    }
    assert all(context["t"] == topic.title for context in contexts)
    assert all(topic.id not in json.dumps(context, ensure_ascii=False) for context in contexts)
    assert all(len(context["c"]) == 4 and context["o"] == [] for context in contexts)
    assert all(
        context["v"] == 9 and context["x"] == [["v0", 0, "release", "accepted", ["e0"]]]
        for context in contexts
    )
    assert provider.actual_chat_requests == 4
    assert provider.actual_chat_requests == provider._budget().journal.actual_inference_calls
    decision_request = next(
        request
        for request in requests
        if json.loads(request["messages"][1]["content"])["k"] == "decision_agreement"
    )
    assert (
        "Event-supported text必须逐字复制sanitized event subject"
        in (decision_request["messages"][0]["content"])
    )


@pytest.mark.asyncio
async def test_provider_maps_event_alias_back_to_canonical_sequence() -> None:
    server, _ = await _server(
        iter(
            (
                {"candidates": [_candidate("ignored", "decision", "e1", source_event_ref="v0")]},
                {"candidates": []},
                {"candidates": []},
                {"candidates": []},
            )
        )
    )
    entry = _entry(1, "we decided")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    event = OutcomeEventFact(
        sequence_number=9,
        topic_id=topic.id,
        subject="we decided",
        result="accepted",
        source_entry_ids=(entry.entry_id,),
    )
    try:
        result = await _extract_all(
            _provider(server),
            topic=topic,
            entries=(entry,),
            events=(event,),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert result.candidates[0].source_entry_ids == (entry.entry_id,)
    assert result.candidates[0].source_event_ids == (event.sequence_number,)


@pytest.mark.asyncio
async def test_provider_attempts_json_repair_only_once() -> None:
    responses = iter(
        (
            "not-json",
            {"candidates": [_candidate("decision-1", "decision", "e1")]},
            {"candidates": []},
            {"candidates": []},
            {"candidates": []},
        )
    )
    server, requests = await _server(responses)
    entry = _entry(1, "we decided to release")
    try:
        provider = _provider(server)
        result = await _extract_all(
            provider,
            topic=TopicSegment(
                "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert len(requests) == 5
    repair_messages = requests[1]["messages"]
    assert len(repair_messages) == 3
    assert sum("上一次输出未通过" in message["content"] for message in repair_messages) == 1
    assert "json:json_syntax_other" in repair_messages[-1]["content"]
    assert all("not-json" not in message["content"] for message in repair_messages)
    assert result.failure_types == ()
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.JSON, OutcomeFailureAttempt.INITIAL)
    ]


@pytest.mark.asyncio
async def test_fix31_outcome_http_body_preserves_non_system_protocol_exactly() -> None:
    server, requests = await _server(iter(({"complete": True, "candidates": []},)))
    entry = _entry(1, "Synthetic protocol question?")
    topic = TopicSegment(
        "topic-001",
        "Topic 1",
        (entry.entry_id,),
        1_000,
        1_800,
        "deterministic_fallback",
    )
    partition = plan_outcome_partitions(topic, (entry,), (), allowed_participant_ids=(7,))[0]
    request_bytes = serialize_outcome_request(partition, OutcomeRequestKind.QUESTION)
    try:
        provider = _provider(server)
        _bind_provider(provider, (partition,))
        await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=request_bytes,
        )
    finally:
        await server.close()

    assert len(requests) == 1
    observed = requests[0]
    assert set(observed) == {
        "model",
        "messages",
        "stream",
        "thinking",
        "response_format",
        "temperature",
        "max_tokens",
    }
    assert observed["model"] == APPROVED_TOPIC_MODEL
    assert observed["messages"][1] == {
        "role": "user",
        "content": request_bytes.decode("utf-8"),
    }
    assert observed["response_format"] == {"type": "json_object"}
    assert observed["temperature"] == 0
    assert observed["max_tokens"] == 768
    assert observed["stream"] is False
    assert observed["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "expected_prefix_chars", "expected_prefix_bytes", "expected_prefix_sha256"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            1111,
            2205,
            "a32a908dd1955ee26ee00881f79bcee36bcd73dbc49077b05966abcdd0f2b57c",
        ),
        (
            OutcomeRequestKind.ACTION,
            1016,
            2060,
            "58e04be8a82f9fd07db7681c89b61677563fa62d41d7c96600ca2162007b93de",
        ),
        (
            OutcomeRequestKind.QUESTION,
            998,
            2044,
            "3273a3ea1851cf38eeb17ae5916484c80d036ffd4782f3fe73124179bc39dbdf",
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            1007,
            2061,
            "9372056aabf59d6858b88d815549aaf02f524b7428039f9253abca154da79810",
        ),
    ),
)
async def test_fix36_http_system_prompt_binds_exact_native_dynamic_schema(
    kind: OutcomeRequestKind,
    expected_prefix_chars: int,
    expected_prefix_bytes: int,
    expected_prefix_sha256: str,
) -> None:
    server, requests = await _server(iter(({"complete": True, "candidates": []},)))
    entry = _entry(1, "Synthetic dynamic Schema protocol")
    topic = TopicSegment(
        "topic-001",
        "Topic 1",
        (entry.entry_id,),
        1_000,
        1_800,
        "deterministic_fallback",
    )
    partition = plan_outcome_partitions(topic, (entry,), (), allowed_participant_ids=(7,))[0]
    try:
        provider = _provider(server)
        _bind_provider(provider, (partition,))
        await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=kind,
            request_bytes=serialize_outcome_request(partition, kind),
        )
    finally:
        await server.close()

    assert len(requests) == 1
    observed = requests[0]
    system_message = observed["messages"][0]
    schema = outcome_response_schema(partition, kind)
    assert system_message == {
        "role": "system",
        "content": outcome_system_prompt(kind, schema=schema),
    }
    marker = "只返回精确匹配此 JSON Schema 的 JSON: "
    prose_prefix, embedded_schema = system_message["content"].split(marker, 1)
    assert len(prose_prefix) == expected_prefix_chars
    assert len(prose_prefix.encode("utf-8")) == expected_prefix_bytes
    assert hashlib.sha256(prose_prefix.encode("utf-8")).hexdigest() == expected_prefix_sha256
    assert embedded_schema == json.dumps(
        schema,
        ensure_ascii=False,
        separators=(",", ":"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invalid", "subtype"),
    (
        ("", OutcomeFormatSubtype.JSON_EMPTY),
        (
            '{"complete":true,"candidates":[]} trailing',
            OutcomeFormatSubtype.JSON_EXTRA_DATA,
        ),
        (
            '{"complete":true,"candidates":[{"text":"unterminated',
            OutcomeFormatSubtype.JSON_UNTERMINATED_STRING,
        ),
        ("{", OutcomeFormatSubtype.JSON_SYNTAX_OTHER),
    ),
)
async def test_fix30_json_failure_subtype_is_closed_without_response_content(
    invalid: str,
    subtype: OutcomeFormatSubtype,
) -> None:
    server, requests = await _server(iter((invalid, {"complete": True, "candidates": []})))
    entry = _entry(1, "PRIVATE_FIX30_CONTEXT_SENTINEL")
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(
            TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            (entry,),
            (),
        )[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()

    assert len(requests) == 2
    failure = result.provider_failures[0]
    assert failure.category is OutcomeFailureCategory.JSON
    assert failure.attempt is OutcomeFailureAttempt.INITIAL
    assert failure.format_subtype is subtype
    if invalid:
        assert invalid not in outcome_failure_token(failure)
    assert "PRIVATE_FIX30_CONTEXT_SENTINEL" not in outcome_failure_token(failure)
    assert requests[1]["messages"][:2] == requests[0]["messages"]
    _assert_deepseek_repair_payload(requests[0], requests[1])
    assert f"json:{subtype.value}" in requests[1]["messages"][-1]["content"]
    if invalid:
        assert invalid not in requests[1]["messages"][-1]["content"]
    assert "PRIVATE_FIX30_CONTEXT_SENTINEL" not in requests[1]["messages"][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "invalid", "valid", "subtype"),
    (
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [
                    {"text": "PRIVATE_INVALID_ENTRY_SENTINEL", "source_entry_refs": ["e9"]}
                ],
            },
            {
                "complete": True,
                "candidates": [{"text": "Question", "source_entry_refs": ["e0"]}],
            },
            OutcomeFormatSubtype.SCHEMA_ENTRY_AUTHORITY,
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "PRIVATE_INVALID_EVENT_SENTINEL",
                        "source_entry_refs": [],
                        "source_event_refs": ["v9"],
                    }
                ],
            },
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "Synthetic result",
                        "source_entry_refs": [],
                        "source_event_refs": ["v0"],
                    }
                ],
            },
            OutcomeFormatSubtype.SCHEMA_EVENT_AUTHORITY,
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "PRIVATE_INVALID_OWNER_SENTINEL",
                        "source_entry_refs": ["e0"],
                        "owner_ref": "s9",
                        "deadline": None,
                    }
                ],
            },
            {
                "complete": True,
                "candidates": [
                    {
                        "text": "Action",
                        "source_entry_refs": ["e0"],
                        "owner_ref": "s0",
                        "deadline": None,
                    }
                ],
            },
            OutcomeFormatSubtype.SCHEMA_OWNER_AUTHORITY,
        ),
    ),
)
@pytest.mark.parametrize("repair_succeeds", (True, False))
async def test_fix30_authority_failure_uses_at_most_two_content_safe_repairs(
    kind: OutcomeRequestKind,
    invalid: dict[str, object],
    valid: dict[str, object],
    subtype: OutcomeFormatSubtype,
    repair_succeeds: bool,
) -> None:
    responses = (invalid, valid) if repair_succeeds else (invalid, invalid, invalid)
    server, requests = await _server(iter(responses))
    entry = _entry(1, "Synthetic authority evidence")
    event = OutcomeEventFact(
        sequence_number=1,
        topic_id="topic-001",
        subject="Synthetic result",
        result="passed",
        source_entry_ids=(entry.entry_id,),
    )
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(
            TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            (entry,),
            (event,),
            allowed_participant_ids=(7,),
        )[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=kind,
            request_bytes=serialize_outcome_request(partition, kind),
        )
    finally:
        await server.close()

    assert len(requests) == (2 if repair_succeeds else 3)
    assert requests[1]["messages"][:2] == requests[0]["messages"]
    _assert_deepseek_repair_payload(requests[0], requests[1])
    assert [failure.format_subtype for failure in result.provider_failures] == (
        [subtype] if repair_succeeds else [subtype, subtype, subtype]
    )
    assert [failure.attempt for failure in result.provider_failures] == (
        [OutcomeFailureAttempt.INITIAL]
        if repair_succeeds
        else [
            OutcomeFailureAttempt.INITIAL,
            OutcomeFailureAttempt.REPAIR,
            OutcomeFailureAttempt.REPAIR,
        ]
    )
    assert len(result.candidates) == int(repair_succeeds)
    repair_instruction = requests[1]["messages"][-1]["content"]
    assert f"schema:{subtype.value}" in repair_instruction
    assert "PRIVATE_INVALID" not in repair_instruction
    assert "e9" not in repair_instruction
    assert "v9" not in repair_instruction
    assert "s9" not in repair_instruction


@pytest.mark.asyncio
async def test_fenced_outcome_json_is_not_accepted_as_the_strict_initial_wire_root() -> None:
    server, requests = await _server(
        iter(
            (
                '```json\n{"complete":true,"candidates":[]}\n```',
                {"complete": True, "candidates": []},
            )
        )
    )
    entry = _entry(1, "we discussed an outcome")
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(
            TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            (entry,),
            (),
        )[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()

    assert len(requests) == 2
    assert result.candidates == ()
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.JSON, OutcomeFailureAttempt.INITIAL)
    ]


@pytest.mark.asyncio
async def test_repaired_batch_noncore_reference_uses_repair_attempt_and_drops_entire_batch() -> (
    None
):
    entry = _entry(1, "we decided")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    invalid = _candidate("ignored", "action_item", "outside")
    invalid_response = {"candidates": [invalid]}
    server, requests = await _server(iter(("not-json", invalid_response, invalid_response)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.ACTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.ACTION),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert result.candidates == ()
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.JSON, OutcomeFailureAttempt.INITIAL),
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.REPAIR),
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.REPAIR),
    ]


@pytest.mark.asyncio
async def test_noncore_candidates_produce_one_task_failure_and_keep_none() -> None:
    entry = _entry(1, "we decided")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    response = {
        "candidates": [
            _candidate(f"ignored-{index}", "action_item", f"outside-{index}") for index in range(2)
        ]
    }
    server, requests = await _server(iter((response, response, response)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.ACTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.ACTION),
        )
    finally:
        await server.close()

    assert len(requests) == 3
    assert result.candidates == ()
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.INITIAL),
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.REPAIR),
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.REPAIR),
    ]


@pytest.mark.asyncio
async def test_malformed_batch_falls_back_without_blocking_other_extractors() -> None:
    responses = iter(
        (
            "not-json",
            "still-not-json",
            "still-not-json",
            {"candidates": []},
            {"candidates": [_candidate("question-1", "open_question", "e1")]},
            {"candidates": []},
        )
    )
    server, requests = await _server(responses)
    entry = _entry(1, "when will we release?")
    try:
        provider = _provider(server)
        result = await _extract_all(
            provider,
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert len(requests) == 6
    assert result.extraction_method == "deepseek_partial"
    assert result.failure_types == ()
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.JSON, OutcomeFailureAttempt.INITIAL),
        (OutcomeFailureCategory.JSON, OutcomeFailureAttempt.REPAIR),
        (OutcomeFailureCategory.JSON, OutcomeFailureAttempt.REPAIR),
    ]
    assert [candidate.type for candidate in result.candidates] == [OutcomeType.OPEN_QUESTION]


@pytest.mark.asyncio
async def test_raw_candidate_id_collision_preserves_candidates_with_internal_ids() -> None:
    first = _candidate("raw-collision", "decision", "e1")
    second = {
        **_candidate("raw-collision", "agreement", "e2"),
        "text": "a different supported agreement",
    }
    responses = iter(
        (
            {"candidates": [first, second]},
            {"candidates": []},
            {"candidates": []},
            {"candidates": []},
        )
    )
    server, requests = await _server(responses)
    entries = (_entry(1, "we decided"), _entry(2, "we agree"))
    try:
        provider = _provider(server)
        result = await _extract_all(
            provider,
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                tuple(entry.entry_id for entry in entries),
                1_000,
                2_800,
                "deterministic_fallback",
            ),
            entries=entries,
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert len(requests) == 4
    assert len(result.candidates) == 2
    assert len({candidate.id for candidate in result.candidates}) == 2
    assert all(candidate.id != "raw-collision" for candidate in result.candidates)
    assert "raw-collision" not in result.model_dump_json()
    assert result.provider_failures == ()


@pytest.mark.asyncio
async def test_same_batch_identical_semantics_fold_without_repair_or_model_failure() -> None:
    raw = _candidate("same-raw-id", "decision", "e1")
    responses = iter(
        (
            {"candidates": [raw, raw]},
            {"candidates": []},
            {"candidates": []},
            {"candidates": []},
            {"candidates": []},
        )
    )
    server, _ = await _server(responses)
    entry = _entry(1, "we decided")
    try:
        provider = _provider(server)
        result = await _extract_all(
            provider,
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert len(result.candidates) == 1
    assert result.provider_failures == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_options", "expected_category"),
    (
        ({"chat_status": 503}, OutcomeFailureCategory.HTTP_STATUS),
        ({"chat_body": b"not-an-envelope"}, OutcomeFailureCategory.JSON),
        ({"chat_body": b'{"done":true}'}, OutcomeFailureCategory.ENVELOPE),
        (
            {
                "chat_body": json.dumps(
                    {
                        "id": "content-free-boundary-test",
                        "object": "chat.completion",
                        "model": APPROVED_TOPIC_MODEL,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "length",
                                "message": {
                                    "role": "assistant",
                                    "content": '{"complete":true,"candidates":[]}',
                                },
                            }
                        ],
                    }
                ).encode()
            },
            OutcomeFailureCategory.OUTPUT_LIMIT,
        ),
        (
            {"chat_body": b"x" * (512 * 1024 + 1)},
            OutcomeFailureCategory.OUTPUT_LIMIT,
        ),
        (
            {"chat_delay_seconds": 0.05},
            OutcomeFailureCategory.TRANSPORT_TIMEOUT,
        ),
    ),
)
async def test_provider_boundary_failures_are_closed_and_content_safe(
    server_options: dict[str, Any],
    expected_category: OutcomeFailureCategory,
) -> None:
    sentinel = "PRIVATE_CONTENT_MUST_NOT_ESCAPE"
    server = await _boundary_server(**server_options)
    try:
        provider = _provider(
            server,
            timeout_seconds=0.01
            if expected_category is OutcomeFailureCategory.TRANSPORT_TIMEOUT
            else 1,
            api_key=sentinel,
        )
        failure = await _request_failure(provider)
    finally:
        await server.close()

    assert failure.category is expected_category
    assert failure.request_kind is OutcomeFailureRequestKind.ACTION
    assert failure.attempt is OutcomeFailureAttempt.INITIAL
    assert set(failure.model_dump()) == {
        "category",
        "request_kind",
        "attempt",
        "count",
        "size_bucket",
        "capacity_subtype",
        "format_subtype",
    }
    assert failure.format_subtype is (
        OutcomeFormatSubtype.JSON_DECODE
        if expected_category is OutcomeFailureCategory.JSON
        else None
    )
    assert failure.capacity_subtype is None
    assert sentinel not in failure.model_dump_json()
    assert sentinel not in repr(failure)
    assert provider.actual_chat_requests == 1
    assert provider.actual_chat_requests == provider._budget().journal.actual_inference_calls


@pytest.mark.asyncio
async def test_digest_mismatch_is_preflight_failure_without_chat_call() -> None:
    server = await _boundary_server(tags_digest="sha256:" + "0" * 64)
    entry = _entry(1, "we decided")
    provider = _provider(server)
    try:
        result = await _extract_all(
            provider,
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert {failure.category for failure in result.provider_failures} == {
        OutcomeFailureCategory.DIGEST
    }
    assert provider.actual_chat_requests == 0


@pytest.mark.asyncio
async def test_context_and_wall_preflight_make_zero_actual_chat_requests() -> None:
    server = await _boundary_server()
    entry = _entry(1, "we decided")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    partition = plan_outcome_partitions(topic, (entry,), ())[0]
    provider = _provider(server)
    controller, job_hash = _bind_provider(provider, (partition,))
    try:
        context_result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.ACTION,
            request_bytes=b"{}",
        )
        assert context_result.provider_failures[0].category is OutcomeFailureCategory.CONTEXT
        assert context_result.task_disposition is OutcomeTaskDisposition.TERMINAL_FAILURE
        assert provider.actual_chat_requests == 0

        expired = UnifiedLlmBudgetJournal.new(
            job_hash=job_hash,
            expected_task_hashes=controller.journal.expected_task_hashes,
            now=0.0,
        )
        provider.bind_llm_budget(LlmBudgetController(expired, lambda _journal: None))
        with pytest.raises(OutcomePartitionError, match="wall_budget"):
            await provider.extract_topic_outcome_kind(
                partition=partition,
                kind=OutcomeRequestKind.ACTION,
                request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.ACTION),
            )
    finally:
        await server.close()

    assert provider.actual_chat_requests == 0


def test_post_meeting_provider_rejects_an_alternate_endpoint() -> None:
    with pytest.raises(ValueError, match="approved official endpoint"):
        DeepSeekPostMeetingInferenceProvider(
            base_url="https://example.com/v1",
            model=APPROVED_TOPIC_MODEL,
            expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
            api_key="PRIVATE_DEEPSEEK_TEST_KEY",
            allow_remote=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_kwargs", "provider_kwargs", "category"),
    (
        (
            {"tags_digest": "sha256:" + "0" * 64},
            {},
            OutcomeFailureCategory.DIGEST,
        ),
        (
            {"tags_status": 503},
            {},
            OutcomeFailureCategory.HTTP_STATUS,
        ),
        (
            {"tags_delay_seconds": 0.05},
            {"timeout_seconds": 0.01},
            OutcomeFailureCategory.TRANSPORT_TIMEOUT,
        ),
    ),
)
async def test_digest_terminal_failure_completes_unified_ledger_without_generic_incomplete(
    server_kwargs: dict[str, Any],
    provider_kwargs: dict[str, Any],
    category: OutcomeFailureCategory,
) -> None:
    server = await _boundary_server(**server_kwargs)
    entry = _entry(1, "we decided")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    partition = plan_outcome_partitions(topic, (entry,), ())[0]
    provider = _provider(server, **provider_kwargs)
    controller, job_hash = _bind_provider(provider, (partition,))
    try:
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.ACTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.ACTION),
        )
    finally:
        await server.close()

    assert result.provider_failures[0].category is category
    digest_hash = digest_task_hash(job_hash)
    digest_action = next(
        action for action in controller.journal.actions if action.task_hash == digest_hash
    )
    assert digest_action.completed
    assert digest_action.failure_category is category
    assert digest_hash in controller.journal.result_task_hashes
    for kind in OutcomeRequestKind:
        task_hash = outcome_task_hash(job_hash, partition, kind)
        controller.reserve_task(task_hash)
        controller.reconcile_result(task_hash)
    controller.journal.validate_final()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inventory", "category"),
    (
        ([], OutcomeFailureCategory.DIGEST),
        ({"models": {}}, OutcomeFailureCategory.DIGEST),
        ({"models": [1]}, OutcomeFailureCategory.DIGEST),
        ({"models": [{}]}, OutcomeFailureCategory.DIGEST),
        ({"models": [{"name": 7, "digest": "a" * 64}]}, OutcomeFailureCategory.DIGEST),
        (
            {"models": [{"name": APPROVED_TOPIC_MODEL, "digest": 7}]},
            OutcomeFailureCategory.DIGEST,
        ),
        (
            {"models": [{"name": APPROVED_TOPIC_MODEL}]},
            OutcomeFailureCategory.DIGEST,
        ),
        ({"models": []}, OutcomeFailureCategory.DIGEST),
    ),
)
async def test_digest_inventory_shape_failure_is_closed_and_terminal(
    inventory: object,
    category: OutcomeFailureCategory,
) -> None:
    server = await _boundary_server(
        tags_body=json.dumps(inventory, separators=(",", ":")).encode("utf-8")
    )
    entry = _entry(1, "we decided")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    partition = plan_outcome_partitions(topic, (entry,), ())[0]
    provider = _provider(server)
    controller, job_hash = _bind_provider(provider, (partition,))
    try:
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.ACTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.ACTION),
        )
    finally:
        await server.close()

    assert result.provider_failures[0].category is category
    digest_hash = digest_task_hash(job_hash)
    digest_actions = [
        action for action in controller.journal.actions if action.task_hash == digest_hash
    ]
    assert len(digest_actions) == 1
    assert digest_actions[0].completed
    assert digest_actions[0].failure_category is category
    assert digest_hash in controller.journal.result_task_hashes
    for kind in OutcomeRequestKind:
        task_hash = outcome_task_hash(job_hash, partition, kind)
        controller.reserve_task(task_hash)
        controller.reconcile_result(task_hash)
    controller.journal.validate_final()


@pytest.mark.asyncio
async def test_invalid_utf8_model_inventory_is_closed_digest_failure() -> None:
    server = await _boundary_server(tags_body=b"\xffPRIVATE_BYTES")
    entry = _entry(1, "we decided")
    try:
        result = await _extract_all(
            _provider(server),
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert {failure.category for failure in result.provider_failures} == {
        OutcomeFailureCategory.DIGEST
    }
    assert "PRIVATE_BYTES" not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_body", "category"),
    (
        (b"\xffPRIVATE_BYTES", OutcomeFailureCategory.JSON),
        (
            b'{"message":{"content":"\\ud800PRIVATE_TEXT"}}',
            OutcomeFailureCategory.ENVELOPE,
        ),
    ),
)
async def test_invalid_chat_unicode_is_closed_without_content_leakage(
    chat_body: bytes,
    category: OutcomeFailureCategory,
) -> None:
    server = await _boundary_server(chat_body=chat_body)
    try:
        failure = await _request_failure(_provider(server))
    finally:
        await server.close()

    assert failure.category is category
    assert "PRIVATE" not in failure.model_dump_json()


@pytest.mark.asyncio
async def test_input_limit_is_classified_without_model_calls() -> None:
    server, requests = await _server(iter(()))
    entry = _entry(1, "x" * 1_100)
    try:
        result = await _extract_all(
            _provider(server, max_input_chars=1_000),
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert requests == []
    assert len(result.provider_failures) == 4
    assert {failure.category for failure in result.provider_failures} == {
        OutcomeFailureCategory.INPUT_LIMIT
    }
    assert {failure.attempt for failure in result.provider_failures} == {
        OutcomeFailureAttempt.PREFLIGHT
    }
    assert {failure.count for failure in result.provider_failures} == {1}
    assert {failure.request_kind for failure in result.provider_failures} == {
        OutcomeFailureRequestKind.DECISION_AGREEMENT,
        OutcomeFailureRequestKind.ACTION,
        OutcomeFailureRequestKind.QUESTION,
        OutcomeFailureRequestKind.PROPOSAL_RISK_DISAGREEMENT,
    }


@pytest.mark.asyncio
async def test_item_count_is_not_silently_clipped_before_partition_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = DeepSeekPostMeetingInferenceProvider(
        base_url=APPROVED_TOPIC_BASE_URL,
        model=APPROVED_TOPIC_MODEL,
        expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
        api_key="PRIVATE_DEEPSEEK_TEST_KEY",
        allow_remote=True,
    )
    provider._endpoint = "http://127.0.0.1:1/chat/completions"
    network_called = False

    async def unexpected_network() -> None:
        nonlocal network_called
        network_called = True

    monkeypatch.setattr(provider, "_verify_model_digest", unexpected_network)
    entries = tuple(_entry(index, "statement") for index in range(1, 82))
    result = await _extract_all(
        provider,
        topic=TopicSegment(
            "topic-001",
            "Topic 1",
            tuple(entry.entry_id for entry in entries),
            1_000,
            82_000,
            "deterministic_fallback",
        ),
        entries=entries,
        events=(),
        allowed_participant_ids=frozenset({7}),
    )

    assert network_called is True
    partition_count = len(
        plan_outcome_partitions(
            TopicSegment(
                "topic-001",
                "Topic 1",
                tuple(entry.entry_id for entry in entries),
                1_000,
                82_000,
                "deterministic_fallback",
            ),
            entries,
            (),
            (7,),
        )
    )
    assert len(result.provider_failures) == 4 * partition_count
    assert {failure.category for failure in result.provider_failures} == {
        OutcomeFailureCategory.TRANSPORT_ERROR
    }
    assert {failure.attempt for failure in result.provider_failures} == {
        OutcomeFailureAttempt.INITIAL
    }


@pytest.mark.asyncio
async def test_oversized_repair_context_is_preflight_and_makes_no_repair_call() -> None:
    server, requests = await _server(
        iter(
            (
                "x" * 4_097,
                {"candidates": []},
                {"candidates": []},
                {"candidates": []},
            )
        )
    )
    entry = _entry(1, "we discussed an outcome")
    try:
        result = await _extract_all(
            _provider(server),
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert len(requests) == 4
    assert [(failure.category, failure.attempt) for failure in result.provider_failures] == [
        (OutcomeFailureCategory.JSON, OutcomeFailureAttempt.INITIAL),
        (OutcomeFailureCategory.INPUT_LIMIT, OutcomeFailureAttempt.PREFLIGHT),
    ]
    assert all(failure.count == 1 for failure in result.provider_failures)


def test_provider_failure_count_is_exactly_one_observed_boundary_event() -> None:
    with pytest.raises(ValidationError):
        OutcomeProviderFailure(
            category=OutcomeFailureCategory.HTTP_STATUS,
            request_kind=OutcomeFailureRequestKind.ACTION,
            attempt=OutcomeFailureAttempt.INITIAL,
            count=2,
            size_bucket=OutcomeSizeBucket.NOT_OBSERVED,
            capacity_subtype=None,
        )


def test_fix24_provider_failure_capacity_subtype_is_closed_and_required() -> None:
    base = {
        "category": OutcomeFailureCategory.OUTPUT_CAPACITY,
        "request_kind": OutcomeFailureRequestKind.ACTION,
        "attempt": OutcomeFailureAttempt.INITIAL,
        "count": 1,
        "size_bucket": OutcomeSizeBucket.LTE_1K,
    }
    with pytest.raises(ValidationError):
        OutcomeProviderFailure.model_validate(base)
    with pytest.raises(ValidationError):
        OutcomeProviderFailure.model_validate({**base, "capacity_subtype": None})
    with pytest.raises(ValidationError):
        OutcomeProviderFailure.model_validate(
            {
                **base,
                "category": OutcomeFailureCategory.JSON,
                "capacity_subtype": OutcomeCapacitySubtype.TEXT_CHARS,
            }
        )
    with pytest.raises(ValidationError):
        OutcomeProviderFailure.model_validate({**base, "capacity_subtype": "unknown"})
    with pytest.raises(ValidationError):
        OutcomeProviderFailure.model_validate(
            {**base, "capacity_subtype": OutcomeCapacitySubtype.TEXT_CHARS, "extra": None}
        )

    failure = OutcomeProviderFailure.model_validate(
        {**base, "capacity_subtype": OutcomeCapacitySubtype.TEXT_CHARS}
    )
    assert outcome_failure_token(failure) == (
        "output_capacity:action:initial:1:lte_1k:text_chars:none"
    )


def test_fix26_json_schema_failure_subtypes_are_closed_and_content_safe() -> None:
    base = {
        "request_kind": OutcomeFailureRequestKind.ACTION,
        "attempt": OutcomeFailureAttempt.INITIAL,
        "count": 1,
        "size_bucket": OutcomeSizeBucket.LTE_1K,
        "capacity_subtype": None,
    }
    json_subtypes = {
        OutcomeFormatSubtype.JSON_DECODE,
        OutcomeFormatSubtype.JSON_EMPTY,
        OutcomeFormatSubtype.JSON_EXTRA_DATA,
        OutcomeFormatSubtype.JSON_UNTERMINATED_STRING,
        OutcomeFormatSubtype.JSON_SYNTAX_OTHER,
    }
    reachable_schema_subtypes = set(OutcomeFormatSubtype) - json_subtypes
    assert reachable_schema_subtypes == {
        OutcomeFormatSubtype.SCHEMA_ROOT,
        OutcomeFormatSubtype.SCHEMA_CANDIDATE_FIELDS,
        OutcomeFormatSubtype.SCHEMA_TYPE,
        OutcomeFormatSubtype.SCHEMA_TEXT,
        OutcomeFormatSubtype.SCHEMA_ENTRY_REFS,
        OutcomeFormatSubtype.SCHEMA_EVENT_REFS,
        OutcomeFormatSubtype.SCHEMA_EVIDENCE_EMPTY,
        OutcomeFormatSubtype.SCHEMA_ENTRY_AUTHORITY,
        OutcomeFormatSubtype.SCHEMA_EVENT_AUTHORITY,
        OutcomeFormatSubtype.SCHEMA_OWNER_AUTHORITY,
        OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
        OutcomeFormatSubtype.DUPLICATE_BINDING_CANONICAL_ALIAS_COLLISION,
    }
    reachable_schema_kinds = {
        OutcomeFormatSubtype.SCHEMA_ROOT: set(OutcomeRequestKind),
        OutcomeFormatSubtype.SCHEMA_CANDIDATE_FIELDS: set(OutcomeRequestKind),
        OutcomeFormatSubtype.SCHEMA_TYPE: {
            OutcomeRequestKind.DECISION_AGREEMENT,
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
        },
        OutcomeFormatSubtype.SCHEMA_TEXT: set(OutcomeRequestKind),
        OutcomeFormatSubtype.SCHEMA_ENTRY_REFS: set(OutcomeRequestKind),
        OutcomeFormatSubtype.SCHEMA_EVENT_REFS: {OutcomeRequestKind.DECISION_AGREEMENT},
        OutcomeFormatSubtype.SCHEMA_EVIDENCE_EMPTY: {OutcomeRequestKind.DECISION_AGREEMENT},
        OutcomeFormatSubtype.SCHEMA_ENTRY_AUTHORITY: set(OutcomeRequestKind),
        OutcomeFormatSubtype.SCHEMA_EVENT_AUTHORITY: {OutcomeRequestKind.DECISION_AGREEMENT},
        OutcomeFormatSubtype.SCHEMA_OWNER_AUTHORITY: {OutcomeRequestKind.ACTION},
        OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT: set(OutcomeRequestKind),
        OutcomeFormatSubtype.DUPLICATE_BINDING_CANONICAL_ALIAS_COLLISION: set(OutcomeRequestKind),
    }
    assert set(reachable_schema_kinds) == reachable_schema_subtypes
    for category, subtype, request_kind in (
        *(
            (OutcomeFailureCategory.JSON, subtype, OutcomeFailureRequestKind.ACTION)
            for subtype in json_subtypes
        ),
        *(
            (
                OutcomeFailureCategory.SCHEMA,
                subtype,
                OutcomeFailureRequestKind(kind.value),
            )
            for subtype, kinds in reachable_schema_kinds.items()
            for kind in kinds
        ),
    ):
        failure = OutcomeProviderFailure.model_validate(
            {
                **base,
                "category": category,
                "request_kind": request_kind,
                "format_subtype": subtype,
            }
        )
        token = outcome_failure_token(failure)
        assert subtype.value in token
        assert "PRIVATE_CONTENT_SENTINEL" not in token

    for subtype, kinds in reachable_schema_kinds.items():
        for kind in set(OutcomeRequestKind) - kinds:
            with pytest.raises(ValidationError):
                OutcomeProviderFailure.model_validate(
                    {
                        **base,
                        "category": OutcomeFailureCategory.SCHEMA,
                        "request_kind": OutcomeFailureRequestKind(kind.value),
                        "format_subtype": subtype,
                    }
                )

    for payload in (
        {**base, "category": OutcomeFailureCategory.JSON, "format_subtype": None},
        {
            **base,
            "category": OutcomeFailureCategory.JSON,
            "format_subtype": OutcomeFormatSubtype.SCHEMA_CANDIDATE_FIELDS,
        },
        {
            **base,
            "category": OutcomeFailureCategory.SCHEMA,
            "format_subtype": OutcomeFormatSubtype.JSON_DECODE,
        },
        {
            **base,
            "category": OutcomeFailureCategory.HTTP_STATUS,
            "format_subtype": OutcomeFormatSubtype.SCHEMA_OWNER_AUTHORITY,
        },
        {
            **base,
            "category": OutcomeFailureCategory.JSON,
            "request_kind": OutcomeFailureRequestKind.PREFLIGHT,
            "format_subtype": OutcomeFormatSubtype.JSON_DECODE,
        },
        {
            **base,
            "category": OutcomeFailureCategory.SCHEMA,
            "attempt": OutcomeFailureAttempt.PREFLIGHT,
            "format_subtype": OutcomeFormatSubtype.SCHEMA_CANDIDATE_FIELDS,
        },
        {
            **base,
            "category": OutcomeFailureCategory.SCHEMA,
            "request_kind": OutcomeFailureRequestKind.PREFLIGHT,
            "format_subtype": OutcomeFormatSubtype.SCHEMA_ENTRY_AUTHORITY,
        },
        {
            **base,
            "category": OutcomeFailureCategory.JSON,
            "attempt": OutcomeFailureAttempt.PREFLIGHT,
            "format_subtype": OutcomeFormatSubtype.JSON_EMPTY,
        },
    ):
        with pytest.raises(ValidationError):
            OutcomeProviderFailure.model_validate(payload)

    preflight_json = OutcomeProviderFailure.model_validate(
        {
            **base,
            "category": OutcomeFailureCategory.JSON,
            "attempt": OutcomeFailureAttempt.PREFLIGHT,
            "format_subtype": OutcomeFormatSubtype.JSON_DECODE,
        }
    )
    assert preflight_json.attempt is OutcomeFailureAttempt.PREFLIGHT


@pytest.mark.parametrize(
    ("kind", "candidate"),
    (
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "type": "decision",
                "text": "supported decision",
                "source_entry_refs": ["e0"],
                "source_event_refs": [],
            },
        ),
        (
            OutcomeRequestKind.ACTION,
            {
                "text": "supported action",
                "source_entry_refs": ["e0"],
                "owner_ref": None,
                "deadline": None,
            },
        ),
        (
            OutcomeRequestKind.QUESTION,
            {"text": "supported question", "source_entry_refs": ["e0"]},
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            {
                "type": "risk",
                "text": "supported risk",
                "source_entry_refs": ["e0"],
            },
        ),
    ),
)
def test_fix29_duplicate_binding_is_closed_before_internal_identity(
    kind: OutcomeRequestKind,
    candidate: dict[str, object],
) -> None:
    divergent = {**candidate, "text": "different supported wording"}
    folded = parse_outcome_batch(
        json.dumps({"complete": True, "candidates": [candidate, candidate]}),
        kind,
    )
    assert len(folded) == 1
    with pytest.raises(OutcomeResponseSchemaError) as captured:
        parse_outcome_batch(
            json.dumps({"complete": True, "candidates": [candidate, divergent]}),
            kind,
        )
    assert captured.value.subtype is OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT


def test_fix29_ordered_evidence_bindings_remain_distinct() -> None:
    candidates = (
        {"text": "first supported question", "source_entry_refs": ["e0", "e1"]},
        {"text": "second supported question", "source_entry_refs": ["e1", "e0"]},
    )

    parsed = parse_outcome_batch(
        json.dumps({"complete": True, "candidates": candidates}),
        OutcomeRequestKind.QUESTION,
    )

    assert tuple(candidate.source_entry_refs for candidate in parsed) == (
        ("e0", "e1"),
        ("e1", "e0"),
    )


def test_fix29_partition_aliases_collapsing_to_one_canonical_binding_are_rejected() -> None:
    entry = _entry(1, "synthetic evidence")
    partition = plan_outcome_partitions(
        TopicSegment(
            "topic-001",
            "Topic 1",
            (entry.entry_id,),
            1_000,
            1_800,
            "deterministic_fallback",
        ),
        (entry,),
        (),
    )[0]
    first = partition.core_entries[0]
    partition = partition.model_copy(
        update={
            "core_entries": (
                first,
                first.model_copy(update={"fragment_index": first.fragment_index + 1}),
            )
        }
    )
    response = {
        "complete": True,
        "candidates": [
            {"text": "first supported question", "source_entry_refs": ["e0"]},
            {"text": "second supported question", "source_entry_refs": ["e1"]},
        ],
    }

    with pytest.raises(OutcomeResponseSchemaError) as captured:
        parse_partition_outcome_batch(
            json.dumps(response),
            OutcomeRequestKind.QUESTION,
            partition,
        )

    assert (
        captured.value.subtype is OutcomeFormatSubtype.DUPLICATE_BINDING_CANONICAL_ALIAS_COLLISION
    )


def test_fix32_partition_alias_collision_exact_is_closed_without_private_values() -> None:
    entry = _entry(1, "synthetic evidence")
    partition = plan_outcome_partitions(
        TopicSegment(
            "topic-001",
            "Topic 1",
            (entry.entry_id,),
            1_000,
            1_800,
            "deterministic_fallback",
        ),
        (entry,),
        (),
    )[0]
    first = partition.core_entries[0]
    partition = partition.model_copy(
        update={
            "core_entries": (
                first,
                first.model_copy(update={"fragment_index": first.fragment_index + 1}),
            )
        }
    )
    response = {
        "complete": True,
        "candidates": [
            {"text": "same supported question", "source_entry_refs": ["e0"]},
            {"text": "same supported question", "source_entry_refs": ["e1"]},
        ],
    }

    with pytest.raises(OutcomeResponseSchemaError) as captured:
        parse_partition_outcome_batch(
            json.dumps(response),
            OutcomeRequestKind.QUESTION,
            partition,
        )

    assert (
        captured.value.subtype is OutcomeFormatSubtype.DUPLICATE_BINDING_CANONICAL_ALIAS_COLLISION
    )


@pytest.mark.parametrize(
    "update",
    (
        {"text": "different supported action"},
        {"owner_ref": "s1"},
        {"deadline": "Friday"},
    ),
)
def test_fix29_action_binding_rejects_duplicates_with_other_field_differences(
    update: dict[str, object],
) -> None:
    candidate = {
        "text": "supported action",
        "source_entry_refs": ["e0"],
        "owner_ref": "s0",
        "deadline": None,
    }

    with pytest.raises(OutcomeResponseSchemaError) as captured:
        parse_outcome_batch(
            json.dumps(
                {
                    "complete": True,
                    "candidates": [candidate, {**candidate, **update}],
                }
            ),
            OutcomeRequestKind.ACTION,
            allowed_owner_aliases=("s0", "s1"),
        )

    assert captured.value.subtype is OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_succeeds", (True, False))
async def test_fix29_duplicate_binding_uses_at_most_two_content_safe_repairs(
    repair_succeeds: bool,
) -> None:
    first = _candidate("private-raw-a", "open_question", "e1")
    second = {**first, "id": "private-raw-b", "text": "different supported wording"}
    repair_candidates = [first] if repair_succeeds else [first, second]
    responses = (
        ({"candidates": [first, second]}, {"candidates": repair_candidates})
        if repair_succeeds
        else (
            {"candidates": [first, second]},
            {"candidates": repair_candidates},
            {"candidates": repair_candidates},
        )
    )
    server, requests = await _server(iter(responses))
    entry = _entry(1, "synthetic evidence")
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(
            TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            (entry,),
            (),
        )[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=OutcomeRequestKind.QUESTION,
            request_bytes=serialize_outcome_request(partition, OutcomeRequestKind.QUESTION),
        )
    finally:
        await server.close()

    assert len(requests) == (2 if repair_succeeds else 3)
    assert [failure.format_subtype for failure in result.provider_failures] == (
        [OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT]
        if repair_succeeds
        else [
            OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
            OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
            OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
        ]
    )
    assert [failure.attempt for failure in result.provider_failures] == (
        [OutcomeFailureAttempt.INITIAL]
        if repair_succeeds
        else [
            OutcomeFailureAttempt.INITIAL,
            OutcomeFailureAttempt.REPAIR,
            OutcomeFailureAttempt.REPAIR,
        ]
    )
    assert len(result.candidates) == int(repair_succeeds)
    repair_instruction = requests[1]["messages"][-1]["content"]
    assert "schema:duplicate_binding_wire_divergent" in repair_instruction
    assert "private-raw" not in repair_instruction
    assert "different supported wording" not in repair_instruction


@pytest.mark.asyncio
async def test_fix29_four_duplicate_batches_preserve_eight_call_ceiling() -> None:
    by_kind = (
        _candidate("ignored-da", "decision", "e1"),
        _candidate("ignored-action", "action_item", "e1"),
        _candidate("ignored-question", "open_question", "e1"),
        _candidate("ignored-prd", "risk", "e1"),
    )
    responses = tuple(
        {"candidates": [candidate, {**candidate, "id": f"{candidate['id']}-duplicate"}]}
        for candidate in by_kind
    )
    server, requests = await _server(iter(responses))
    entry = _entry(1, "synthetic evidence")
    try:
        result = await _extract_all(
            _provider(server),
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert len(requests) == 4
    assert len(result.candidates) == 4
    assert result.provider_failures == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "invalid", "valid", "subtype"),
    (
        (
            OutcomeRequestKind.QUESTION,
            [],
            {"complete": True, "candidates": []},
            OutcomeFormatSubtype.SCHEMA_ROOT,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {"complete": True, "candidates": [None]},
            {"complete": True, "candidates": []},
            OutcomeFormatSubtype.SCHEMA_CANDIDATE_FIELDS,
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            {
                "complete": True,
                "candidates": [{"type": 1, "text": "x", "source_entry_refs": ["e0"]}],
            },
            {"complete": True, "candidates": []},
            OutcomeFormatSubtype.SCHEMA_TYPE,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [{"text": None, "source_entry_refs": ["e0"]}],
            },
            {"complete": True, "candidates": []},
            OutcomeFormatSubtype.SCHEMA_TEXT,
        ),
        (
            OutcomeRequestKind.QUESTION,
            {
                "complete": True,
                "candidates": [{"text": "x", "source_entry_refs": "e0"}],
            },
            {"complete": True, "candidates": []},
            OutcomeFormatSubtype.SCHEMA_ENTRY_REFS,
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "x",
                        "source_entry_refs": [],
                        "source_event_refs": "v0",
                    }
                ],
            },
            {"complete": True, "candidates": []},
            OutcomeFormatSubtype.SCHEMA_EVENT_REFS,
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            {
                "complete": True,
                "candidates": [
                    {
                        "type": "decision",
                        "text": "x",
                        "source_entry_refs": [],
                        "source_event_refs": [],
                    }
                ],
            },
            {"complete": True, "candidates": []},
            OutcomeFormatSubtype.SCHEMA_EVIDENCE_EMPTY,
        ),
    ),
)
async def test_fix27_provider_retains_initial_closed_schema_subtype_after_repair(
    kind: OutcomeRequestKind,
    invalid: object,
    valid: object,
    subtype: OutcomeFormatSubtype,
) -> None:
    entry = _entry(1, "synthetic evidence")
    topic = TopicSegment(
        "topic-001", "Topic 1", (entry.entry_id,), 1_000, 1_800, "deterministic_fallback"
    )
    server, requests = await _server(iter((invalid, valid)))
    try:
        provider = _provider(server)
        partition = plan_outcome_partitions(topic, (entry,), ())[0]
        _bind_provider(provider, (partition,))
        result = await provider.extract_topic_outcome_kind(
            partition=partition,
            kind=kind,
            request_bytes=serialize_outcome_request(partition, kind),
        )
    finally:
        await server.close()

    assert len(requests) == 2
    assert result.provider_failures[0].attempt is OutcomeFailureAttempt.INITIAL
    assert result.provider_failures[0].format_subtype is subtype
    assert f"schema:{subtype.value}" in requests[1]["messages"][-1]["content"]
    assert "synthetic evidence" not in outcome_failure_token(result.provider_failures[0])


def test_fix24_every_capacity_subtype_has_a_reachable_strict_failure_fact() -> None:
    common_response_subtypes = {
        OutcomeCapacitySubtype.COMPLETE_FALSE,
        OutcomeCapacitySubtype.ROW_SATURATION,
        OutcomeCapacitySubtype.TEXT_CHARS,
        OutcomeCapacitySubtype.ENTRY_REFS,
    }
    reachable = {
        *common_response_subtypes,
        OutcomeCapacitySubtype.EVENT_REFS,
        OutcomeCapacitySubtype.OWNER_REF,
        OutcomeCapacitySubtype.DEADLINE,
        OutcomeCapacitySubtype.EVENT_SUBJECT,
        OutcomeCapacitySubtype.EVENT_ROWS,
        OutcomeCapacitySubtype.AUTHORITATIVE_ENTRY_REFS,
    }
    assert reachable == set(OutcomeCapacitySubtype)

    for subtype in common_response_subtypes:
        for request_kind in (
            OutcomeFailureRequestKind.DECISION_AGREEMENT,
            OutcomeFailureRequestKind.ACTION,
            OutcomeFailureRequestKind.QUESTION,
            OutcomeFailureRequestKind.PROPOSAL_RISK_DISAGREEMENT,
        ):
            for attempt in (OutcomeFailureAttempt.INITIAL, OutcomeFailureAttempt.REPAIR):
                OutcomeProviderFailure(
                    category=OutcomeFailureCategory.OUTPUT_CAPACITY,
                    request_kind=request_kind,
                    attempt=attempt,
                    count=1,
                    size_bucket=OutcomeSizeBucket.LTE_1K,
                    capacity_subtype=subtype,
                )

    for subtype, request_kind, attempt in (
        (
            OutcomeCapacitySubtype.EVENT_REFS,
            OutcomeFailureRequestKind.DECISION_AGREEMENT,
            OutcomeFailureAttempt.INITIAL,
        ),
        (
            OutcomeCapacitySubtype.OWNER_REF,
            OutcomeFailureRequestKind.ACTION,
            OutcomeFailureAttempt.REPAIR,
        ),
        (
            OutcomeCapacitySubtype.DEADLINE,
            OutcomeFailureRequestKind.ACTION,
            OutcomeFailureAttempt.INITIAL,
        ),
        (
            OutcomeCapacitySubtype.EVENT_SUBJECT,
            OutcomeFailureRequestKind.DECISION_AGREEMENT,
            OutcomeFailureAttempt.PREFLIGHT,
        ),
        (
            OutcomeCapacitySubtype.EVENT_ROWS,
            OutcomeFailureRequestKind.DECISION_AGREEMENT,
            OutcomeFailureAttempt.PREFLIGHT,
        ),
        (
            OutcomeCapacitySubtype.AUTHORITATIVE_ENTRY_REFS,
            OutcomeFailureRequestKind.DECISION_AGREEMENT,
            OutcomeFailureAttempt.PREFLIGHT,
        ),
    ):
        OutcomeProviderFailure(
            category=OutcomeFailureCategory.OUTPUT_CAPACITY,
            request_kind=request_kind,
            attempt=attempt,
            count=1,
            size_bucket=OutcomeSizeBucket.NOT_OBSERVED,
            capacity_subtype=subtype,
        )


@pytest.mark.parametrize(
    ("request_kind", "attempt", "subtype"),
    (
        (
            OutcomeFailureRequestKind.ACTION,
            OutcomeFailureAttempt.INITIAL,
            OutcomeCapacitySubtype.EVENT_SUBJECT,
        ),
        (
            OutcomeFailureRequestKind.QUESTION,
            OutcomeFailureAttempt.REPAIR,
            OutcomeCapacitySubtype.OWNER_REF,
        ),
        (
            OutcomeFailureRequestKind.PREFLIGHT,
            OutcomeFailureAttempt.INITIAL,
            OutcomeCapacitySubtype.DEADLINE,
        ),
        (
            OutcomeFailureRequestKind.DECISION_AGREEMENT,
            OutcomeFailureAttempt.INITIAL,
            OutcomeCapacitySubtype.EVENT_ROWS,
        ),
        (
            OutcomeFailureRequestKind.ACTION,
            OutcomeFailureAttempt.PREFLIGHT,
            OutcomeCapacitySubtype.COMPLETE_FALSE,
        ),
        (
            OutcomeFailureRequestKind.QUESTION,
            OutcomeFailureAttempt.INITIAL,
            OutcomeCapacitySubtype.EVENT_REFS,
        ),
        (
            OutcomeFailureRequestKind.DECISION_AGREEMENT,
            OutcomeFailureAttempt.REPAIR,
            OutcomeCapacitySubtype.DEADLINE,
        ),
    ),
)
def test_fix24_impossible_capacity_taxonomy_facts_are_rejected(
    request_kind: OutcomeFailureRequestKind,
    attempt: OutcomeFailureAttempt,
    subtype: OutcomeCapacitySubtype,
) -> None:
    with pytest.raises(ValidationError):
        OutcomeProviderFailure(
            category=OutcomeFailureCategory.OUTPUT_CAPACITY,
            request_kind=request_kind,
            attempt=attempt,
            count=1,
            size_bucket=OutcomeSizeBucket.NOT_OBSERVED,
            capacity_subtype=subtype,
        )


@pytest.mark.asyncio
async def test_every_invalid_response_has_two_repairs_and_preserves_call_budget() -> None:
    server, requests = await _server(iter("not-json" for _ in range(12)))
    entry = _entry(1, "we discussed an outcome")
    try:
        result = await _extract_all(
            _provider(server),
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert len(requests) == 12
    assert len(result.provider_failures) == 12
    assert Counter(failure.attempt for failure in result.provider_failures) == {
        OutcomeFailureAttempt.INITIAL: 4,
        OutcomeFailureAttempt.REPAIR: 8,
    }


@pytest.mark.asyncio
async def test_two_partitions_make_exactly_four_calls_each_without_repair() -> None:
    server, requests = await _server(iter({"candidates": []} for _ in range(8)))
    provider = _provider(server)
    entry = _entry(1, "x" * 5_000)
    topic = TopicSegment(
        "topic-001",
        "Topic 1",
        (entry.entry_id,),
        1_000,
        1_800,
        "deterministic_fallback",
    )
    try:
        partitions = plan_outcome_partitions(topic, (entry,), ())
        assert len(partitions) == 2
        _bind_provider(provider, partitions)
        for partition in partitions:
            for kind in OutcomeRequestKind:
                result = await provider.extract_topic_outcome_kind(
                    partition=partition,
                    kind=kind,
                    request_bytes=serialize_outcome_request(partition, kind),
                )
                assert not result.provider_failures
    finally:
        await server.close()

    assert len(requests) == 8


@pytest.mark.asyncio
async def test_closed_schema_failures_are_distinct_from_json_failures() -> None:
    invalid_schema = {"candidates": [{"id": "raw-only"}]}
    server, requests = await _server(
        iter(
            (
                invalid_schema,
                invalid_schema,
                invalid_schema,
                {"candidates": []},
                {"candidates": []},
                {"candidates": []},
            )
        )
    )
    entry = _entry(1, "we discussed an outcome")
    try:
        result = await _extract_all(
            _provider(server),
            topic=TopicSegment(
                "topic-001",
                "Topic 1",
                (entry.entry_id,),
                1_000,
                1_800,
                "deterministic_fallback",
            ),
            entries=(entry,),
            events=(),
            allowed_participant_ids=frozenset({7}),
        )
    finally:
        await server.close()

    assert len(requests) == 6
    assert [(failure.category, failure.attempt) for failure in result.provider_failures[:3]] == [
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.INITIAL),
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.REPAIR),
        (OutcomeFailureCategory.SCHEMA, OutcomeFailureAttempt.REPAIR),
    ]


@pytest.mark.asyncio
async def test_transport_connection_failure_uses_closed_category() -> None:
    server = await _boundary_server(chat_abort=True)
    provider = _provider(server)
    try:
        failure = await _request_failure(provider)
    finally:
        await server.close()

    assert failure.category is OutcomeFailureCategory.TRANSPORT_ERROR
    assert provider.actual_chat_requests == 1
