import json
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from lingxu.domain.errors import SummaryProviderError
from lingxu.transcript.models import CanonicalTranscriptEntry
from lingxu.worker.outcome_partitions import (
    LlmBudgetController,
    UnifiedLlmBudgetJournal,
    boundary_task_hash,
    digest_task_hash,
    llm_job_hash,
)
from lingxu.worker.summary import DeepSeekPostMeetingInferenceProvider
from lingxu.worker.topics import (
    APPROVED_TOPIC_BASE_URL,
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    TOPIC_CANDIDATE_CONFIG_VERSION,
    TopicBoundaryCandidate,
    TopicBoundaryProviderPolicy,
    topic_boundary_policy_failure,
)


def _bind_provider(
    provider: DeepSeekPostMeetingInferenceProvider,
    *,
    boundary_entry_id: str | None = None,
) -> LlmBudgetController:
    job_hash = llm_job_hash(canonical_hash="a" * 64, pipeline_revision="b" * 64)
    expected = [digest_task_hash(job_hash)]
    if boundary_entry_id is not None:
        expected.append(boundary_task_hash(job_hash, boundary_entry_id))
    controller = LlmBudgetController(
        UnifiedLlmBudgetJournal.new(
            job_hash=job_hash,
            expected_task_hashes=tuple(expected),
        ),
        lambda _journal: None,
    )
    provider.bind_llm_budget(controller)
    return controller


def _topic_entry(index: int, text: str) -> CanonicalTranscriptEntry:
    return CanonicalTranscriptEntry(
        entry_id=f"entry-{index}",
        source_entry_ids=(f"source-{index}",),
        speaker_user_id=100,
        speaker_name="主持人甲",
        started_ms=index * 1_000,
        ended_ms=index * 1_000 + 800,
        raw_texts=(text,),
        text=text,
        entry_type="statement",
        language="zh",
        confidence=0.9,
        utterance_group_id=f"group-{index}",
        normalisation_changes=(),
        asr_tags=(),
        emotion=None,
    )


async def _server(
    content: dict[str, Any] | list[Any] | str,
    *,
    response_model: str = APPROVED_TOPIC_MODEL,
    finish_reason: str = "stop",
    fenced: bool = False,
    inventory_model: str = APPROVED_TOPIC_MODEL,
) -> tuple[TestServer, list[dict[str, Any]], list[str | None]]:
    requests: list[dict[str, Any]] = []
    authorizations: list[str | None] = []

    async def models(request: web.Request) -> web.Response:
        authorizations.append(request.headers.get("Authorization"))
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": inventory_model,
                        "object": "model",
                        "owned_by": "deepseek",
                    }
                ],
            }
        )

    async def complete(request: web.Request) -> web.Response:
        authorizations.append(request.headers.get("Authorization"))
        requests.append(await request.json())
        encoded = content if isinstance(content, str) else json.dumps(content)
        if fenced:
            encoded = f"```json\n{encoded}\n```"
        return web.json_response(
            {
                "id": "content-free-test",
                "object": "chat.completion",
                "model": response_model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": encoded},
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_get("/models", models)
    app.router.add_post("/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    return server, requests, authorizations


def _provider(
    server: TestServer, *, api_key: str = "test-only-sentinel"
) -> DeepSeekPostMeetingInferenceProvider:
    provider = DeepSeekPostMeetingInferenceProvider(
        base_url=APPROVED_TOPIC_BASE_URL,
        model=APPROVED_TOPIC_MODEL,
        expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
        api_key=api_key,
        max_output_tokens=2_048,
        allow_remote=True,
    )
    provider._models_endpoint = str(server.make_url("/models"))
    provider._endpoint = str(server.make_url("/chat/completions"))
    return provider


def test_topic_boundary_policy_rejects_retired_ollama_provider() -> None:
    class RetiredOllamaProvider:
        topic_boundary_policy = TopicBoundaryProviderPolicy(
            provider="ollama",
            base_url="http://lingxu-ollama:11434/v1",
            model="qwen3.5:9b",
            expected_model_digest=(
                "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
            ),
            allow_remote=False,
        )

    assert topic_boundary_policy_failure(RetiredOllamaProvider()) == "provider_not_approved"


@pytest.mark.asyncio
async def test_deepseek_topic_confirmation_uses_one_pinned_nonthinking_path() -> None:
    sentinel = "PRIVATE_DEEPSEEK_TEST_KEY"
    server, requests, authorizations = await _server({"confirmed": True, "confidence": 0.91})
    try:
        provider = _provider(server, api_key=sentinel)
        controller = _bind_provider(provider, boundary_entry_id="entry-1")
        result = await provider.confirm_topic_boundary(
            candidate=TopicBoundaryCandidate("entry-1", ("transition_phrase",), 0.55),
            before=tuple(_topic_entry(index, f"before {index}") for index in range(10)),
            after=tuple(_topic_entry(index, f"after {index}") for index in range(1, 11)),
        )
    finally:
        await server.close()

    assert result.entry_id == "entry-1"
    assert result.confirmed is True
    assert authorizations == [f"Bearer {sentinel}", f"Bearer {sentinel}"]
    assert len(requests) == 1
    payload = requests[0]
    assert set(payload) == {
        "model",
        "messages",
        "stream",
        "thinking",
        "response_format",
        "temperature",
        "max_tokens",
    }
    assert payload["model"] == APPROVED_TOPIC_MODEL
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0
    assert payload["max_tokens"] <= 128
    context = json.loads(payload["messages"][1]["content"])
    assert len(context["before"]) == 10
    assert len(context["after"]) == 10
    assert set(context) == {"candidate", "before", "after"}
    assert topic_boundary_policy_failure(provider) is None
    assert provider.actual_chat_requests == controller.journal.actual_inference_calls == 1
    assert TOPIC_CANDIDATE_CONFIG_VERSION == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", (True, False))
async def test_deepseek_closed_topic_object_preserves_confirmation(confirmed: bool) -> None:
    server, requests, _ = await _server({"confirmed": confirmed, "confidence": 0.91})
    candidate = TopicBoundaryCandidate("entry-1", ("transition_phrase",), 0.55)
    try:
        provider = _provider(server)
        _bind_provider(provider, boundary_entry_id=candidate.entry_id)
        result = await provider.confirm_topic_boundary(
            candidate=candidate,
            before=(_topic_entry(0, "before"),),
            after=(_topic_entry(1, "after"),),
        )
    finally:
        await server.close()

    assert result.entry_id == candidate.entry_id
    assert result.confirmed is confirmed
    assert len(requests) == 1
    system_prompt = requests[0]["messages"][0]["content"]
    assert '"additionalProperties":false' in system_prompt
    assert '"required":["confirmed","confidence"]' in system_prompt
    assert "entry_id" not in system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    (
        "not-json",
        [True, 0.91],
        {"entry_id": "entry-1", "confirmed": True, "confidence": 0.99},
        {"confirmed": "true", "confidence": "0.99"},
    ),
)
async def test_deepseek_topic_confirmation_rejects_non_schema_response(response: Any) -> None:
    server, requests, _ = await _server(response)
    try:
        provider = _provider(server)
        _bind_provider(provider, boundary_entry_id="entry-1")
        with pytest.raises(SummaryProviderError):
            await provider.confirm_topic_boundary(
                candidate=TopicBoundaryCandidate("entry-1", ("transition_phrase",), 0.55),
                before=(_topic_entry(0, "before"),),
                after=(_topic_entry(1, "after"),),
            )
    finally:
        await server.close()

    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_model", "finish_reason"),
    (("unapproved-model", "stop"), (APPROVED_TOPIC_MODEL, "length")),
)
async def test_deepseek_topic_confirmation_rejects_wrong_or_truncated_envelope(
    response_model: str,
    finish_reason: str,
) -> None:
    server, _, _ = await _server(
        {"confirmed": True, "confidence": 0.9},
        response_model=response_model,
        finish_reason=finish_reason,
    )
    try:
        provider = _provider(server)
        controller = _bind_provider(provider, boundary_entry_id="entry-1")
        with pytest.raises(SummaryProviderError):
            await provider.confirm_topic_boundary(
                candidate=TopicBoundaryCandidate("entry-1", ("transition_phrase",), 0.55),
                before=(_topic_entry(0, "before"),),
                after=(_topic_entry(1, "after"),),
            )
    finally:
        await server.close()

    boundary_actions = [
        action for action in controller.journal.actions if action.kind.value == "boundary_chat"
    ]
    assert len(boundary_actions) == 1
    assert boundary_actions[0].completed


@pytest.mark.asyncio
async def test_deepseek_topic_confirmation_rejects_fenced_json_without_repair() -> None:
    server, requests, _ = await _server({"confirmed": True, "confidence": 0.9}, fenced=True)
    try:
        provider = _provider(server)
        _bind_provider(provider, boundary_entry_id="entry-1")
        with pytest.raises(SummaryProviderError):
            await provider.confirm_topic_boundary(
                candidate=TopicBoundaryCandidate("entry-1", ("transition_phrase",), 0.55),
                before=(_topic_entry(0, "before"),),
                after=(_topic_entry(1, "after"),),
            )
    finally:
        await server.close()

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_deepseek_inventory_mismatch_fails_before_chat() -> None:
    server, requests, _ = await _server(
        {"confirmed": True, "confidence": 0.9}, inventory_model="not-approved"
    )
    try:
        provider = _provider(server)
        _bind_provider(provider, boundary_entry_id="entry-1")
        with pytest.raises(SummaryProviderError):
            await provider.confirm_topic_boundary(
                candidate=TopicBoundaryCandidate("entry-1", ("transition_phrase",), 0.55),
                before=(_topic_entry(0, "before"),),
                after=(_topic_entry(1, "after"),),
            )
    finally:
        await server.close()

    assert requests == []
