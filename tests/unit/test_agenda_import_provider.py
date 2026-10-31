import json
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from lingxu.domain.enums import AgendaFieldOrigin
from lingxu.domain.errors import AgendaImportProviderError
from lingxu.meeting.agenda_import import (
    AgendaSourceBlock,
    OllamaAgendaImportProvider,
    agenda_import_json_schema,
    build_source_blocks,
    chunk_source_blocks,
    decode_markdown,
)


def _model_item(**changes: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "title": "预算审批",
        "title_origin": "INFERRED",
        "source_block_ids": ["B0001"],
        "duration_minutes": 99,
        "duration_origin": "DEFAULTED",
        "requires_vote": True,
        "vote_origin": "INFERRED",
        "vote_content": "是否批准预算?",
        "vote_content_origin": "INFERRED",
        "vote_duration_seconds": 999,
        "vote_duration_origin": "DEFAULTED",
        "language": "zh",
        "language_origin": "INFERRED",
        "confidence": 0.93,
        "completion_notes": ["根据审批语义整理为是/否问题"],
    }
    item.update(changes)
    return item


async def _ollama_server(
    content: dict[str, Any],
) -> tuple[TestServer, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    async def tags(_request: web.Request) -> web.Response:
        return web.json_response({"models": [{"name": "qwen3.5:9b", "digest": "a" * 64}]})

    async def chat(request: web.Request) -> web.Response:
        requests.append(await request.json())
        return web.json_response({"message": {"content": json.dumps(content)}})

    app = web.Application()
    app.router.add_get("/api/tags", tags)
    app.router.add_post("/api/chat", chat)
    server = TestServer(app)
    await server.start_server()
    return server, requests


def test_markdown_decoding_and_bounded_overlapping_blocks() -> None:
    text = "# 会议\n\n- 预算审批\n- 发布计划"
    assert decode_markdown(text.encode("utf-16")) == text
    assert decode_markdown(text.encode("gb18030")) == text
    blocks = build_source_blocks(text)
    chunks = chunk_source_blocks(blocks, max_chunk_chars=45)

    assert [block.id for block in blocks] == ["B0001", "B0002", "B0003"]
    assert len(chunks) >= 2
    assert chunks[0][-1].id == chunks[1][0].id


def test_agenda_import_schema_is_closed_and_requires_provenance() -> None:
    schema = agenda_import_json_schema()

    assert schema["additionalProperties"] is False
    item_schema = schema["$defs"]["_ModelAgendaItem"]
    assert item_schema["additionalProperties"] is False
    assert "title_origin" in item_schema["required"]
    assert "source_block_ids" in item_schema["required"]


@pytest.mark.asyncio
async def test_local_provider_uses_native_schema_and_enforces_fixed_defaults() -> None:
    server, requests = await _ollama_server({"items": [_model_item()], "blocking_issues": []})
    try:
        provider = OllamaAgendaImportProvider(
            base_url=str(server.make_url("/v1")).rstrip("/"),
            model="qwen3.5:9b",
            expected_model_digest="a" * 64,
        )
        result = await provider.extract(
            blocks=(AgendaSourceBlock("B0001", "审批本季度预算并进行表决"),),
            default_duration_minutes=15,
            default_vote_seconds=60,
        )
    finally:
        await server.close()

    assert len(result.items) == 1
    item = result.items[0]
    assert item.duration_minutes == 15
    assert item.duration_origin is AgendaFieldOrigin.DEFAULTED
    assert item.vote_duration_seconds == 60
    assert item.vote_duration_origin is AgendaFieldOrigin.DEFAULTED
    payload = requests[0]
    assert payload["model"] == "qwen3.5:9b"
    assert payload["think"] is False
    assert payload["format"]["additionalProperties"] is False
    assert json.loads(payload["messages"][1]["content"])["blocks"][0]["id"] == "B0001"


@pytest.mark.asyncio
async def test_provider_rejects_unknown_source_reference() -> None:
    server, _ = await _ollama_server(
        {
            "items": [_model_item(source_block_ids=["B9999"])],
            "blocking_issues": [],
        }
    )
    try:
        provider = OllamaAgendaImportProvider(
            base_url=str(server.make_url("/v1")).rstrip("/"),
            model="qwen3.5:9b",
            expected_model_digest="a" * 64,
        )
        with pytest.raises(AgendaImportProviderError):
            await provider.extract(
                blocks=(AgendaSourceBlock("B0001", "预算"),),
                default_duration_minutes=15,
                default_vote_seconds=60,
            )
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_provider_accepts_null_vote_origin_for_non_voting_item() -> None:
    server, _ = await _ollama_server(
        {
            "items": [
                _model_item(
                    requires_vote=False,
                    vote_origin=None,
                    vote_content=None,
                    vote_content_origin=None,
                    vote_duration_seconds=None,
                    vote_duration_origin=None,
                )
            ],
            "blocking_issues": [],
        }
    )
    try:
        provider = OllamaAgendaImportProvider(
            base_url=str(server.make_url("/v1")).rstrip("/"),
            model="qwen3.5:9b",
            expected_model_digest="a" * 64,
        )
        result = await provider.extract(
            blocks=(AgendaSourceBlock("B0001", "项目进度同步"),),
            default_duration_minutes=15,
            default_vote_seconds=60,
        )
    finally:
        await server.close()

    assert result.items[0].requires_vote is False
    assert result.items[0].vote_origin is AgendaFieldOrigin.INFERRED
