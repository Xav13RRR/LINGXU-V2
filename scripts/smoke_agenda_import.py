"""Content-safe real local-model smoke for the selected M1+P2 agenda importer."""

from __future__ import annotations

import asyncio
import hashlib
import json

from lingxu.config import Settings
from lingxu.domain.enums import AgendaFieldOrigin
from lingxu.meeting.agenda_import import OllamaAgendaImportProvider, build_source_blocks


async def _run() -> dict[str, object]:
    settings = Settings()
    api_key = (
        settings.agenda_import_llm_api_key.get_secret_value()
        if settings.agenda_import_llm_api_key is not None
        else None
    )
    provider = OllamaAgendaImportProvider(
        base_url=settings.agenda_import_llm_base_url,
        model=settings.agenda_import_llm_model,
        expected_model_digest=settings.agenda_import_llm_model_digest,
        api_key=api_key,
        timeout_seconds=settings.agenda_import_llm_timeout_seconds,
        max_output_tokens=settings.agenda_import_llm_max_output_tokens,
    )
    blocks = build_source_blocks(
        """# 产品例会
## 状态同步
开场先快速同步上周进度和阻塞项。

| 环节 | 说明 | 时间 |
| --- | --- | --- |
| 发布计划 | 讨论发布窗口和回滚安排 | 20m |

## 发布决定
最后请表决是否批准本轮发布计划。"""
    )
    result = await provider.extract(
        blocks=blocks,
        default_duration_minutes=settings.agenda_import_default_duration_minutes,
        default_vote_seconds=settings.agenda_import_default_vote_seconds,
    )
    if result.blocking_issues or not 1 <= len(result.items) <= 6:
        raise RuntimeError("real agenda import smoke returned no usable agenda")
    allowed_ids = {block.id for block in blocks}
    if any(not set(item.source_block_ids) <= allowed_ids for item in result.items):
        raise RuntimeError("real agenda import smoke returned an unknown source")
    if not any(item.requires_vote for item in result.items):
        raise RuntimeError("real agenda import smoke missed explicit vote semantics")
    if not any(item.duration_origin is AgendaFieldOrigin.DEFAULTED for item in result.items):
        raise RuntimeError("real agenda import smoke missed the fixed duration default")
    identity = json.dumps(
        [
            {
                "title": item.title,
                "sources": item.source_block_ids,
                "duration": item.duration_minutes,
                "vote": item.requires_vote,
            }
            for item in result.items
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "ok": True,
        "item_count": len(result.items),
        "vote_count": sum(item.requires_vote for item in result.items),
        "defaulted_duration_count": sum(
            item.duration_origin is AgendaFieldOrigin.DEFAULTED for item in result.items
        ),
        "result_sha256": hashlib.sha256(identity).hexdigest(),
    }


def main() -> None:
    print(json.dumps(asyncio.run(_run()), separators=(",", ":")))


if __name__ == "__main__":
    main()
