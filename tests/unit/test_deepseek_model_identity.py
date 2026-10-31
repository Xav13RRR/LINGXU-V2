import hashlib

import pytest

from lingxu.worker.full_transcript_preview import (
    DeepSeekFullTranscriptPreviewClient,
    FullTranscriptPreviewError,
    build_preview_request,
)
from lingxu.worker.outcome_partitions import OutcomePartitionManifest, OutcomePartitionResult
from lingxu.worker.topics import APPROVED_TOPIC_MODEL, APPROVED_TOPIC_MODEL_DIGEST


def test_v41_identity_keeps_the_final_request_parameters() -> None:
    messages = [{"role": "user", "content": "synthetic json request"}]
    assert build_preview_request(messages, output_format="json_object") == {
        "model": "deepseek-flash",
        "messages": messages,
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "max_tokens": 131_072,
        "response_format": {"type": "json_object"},
    }
    assert APPROVED_TOPIC_MODEL == "deepseek-flash"
    contract = (
        "lingxu-deepseek-flash-v4.1-contract-v5-initial768-nonthinking-"
        "repair32768-thinking-high-json-object-output-limit-two-repairs"
    )
    assert APPROVED_TOPIC_MODEL_DIGEST == hashlib.sha256(contract.encode()).hexdigest()


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4.1-flash"])
def test_retired_and_invented_model_ids_cannot_authorize_calls(model: str) -> None:
    with pytest.raises(FullTranscriptPreviewError, match="approved_model_policy_failed"):
        DeepSeekFullTranscriptPreviewClient(api_key="synthetic-credential", model=model)


def test_old_policy_digest_cannot_authorize_new_model_calls() -> None:
    with pytest.raises(FullTranscriptPreviewError, match="approved_model_policy_failed"):
        DeepSeekFullTranscriptPreviewClient(
            api_key="synthetic-credential",
            expected_model_digest=(
                "6eacea9c9db049e78f65c1a6d2f3df64e12cc3fe82b7292dad5052ae28320d86"
            ),
        )


@pytest.mark.parametrize("model", [OutcomePartitionManifest, OutcomePartitionResult])
def test_retained_metadata_can_still_decode_historical_model_labels(model: type) -> None:
    assert model.model_json_schema()["properties"]["model"]["enum"] == [
        "qwen3.5:9b",
        "deepseek-v4-flash",
        "deepseek-flash",
    ]
