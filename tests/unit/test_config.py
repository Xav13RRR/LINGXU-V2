import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lingxu.config import Settings

_ROOT = Path(__file__).parents[2]


def test_independent_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.database_url.endswith("lingxu.db")
    assert settings.discord_token is None
    assert settings.has_discord_credentials is False


def test_blank_optional_compose_values_are_absent() -> None:
    settings = Settings(
        _env_file=None,
        discord_token="",
        discord_application_id="",
        discord_test_guild_id="",
    )

    assert settings.discord_token is None
    assert settings.discord_application_id is None
    assert settings.discord_test_guild_id is None


def test_local_llm_defaults_are_disabled_and_explicit() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_intents_enabled is False
    assert settings.llm_base_url == "http://lingxu-ollama:11434/v1"
    assert settings.llm_model == "qwen3.5:9b"
    assert settings.llm_api_key is None
    assert settings.summary_llm_enabled is False
    assert settings.summary_llm_model == "deepseek-flash"
    assert settings.summary_llm_api_key is None
    assert settings.summary_llm_timeout_seconds == 1_200
    assert settings.summary_llm_max_output_tokens == 32_768
    assert settings.summary_final_minutes_max_output_tokens == 131_072
    assert settings.meeting_assistant_enabled is False
    assert settings.meeting_assistant_wake_phrase == "呼叫指挥中心"
    assert settings.meeting_assistant_kws_window_ms == 4_000
    assert settings.meeting_assistant_kws_hop_ms == 500
    assert settings.meeting_assistant_kws_min_window_ms == 1_200
    assert settings.meeting_assistant_kws_preroll_ms == 300
    assert settings.meeting_assistant_queue_max_utterances == 64
    assert settings.meeting_assistant_llm_model == "qwen3.5:9b"
    assert settings.resident_assistant_enabled is False
    assert settings.resident_assistant_recover_on_restart is True
    assert settings.resident_assistant_max_sessions == 1
    assert settings.agenda_import_llm_enabled is False
    assert settings.agenda_import_llm_model == "qwen3.5:9b"
    assert settings.agenda_import_default_duration_minutes == 15
    assert settings.agenda_import_default_vote_seconds == 60


def test_post_meeting_inference_has_one_fixed_deepseek_path() -> None:
    from lingxu.worker.topics import (
        APPROVED_TOPIC_BASE_URL,
        APPROVED_TOPIC_MODEL,
        APPROVED_TOPIC_MODEL_DIGEST,
    )

    settings = Settings(_env_file=None)
    worker_main = (_ROOT / "src" / "lingxu" / "worker_main.py").read_text(encoding="utf-8")
    preview_script = (_ROOT / "scripts" / "preview_full_transcript_minutes.py").read_text(
        encoding="utf-8"
    )
    provider_source = (_ROOT / "src" / "lingxu" / "worker" / "summary.py").read_text(
        encoding="utf-8"
    )
    compose = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    golden = (_ROOT / "docker-compose.golden.yml").read_text(encoding="utf-8")

    assert "summary_llm_provider" not in Settings.model_fields
    assert settings.summary_llm_base_url == APPROVED_TOPIC_BASE_URL
    assert settings.summary_llm_model == APPROVED_TOPIC_MODEL
    assert settings.summary_llm_model_digest == APPROVED_TOPIC_MODEL_DIGEST
    assert settings.summary_llm_allow_remote is True
    assert "OllamaSummaryProvider" not in worker_main
    assert "OllamaSummaryProvider" not in provider_source
    assert "DeepSeekPostMeetingInferenceProvider" in worker_main
    assert "max_output_tokens=settings.summary_final_minutes_max_output_tokens" in worker_main
    assert "max_output_tokens=settings.summary_final_minutes_max_output_tokens" in preview_script
    assert "max_tokens_sent" not in preview_script
    assert "the fixed DeepSeek post-meeting path must be enabled" in worker_main
    assert "if settings.summary_llm_enabled" not in worker_main
    assert "SUMMARY_LLM_PROVIDER" not in compose
    assert "SUMMARY_LLM_PROVIDER" not in golden
    assert (
        "SUMMARY_FINAL_MINUTES_MAX_OUTPUT_TOKENS: "
        "${SUMMARY_FINAL_MINUTES_MAX_OUTPUT_TOKENS:-131072}" in compose
    )


def test_enabled_local_llm_rejects_remote_endpoint_and_bad_digest() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_intents_enabled=True,
            llm_base_url="https://example.com/v1",
        )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_model_digest="not-a-sha256")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_intents_enabled=True)
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            summary_llm_enabled=True,
            summary_llm_base_url="https://example.com/v1",
        )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, summary_llm_model_digest="not-a-sha256")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, summary_llm_enabled=True)
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            meeting_assistant_enabled=True,
            meeting_assistant_llm_base_url="https://example.com/v1",
        )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, meeting_assistant_enabled=True)
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            resident_assistant_enabled=True,
            meeting_assistant_llm_base_url="https://example.com/v1",
            meeting_assistant_llm_model_digest="a" * 64,
        )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, resident_assistant_enabled=True)
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            agenda_import_llm_enabled=True,
            agenda_import_llm_base_url="https://example.com/v1",
        )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agenda_import_llm_model_digest="not-a-sha256")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agenda_import_llm_enabled=True)


def test_deepseek_summary_provider_requires_exact_remote_policy_and_redacts_key() -> None:
    from lingxu.worker.topics import (
        APPROVED_TOPIC_BASE_URL,
        APPROVED_TOPIC_MODEL,
        APPROVED_TOPIC_MODEL_DIGEST,
    )

    sentinel = "PRIVATE_DEEPSEEK_TEST_KEY"
    settings = Settings(
        _env_file=None,
        summary_llm_enabled=True,
        summary_llm_base_url=APPROVED_TOPIC_BASE_URL,
        summary_llm_api_key=sentinel,
        summary_llm_model=APPROVED_TOPIC_MODEL,
        summary_llm_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
        summary_llm_allow_remote=True,
    )

    assert sentinel in settings.configured_secret_values
    assert sentinel not in repr(settings)
    assert settings.summary_llm_timeout_seconds == 1_200
    for override in (
        {"summary_llm_api_key": ""},
        {"summary_llm_allow_remote": False},
        {"summary_llm_base_url": "https://example.com"},
        {"summary_llm_model": "deepseek-v4-pro"},
        {"summary_llm_model": "deepseek-v4-flash"},
        {"summary_llm_model": "deepseek-v4.1-flash"},
        {"summary_llm_model_digest": "a" * 64},
    ):
        with pytest.raises(ValidationError):
            values = {
                "summary_llm_enabled": True,
                "summary_llm_base_url": APPROVED_TOPIC_BASE_URL,
                "summary_llm_api_key": sentinel,
                "summary_llm_model": APPROVED_TOPIC_MODEL,
                "summary_llm_model_digest": APPROVED_TOPIC_MODEL_DIGEST,
                "summary_llm_allow_remote": True,
                **override,
            }
            Settings(
                _env_file=None,
                **values,
            )

    with pytest.raises(ValidationError):
        Settings(_env_file=None, summary_llm_timeout_seconds=1_200.01)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, summary_final_minutes_max_output_tokens=384_001)


def test_assistant_streaming_window_bounds_are_consistent() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            meeting_assistant_kws_window_ms=1_000,
            meeting_assistant_kws_min_window_ms=1_200,
        )
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            meeting_assistant_kws_window_ms=1_000,
            meeting_assistant_kws_hop_ms=1_100,
        )
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            meeting_assistant_kws_window_ms=1_000,
            meeting_assistant_kws_preroll_ms=1_100,
        )


def test_cosyvoice_rejects_remote_endpoint_and_bad_pins() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            tts_provider="cosyvoice-sft",
            tts_base_url="https://example.com",
        )
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tts_model_revision="latest")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tts_model_digest="not-a-sha256")


def test_host_console_is_disabled_by_default_and_derives_a_separate_token() -> None:
    disabled = Settings(_env_file=None)
    assert disabled.host_console_enabled is False
    assert disabled.effective_host_console_token is None

    enabled = Settings(
        _env_file=None,
        host_console_enabled=True,
        discord_test_guild_id=42,
        file_encryption_key="root-secret-that-is-long-enough-for-tests",
    )
    assert enabled.effective_host_console_guild_id == 42
    assert enabled.effective_host_console_token is not None
    assert enabled.effective_host_console_token != "root-secret-that-is-long-enough-for-tests"
    assert len(enabled.effective_host_console_token) == 64


def test_enabled_host_console_requires_one_guild_and_secret_root() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            host_console_enabled=True,
            host_console_token="x" * 32,
        )
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            host_console_enabled=True,
            host_console_guild_id=42,
        )


def test_settings_env_example_and_compose_keys_stay_in_sync() -> None:
    settings_keys = {name.upper() for name in Settings.model_fields}
    example_keys = {
        match.group(1)
        for line in (_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if (match := re.fullmatch(r"([A-Z][A-Z0-9_]*)=.*", line)) is not None
    }
    compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    common = compose["x-lingxu-common"]
    services = compose["services"]
    bot = services["lingxu-bot"]
    worker = services["lingxu-worker"]
    migration = services["lingxu-migrate"]
    bot_keys = set(bot["environment"])
    worker_keys = set(worker["environment"])
    configured_secret_keys = {
        "DISCORD_TOKEN",
        "FILE_ENCRYPTION_KEY",
        "LLM_API_KEY",
        "SUMMARY_LLM_API_KEY",
        "MEETING_ASSISTANT_LLM_API_KEY",
        "AGENDA_IMPORT_LLM_API_KEY",
        "HOST_CONSOLE_TOKEN",
    }

    assert example_keys == settings_keys
    assert bot_keys | worker_keys == settings_keys | {"MODELSCOPE_CACHE"}
    assert common["environment"]["GLOSSARY_ROOT"] == "/data/glossaries"
    assert configured_secret_keys.isdisjoint(common["environment"])
    assert configured_secret_keys & bot_keys == {
        "DISCORD_TOKEN",
        "FILE_ENCRYPTION_KEY",
        "LLM_API_KEY",
        "MEETING_ASSISTANT_LLM_API_KEY",
        "AGENDA_IMPORT_LLM_API_KEY",
        "HOST_CONSOLE_TOKEN",
    }
    assert configured_secret_keys & worker_keys == {
        "DISCORD_TOKEN",
        "FILE_ENCRYPTION_KEY",
        "SUMMARY_LLM_API_KEY",
    }
    assert configured_secret_keys.isdisjoint(migration["environment"])
    assert "SUMMARY_LLM_ENABLED" not in bot["environment"]
    assert worker["environment"]["SUMMARY_LLM_ENABLED"] == "${SUMMARY_LLM_ENABLED:-true}"
    assert set(migration["environment"]) == {"DATABASE_URL"}
    assert worker["networks"] == ["default"]
    assert {
        "MEETING_ASSISTANT_ENABLED",
        "RESIDENT_ASSISTANT_ENABLED",
        "LLM_INTENTS_ENABLED",
        "AGENDA_IMPORT_LLM_ENABLED",
        "HOST_CONSOLE_ENABLED",
        "TTS_PROVIDER",
        "VOICE_DUPLEX_OUTPUT_DIR",
    }.isdisjoint(worker["environment"])
    assert bot["depends_on"]["lingxu-migrate"]["condition"] == "service_completed_successfully"
    assert worker["depends_on"]["lingxu-migrate"]["condition"] == "service_completed_successfully"


def test_application_processes_share_one_runtime_image_and_build_owner() -> None:
    compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    migration = services["lingxu-migrate"]
    bot = services["lingxu-bot"]
    worker = services["lingxu-worker"]

    assert migration["image"] == bot["image"] == worker["image"]
    assert migration["image"] == "${LINGXU_APP_IMAGE:-lingxu-app}:${LINGXU_APP_IMAGE_TAG:-latest}"
    assert migration["build"]["target"] == "runtime"
    assert "build" not in bot
    assert "build" not in worker

    tts = services["lingxu-tts"]
    assert tts["image"] == "${LINGXU_TTS_IMAGE:-lingxu-lingxu-tts}:${LINGXU_TTS_IMAGE_TAG:-latest}"
    assert tts["build"]["args"]["TTS_BUILD_FINGERPRINT"] == (
        "${LINGXU_TTS_BUILD_FINGERPRINT:-unknown}"
    )


def test_golden_preview_uses_the_production_summary_timeout() -> None:
    golden = yaml.safe_load((_ROOT / "docker-compose.golden.yml").read_text(encoding="utf-8"))

    preview = golden["services"]["lingxu-full-transcript-preview"]

    assert preview["environment"]["SUMMARY_LLM_TIMEOUT_SECONDS"] == "1200"


def test_completed_gate_only_runtime_and_annotation_surfaces_are_retired() -> None:
    retired_paths = (
        "src/lingxu/bot/commands.py",
        "src/lingxu/bot/probe.py",
        "src/lingxu/cli/validate_duplex.py",
        "scripts/voice_duplex_probe.py",
        "scripts/evaluate_private_golden.py",
        "docs/TASK13_GOLDEN_ANNOTATION_GUIDE.md",
    )
    assert not [path for path in retired_paths if (_ROOT / path).exists()]

    client = (_ROOT / "src" / "lingxu" / "bot" / "client.py").read_text(encoding="utf-8")
    golden = yaml.safe_load((_ROOT / "docker-compose.golden.yml").read_text(encoding="utf-8"))
    assert "ProbeCommands" not in client
    assert "ProbeRegistry" not in client
    assert set(golden["services"]) == {"lingxu-full-transcript-preview"}
    assert "--prepare-annotations" not in (_ROOT / "docker-compose.golden.yml").read_text(
        encoding="utf-8"
    )


def test_runtime_image_uses_an_explicit_operational_script_allowlist() -> None:
    dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY scripts ./scripts" not in dockerfile
    assert "FROM runtime-base AS tools" in dockerfile
    assert "FROM runtime-base AS runtime" in dockerfile
    assert "scripts/check_deployment_idle.py" in dockerfile
    assert "scripts/download_asr_models.py" in dockerfile
    assert "scripts/download_wakeword_model.py" in dockerfile
    assert "scripts/preview_full_transcript_minutes.py" in dockerfile
    for retired in (
        "smoke_cosyvoice_tts.py",
        "smoke_deepseek_api.py",
        "smoke_meeting_components.py",
        "smoke_scenario_privacy.py",
        "evaluate_private_golden.py",
        "voice_duplex_probe.py",
        "validate_duplex_probe.py",
    ):
        assert retired not in dockerfile


def test_asr_downloader_is_explicit_and_runtime_services_do_not_use_it() -> None:
    compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    downloader = services["lingxu-asr-downloader"]

    assert downloader["profiles"] == ["model-download"]
    assert downloader["restart"] == "no"
    assert downloader["read_only"] is True
    assert downloader["command"] == ["python", "/app/scripts/download_asr_models.py"]
    assert downloader["volumes"] == ["lingxu-models:/models"]
    assert "DISCORD_TOKEN" not in downloader["environment"]
    assert "SUMMARY_LLM_API_KEY" not in downloader["environment"]
    assert "/app/scripts/download_asr_models.py" not in services["lingxu-bot"]["command"]
    assert "/app/scripts/download_asr_models.py" not in services["lingxu-worker"]["command"]
