"""Validated, independent Lingxu environment configuration."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and an optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    discord_token: SecretStr | None = None
    discord_application_id: int | None = None
    discord_test_guild_id: int | None = None

    database_url: str = "sqlite+aiosqlite:///data/lingxu.db"
    data_root: Path = Path("data/sessions")
    glossary_root: Path = Path("data/glossaries")
    runtime_tmp: Path = Path(".runtime")
    model_cache_dir: Path = Path("models")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    voice_duplex_dave_ready_timeout_seconds: float = Field(default=15.0, gt=0)

    file_encryption_key: SecretStr | None = None
    worker_poll_interval_seconds: float = Field(default=2.0, gt=0)
    worker_job_lock_seconds: float = Field(default=600.0, gt=0)
    worker_job_max_attempts: int = Field(default=3, ge=1)
    require_realtime_asr_consent: bool = True
    realtime_asr_provider: Literal["sensevoice"] = "sensevoice"
    realtime_asr_model: str = "iic/SenseVoiceSmall"
    realtime_asr_model_revision: str = "master"
    realtime_asr_model_sha256: str = (
        "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
    )
    realtime_asr_vad_model: str = "fsmn-vad"
    realtime_asr_vad_model_revision: str = "v2.0.4"
    realtime_asr_vad_model_sha256: str = (
        "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5"
    )
    realtime_asr_language_mode: str = "auto"
    realtime_asr_hotwords: str = ""
    realtime_asr_timeout_seconds: float = Field(default=8.0, gt=0)
    realtime_asr_max_concurrency: int = Field(default=1, ge=1, le=8)
    realtime_asr_queue_max_frames_per_user: int = Field(default=500, ge=1)
    realtime_asr_queue_max_utterances: int = Field(default=32, ge=1)
    realtime_asr_vad_threshold: int = Field(default=500, ge=0, le=32_767)
    realtime_asr_vad_start_ms: int = Field(default=60, ge=20)
    realtime_asr_vad_end_ms: int = Field(default=800, ge=20)
    realtime_asr_min_utterance_ms: int = Field(default=300, ge=20)
    realtime_asr_max_utterance_seconds: float = Field(default=30.0, gt=0)
    meeting_assistant_enabled: bool = False
    meeting_assistant_wake_phrase: Literal["呼叫指挥中心"] = "呼叫指挥中心"
    meeting_assistant_wakeword_model_dir: Path = Path(
        "models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
    )
    meeting_assistant_wakeword_model_digest: str = (
        "9604d8412a76c099ef8eecedb95fa04d523cb16df2f6a6be724bc956a92870ef"
    )
    meeting_assistant_wakeword_score: float = Field(default=1.5, gt=0.0, le=10.0)
    meeting_assistant_wakeword_threshold: float = Field(
        default=0.18,
        gt=0.0,
        le=1.0,
    )
    meeting_assistant_queue_max_frames_per_user: int = Field(default=500, ge=1)
    meeting_assistant_queue_max_utterances: int = Field(default=64, ge=1, le=64)
    meeting_assistant_kws_window_ms: int = Field(default=4_000, ge=500, le=10_000)
    meeting_assistant_kws_hop_ms: int = Field(default=500, ge=100, le=5_000)
    meeting_assistant_kws_min_window_ms: int = Field(default=1_200, ge=200, le=5_000)
    meeting_assistant_kws_preroll_ms: int = Field(default=300, ge=20, le=2_000)
    meeting_assistant_vad_threshold: int = Field(default=500, ge=0, le=32_767)
    meeting_assistant_vad_start_ms: int = Field(default=60, ge=20)
    meeting_assistant_vad_end_ms: int = Field(default=800, ge=20)
    meeting_assistant_min_utterance_ms: int = Field(default=300, ge=20)
    meeting_assistant_max_utterance_seconds: float = Field(default=30.0, gt=0)
    meeting_assistant_followup_seconds: float = Field(default=4.0, gt=0.0, le=15.0)
    meeting_assistant_quarantine_ms: int = Field(default=600, ge=0, le=5_000)
    meeting_assistant_asr_timeout_seconds: float = Field(default=8.0, gt=0.0, le=60.0)
    meeting_assistant_llm_base_url: str = "http://lingxu-ollama:11434/v1"
    meeting_assistant_llm_api_key: SecretStr | None = None
    meeting_assistant_llm_model: Literal["qwen3.5:9b"] = "qwen3.5:9b"
    meeting_assistant_llm_model_digest: str = ""
    meeting_assistant_llm_timeout_seconds: float = Field(default=30.0, gt=0.0, le=180.0)
    meeting_assistant_llm_max_output_tokens: int = Field(default=64, ge=16, le=128)
    meeting_assistant_max_question_chars: int = Field(default=500, ge=1, le=2_000)
    meeting_assistant_llm_allow_remote: bool = False
    resident_assistant_enabled: bool = False
    resident_assistant_recover_on_restart: bool = True
    resident_assistant_max_sessions: int = Field(default=1, ge=1, le=8)
    tts_provider: Literal["espeak-ng", "cosyvoice-sft"] = "espeak-ng"
    tts_binary: str = "espeak-ng"
    tts_voice: str = "cmn"
    tts_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    tts_timeout_seconds: float = Field(default=30.0, gt=0)
    tts_cache_dir: Path = Path("data/tts_cache")
    tts_base_url: str = "http://lingxu-tts:50000"
    tts_model: str = "FunAudioLLM/CosyVoice-300M-SFT"
    tts_model_revision: str = "fbb71de2afe387ed854eebd80b9f3d078c6b9869"
    tts_model_digest: str = "3a80af814cfd832dadbf1fe677adf08e30f64b8bdcd52b997184515fb33d2198"
    tts_playback_volume: float = Field(default=0.8, ge=0.0, le=2.0)
    tts_echo_debounce_ms: int = Field(default=350, ge=200, le=500)
    meeting_event_snapshot_interval: int = Field(default=50, ge=1)
    meeting_recovery_enabled: bool = True
    rule_intents_enabled: bool = True
    llm_intents_enabled: bool = False
    llm_base_url: str = "http://lingxu-ollama:11434/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "qwen3.5:9b"
    llm_model_digest: str = ""
    llm_intent_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    llm_timeout_seconds: float = Field(default=180.0, gt=0.0, le=600.0)
    llm_max_output_tokens: int = Field(default=64, ge=16, le=256)
    llm_max_utterance_chars: int = Field(default=1_000, ge=1, le=4_000)
    llm_allow_remote: bool = False
    agenda_import_llm_enabled: bool = False
    agenda_import_llm_base_url: str = "http://lingxu-ollama:11434/v1"
    agenda_import_llm_api_key: SecretStr | None = None
    agenda_import_llm_model: Literal["qwen3.5:9b"] = "qwen3.5:9b"
    agenda_import_llm_model_digest: str = ""
    agenda_import_llm_timeout_seconds: float = Field(default=180.0, gt=0.0, le=600.0)
    agenda_import_llm_max_output_tokens: int = Field(default=1_536, ge=256, le=2_048)
    agenda_import_max_attachment_bytes: int = Field(default=65_536, ge=1_024, le=1_048_576)
    agenda_import_max_source_chars: int = Field(default=30_000, ge=1_000, le=100_000)
    agenda_import_max_blocks: int = Field(default=400, ge=1, le=1_000)
    agenda_import_max_chunk_chars: int = Field(default=2_500, ge=500, le=8_000)
    agenda_import_max_chunks: int = Field(default=12, ge=1, le=40)
    agenda_import_max_items: int = Field(default=50, ge=1, le=100)
    agenda_import_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    agenda_import_default_duration_minutes: int = Field(default=15, ge=1, le=480)
    agenda_import_default_vote_seconds: int = Field(default=60, ge=10, le=3_600)
    agenda_import_llm_allow_remote: bool = False
    summary_llm_enabled: bool = False
    summary_llm_base_url: str = "https://api.deepseek.com"
    summary_llm_api_key: SecretStr | None = None
    summary_llm_model: Literal["deepseek-flash"] = "deepseek-flash"
    summary_llm_model_digest: str = (
        "4a87243abc84b3147f3e9c2fc3416e539ae1ebd634706612996fc62018d32a0f"
    )
    summary_llm_timeout_seconds: float = Field(default=1_200.0, gt=0.0, le=1_200.0)
    summary_llm_max_output_tokens: int = Field(default=32_768, ge=128, le=384_000)
    summary_final_minutes_max_output_tokens: int = Field(
        default=131_072,
        ge=128,
        le=384_000,
    )
    summary_llm_max_input_chars: int = Field(default=6_000, ge=1_000, le=20_000)
    summary_llm_allow_remote: bool = True
    host_console_enabled: bool = False
    host_console_bind_host: str = "127.0.0.1"
    host_console_port: int = Field(default=8_765, ge=1, le=65_535)
    host_console_token: SecretStr | None = None
    host_console_guild_id: int | None = None
    default_speaker_seconds: int = Field(default=180, ge=10)
    default_vote_seconds: int = Field(default=60, ge=10)
    default_discussion_seconds: int = Field(default=900, ge=10)
    require_recording_consent: bool = True
    recording_audio_queue_max_frames: int = Field(default=5_000, ge=1)
    recording_silence_split_ms: int = Field(default=800, ge=20)
    recording_silence_threshold: int = Field(default=256, ge=0, le=32_767)
    recording_max_segment_seconds: float = Field(default=300.0, gt=0)
    audio_retention_hours: int = Field(default=24, ge=1)

    @field_validator(
        "discord_application_id",
        "discord_test_guild_id",
        "host_console_guild_id",
        mode="before",
    )
    @classmethod
    def blank_optional_values_are_absent(cls, value: Any) -> Any:
        """Allow optional Compose values to be present as empty strings."""

        return None if value == "" else value

    @field_validator(
        "discord_token",
        "llm_api_key",
        "summary_llm_api_key",
        "meeting_assistant_llm_api_key",
        "agenda_import_llm_api_key",
        "host_console_token",
        mode="before",
    )
    @classmethod
    def blank_token_is_absent(cls, value: Any) -> Any:
        """Treat an empty API token as unconfigured."""

        return None if value == "" else value

    @field_validator("file_encryption_key", mode="before")
    @classmethod
    def blank_encryption_key_is_absent(cls, value: Any) -> Any:
        """Treat an empty encryption key as unconfigured."""

        return None if value == "" else value

    @field_validator(
        "llm_model",
        "summary_llm_model",
        "meeting_assistant_llm_model",
        "agenda_import_llm_model",
        mode="before",
    )
    @classmethod
    def llm_model_must_be_explicit(cls, value: Any) -> Any:
        """Reject a blank or floating local model selection."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("LLM_MODEL must be non-empty")
        return value.strip()

    @field_validator(
        "llm_model_digest",
        "summary_llm_model_digest",
        "meeting_assistant_llm_model_digest",
        "agenda_import_llm_model_digest",
        "meeting_assistant_wakeword_model_digest",
    )
    @classmethod
    def llm_model_digest_must_be_sha256(cls, value: str) -> str:
        """Accept an absent digest during first pull, otherwise require SHA-256."""

        selected = value.strip().lower()
        if selected and (
            len(selected) != 64
            or any(character not in "0123456789abcdef" for character in selected)
        ):
            raise ValueError("LLM model digest must be a 64-character SHA-256 digest")
        return selected

    @field_validator("tts_model_revision")
    @classmethod
    def tts_model_revision_must_be_commit(cls, value: str) -> str:
        selected = value.strip().lower()
        if len(selected) != 40 or any(
            character not in "0123456789abcdef" for character in selected
        ):
            raise ValueError("TTS_MODEL_REVISION must be a 40-character commit SHA")
        return selected

    @field_validator("tts_model_digest")
    @classmethod
    def tts_model_digest_must_be_sha256(cls, value: str) -> str:
        selected = value.strip().lower()
        if len(selected) != 64 or any(
            character not in "0123456789abcdef" for character in selected
        ):
            raise ValueError("TTS_MODEL_DIGEST must be a 64-character SHA-256 digest")
        return selected

    @model_validator(mode="after")
    def local_llm_endpoint_is_fail_closed(self) -> Settings:
        """Prevent the local deployment flag from silently sending text off-host."""

        from urllib.parse import urlparse

        parsed = urlparse(self.llm_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("LLM_BASE_URL must be an absolute HTTP(S) URL")
        local_hosts = {"127.0.0.1", "::1", "localhost", "lingxu-ollama"}
        if (
            self.llm_intents_enabled
            and not self.llm_allow_remote
            and parsed.hostname not in local_hosts
        ):
            raise ValueError(
                "remote LLM_BASE_URL is forbidden unless LLM_ALLOW_REMOTE is explicitly enabled"
            )
        if self.llm_intents_enabled and not self.llm_model_digest:
            raise ValueError("LLM_MODEL_DIGEST is required when LLM intents are enabled")
        agenda_import_parsed = urlparse(self.agenda_import_llm_base_url)
        if (
            agenda_import_parsed.scheme not in {"http", "https"}
            or not agenda_import_parsed.hostname
        ):
            raise ValueError("AGENDA_IMPORT_LLM_BASE_URL must be an absolute HTTP(S) URL")
        if (
            self.agenda_import_llm_enabled
            and not self.agenda_import_llm_allow_remote
            and agenda_import_parsed.hostname not in local_hosts
        ):
            raise ValueError(
                "remote AGENDA_IMPORT_LLM_BASE_URL is forbidden unless "
                "AGENDA_IMPORT_LLM_ALLOW_REMOTE is explicitly enabled"
            )
        if self.agenda_import_llm_enabled and not self.agenda_import_llm_model_digest:
            raise ValueError(
                "AGENDA_IMPORT_LLM_MODEL_DIGEST is required when agenda import is enabled"
            )
        summary_parsed = urlparse(self.summary_llm_base_url)
        if summary_parsed.scheme not in {"http", "https"} or not summary_parsed.hostname:
            raise ValueError("SUMMARY_LLM_BASE_URL must be an absolute HTTP(S) URL")
        if self.summary_llm_enabled:
            from lingxu.worker.topics import (
                APPROVED_TOPIC_BASE_URL,
                APPROVED_TOPIC_MODEL,
                APPROVED_TOPIC_MODEL_DIGEST,
            )

            api_key = self.summary_llm_api_key
            if (
                self.summary_llm_base_url.rstrip("/") != APPROVED_TOPIC_BASE_URL
                or self.summary_llm_model != APPROVED_TOPIC_MODEL
                or self.summary_llm_model_digest != APPROVED_TOPIC_MODEL_DIGEST
                or not self.summary_llm_allow_remote
                or api_key is None
                or not api_key.get_secret_value()
            ):
                raise ValueError("DeepSeek summary LLM policy is not exactly approved")
        assistant_parsed = urlparse(self.meeting_assistant_llm_base_url)
        if assistant_parsed.scheme not in {"http", "https"} or not assistant_parsed.hostname:
            raise ValueError("MEETING_ASSISTANT_LLM_BASE_URL must be an absolute HTTP(S) URL")
        any_voice_assistant_enabled = (
            self.meeting_assistant_enabled or self.resident_assistant_enabled
        )
        if (
            any_voice_assistant_enabled
            and not self.meeting_assistant_llm_allow_remote
            and assistant_parsed.hostname not in local_hosts
        ):
            raise ValueError(
                "remote MEETING_ASSISTANT_LLM_BASE_URL is forbidden unless "
                "MEETING_ASSISTANT_LLM_ALLOW_REMOTE is explicitly enabled"
            )
        if any_voice_assistant_enabled and not self.meeting_assistant_llm_model_digest:
            raise ValueError(
                "MEETING_ASSISTANT_LLM_MODEL_DIGEST is required when a voice assistant is enabled"
            )
        tts_parsed = urlparse(self.tts_base_url)
        if tts_parsed.scheme not in {"http", "https"} or not tts_parsed.hostname:
            raise ValueError("TTS_BASE_URL must be an absolute HTTP(S) URL")
        if self.tts_provider == "cosyvoice-sft" and tts_parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
            "lingxu-tts",
        }:
            raise ValueError("CosyVoice TTS endpoint must remain local")
        if self.host_console_enabled:
            if self.effective_host_console_guild_id is None:
                raise ValueError(
                    "HOST_CONSOLE_GUILD_ID or DISCORD_TEST_GUILD_ID is required "
                    "when the Host console is enabled"
                )
            token = self.effective_host_console_token
            if token is None or len(token) < 32:
                raise ValueError(
                    "HOST_CONSOLE_TOKEN or FILE_ENCRYPTION_KEY is required "
                    "when the Host console is enabled"
                )
        return self

    @model_validator(mode="after")
    def assistant_streaming_window_is_valid(self) -> Settings:
        """Keep rolling KWS bounds internally consistent."""

        if self.meeting_assistant_kws_min_window_ms > self.meeting_assistant_kws_window_ms:
            raise ValueError("MEETING_ASSISTANT_KWS_MIN_WINDOW_MS must not exceed the KWS window")
        if self.meeting_assistant_kws_hop_ms > self.meeting_assistant_kws_window_ms:
            raise ValueError("MEETING_ASSISTANT_KWS_HOP_MS must not exceed the KWS window")
        if self.meeting_assistant_kws_preroll_ms > self.meeting_assistant_kws_window_ms:
            raise ValueError("MEETING_ASSISTANT_KWS_PREROLL_MS must not exceed the KWS window")
        return self

    @property
    def has_discord_credentials(self) -> bool:
        """Return whether both required independent Discord credentials are configured."""

        token = self.discord_token
        return bool(token and token.get_secret_value() and self.discord_application_id)

    @property
    def realtime_asr_hotword_values(self) -> tuple[str, ...]:
        """Return de-duplicated configured hotwords without empty values."""

        return tuple(
            dict.fromkeys(
                part.strip() for part in self.realtime_asr_hotwords.split(",") if part.strip()
            )
        )

    @property
    def effective_host_console_guild_id(self) -> int | None:
        """Resolve the one Guild the local console is permitted to observe."""

        return self.host_console_guild_id or self.discord_test_guild_id

    @property
    def effective_host_console_token(self) -> str | None:
        """Return the explicit token or a domain-separated encryption-key derivative."""

        if self.host_console_token is not None:
            value = self.host_console_token.get_secret_value()
            return value if value else None
        if self.file_encryption_key is None:
            return None
        root = self.file_encryption_key.get_secret_value()
        if not root:
            return None
        return hmac.new(
            root.encode("utf-8"),
            b"lingxu-host-console-v1",
            hashlib.sha256,
        ).hexdigest()

    @property
    def configured_secret_values(self) -> tuple[str, ...]:
        """Return configured secrets solely for value-based log redaction."""

        values: list[str] = []
        for secret in (
            self.discord_token,
            self.file_encryption_key,
            self.llm_api_key,
            self.summary_llm_api_key,
            self.meeting_assistant_llm_api_key,
            self.agenda_import_llm_api_key,
            self.host_console_token,
        ):
            if secret is not None and secret.get_secret_value():
                values.append(secret.get_secret_value())
        return tuple(values)
