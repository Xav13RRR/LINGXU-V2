# ruff: noqa: RUF001
"""Expected Lingxu domain failures."""

from typing import ClassVar


class LingxuError(Exception):
    """Base class for expected Lingxu failures."""


class VoiceModeConflictError(LingxuError):
    """Raised when a Guild already owns a different active voice task."""


class ServiceCapacityFullError(LingxuError):
    """Raised without disclosing which Guild owns the deployment slot."""

    def __init__(self) -> None:
        super().__init__("服务容量已满。聆序目前正在其它 Discord 服务器提供服务，请稍后再试。")


class ResidentAssistantNotActiveError(LingxuError):
    """Raised when no resident voice assistant is active for a Guild."""


class ResidentAssistantPermissionError(LingxuError):
    """Raised when a caller cannot start or stop the resident assistant."""


class InvalidStateTransitionError(LingxuError):
    """Raised when a persisted Meeting Session transition is not allowed."""


class SessionNotFoundError(LingxuError):
    """Raised when a persisted Meeting Session does not exist."""


class SessionAlreadyActiveError(LingxuError):
    """Raised when a guild already owns a non-terminal Meeting Session."""


class AppendOnlyViolationError(LingxuError):
    """Raised when an event mutation would violate append-only storage."""


class EncryptionConfigurationError(LingxuError):
    """Raised when the file-encryption key is missing or malformed."""


class JobClaimError(LingxuError):
    """Raised when a Worker attempts to mutate a job it does not own."""


class RecordingPipelineError(LingxuError):
    """Raised when formal recording cannot finalise without plaintext residue."""


class MeetingPermissionError(LingxuError):
    """Raised when a non-host requests deterministic Meeting control."""


class MeetingVoiceStartupError(LingxuError):
    """Raised when formal voice startup fails at a content-safe named stage."""

    _MESSAGES: ClassVar[dict[str, str]] = {
        "capture_build": "会议未开始：语音采集运行时初始化失败，请管理员检查服务状态后重试。",
        "realtime_asr_prepare": (
            "会议未开始：实时语音识别模型未就绪。请管理员运行 ASR 模型下载与验证后重试。"
        ),
        "meeting_assistant_prepare": (
            "会议未开始：会议语音助手模型未就绪。请管理员检查 ASR、唤醒词和本地模型后重试。"
        ),
        "voice_connect": "会议未开始：聆序无法加入语音频道，请检查语音权限和网络后重试。",
        "dave_ready": (
            "会议未开始：Discord 端到端加密语音未就绪。聆序已安全退出语音频道，请稍后重试。"
        ),
        "voice_runtime_start": "会议未开始：语音运行时启动失败，请管理员查看安全阶段诊断。",
    }

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(self._MESSAGES.get(stage, self._MESSAGES["voice_runtime_start"]))


class MeetingRecoveryError(LingxuError):
    """Raised when persisted state, snapshot, and event history disagree."""


class MeetingComponentError(LingxuError):
    """Raised when a deterministic Meeting component rejects an operation."""


class ParticipantNotPresentError(MeetingComponentError):
    """Raised when an operation requires a participant currently in voice."""


class VoteNotOpenError(MeetingComponentError):
    """Raised when a vote mutation targets a missing, closed, or expired vote."""


class StructuredIntentError(LingxuError):
    """Raised when a structured-intent provider or decision is invalid."""


class StructuredIntentProviderError(StructuredIntentError):
    """Raised when the configured provider cannot return a safe candidate."""


class IntentConfirmationError(StructuredIntentError):
    """Raised when a pending candidate cannot be confirmed or rejected."""


class SummaryProviderError(LingxuError):
    """Raised when a summary provider cannot return a safe grounded result."""


class AgendaImportError(LingxuError):
    """Raised when a Markdown agenda cannot be safely accepted or committed."""


class AgendaImportProviderError(AgendaImportError):
    """Raised when the selected local model cannot return a valid agenda draft."""
