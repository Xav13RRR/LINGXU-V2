"""Construct the independent Phase 7 Meeting application graph."""

from __future__ import annotations

from dataclasses import dataclass

from lingxu.config import Settings
from lingxu.meeting.agenda_import import AgendaImportService, OllamaAgendaImportProvider
from lingxu.meeting.components import (
    AgendaService,
    SpeakingQueueService,
    TimerService,
    VotingService,
)
from lingxu.meeting.controller import MeetingController
from lingxu.meeting.intent_service import MeetingIntentService
from lingxu.meeting.intents import RuleIntentResolver
from lingxu.meeting.mute_enforcement import MuteEnforcementService
from lingxu.meeting.panels import MeetingPanelBindingService
from lingxu.meeting.runtime import MeetingRuntime
from lingxu.meeting.speaking_flow import ControlledSpeakingFlowService
from lingxu.meeting.structured_intents import OpenAICompatibleStructuredIntentProvider
from lingxu.storage.agenda_import_repository import AgendaImportRepository
from lingxu.storage.component_repositories import (
    AgendaRepository,
    ConsentChangeRepository,
    GuildConsentRepository,
    ParticipantRepository,
    TimerRepository,
    VoteRepository,
)
from lingxu.storage.database import Database
from lingxu.storage.intent_repository import IntentCandidateRepository
from lingxu.storage.mute_lease_repository import MuteLeaseRepository
from lingxu.storage.pronunciation_repository import PronunciationAliasRepository
from lingxu.storage.repositories import (
    MeetingEventStore,
    MeetingSessionRepository,
    MeetingSnapshotStore,
    ProcessingJobRepository,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles


@dataclass(slots=True)
class MeetingApplication:
    database: Database
    sessions: MeetingSessionRepository
    participants: ParticipantRepository
    runtime: MeetingRuntime
    controller: MeetingController
    intents: MeetingIntentService
    timers: TimerService
    agenda_imports: AgendaImportService
    panel_bindings: MeetingPanelBindingService
    mute_leases: MuteLeaseRepository
    mute_enforcement: MuteEnforcementService
    pronunciation_aliases: PronunciationAliasRepository


class _TextCommandEchoDetector:
    """Text diagnostic commands are not Discord playback transcripts."""

    @staticmethod
    def is_echo(transcript: str) -> bool:
        return False


def create_meeting_application(settings: Settings) -> MeetingApplication:
    """Build services without performing network or model initialization."""

    database = Database.open(settings.database_url)
    sessions = MeetingSessionRepository(database.sessions)
    agenda_import_repository = AgendaImportRepository(database.sessions)
    agenda_import_api_key = (
        settings.agenda_import_llm_api_key.get_secret_value()
        if settings.agenda_import_llm_api_key is not None
        else None
    )
    agenda_import_provider = (
        OllamaAgendaImportProvider(
            base_url=settings.agenda_import_llm_base_url,
            model=settings.agenda_import_llm_model,
            expected_model_digest=settings.agenda_import_llm_model_digest,
            api_key=agenda_import_api_key,
            timeout_seconds=settings.agenda_import_llm_timeout_seconds,
            max_output_tokens=settings.agenda_import_llm_max_output_tokens,
        )
        if settings.agenda_import_llm_enabled
        else None
    )
    agenda_imports = AgendaImportService(
        sessions=sessions,
        repository=agenda_import_repository,
        provider=agenda_import_provider,
        enabled=settings.agenda_import_llm_enabled,
        model_name=settings.agenda_import_llm_model,
        model_digest=settings.agenda_import_llm_model_digest,
        max_attachment_bytes=settings.agenda_import_max_attachment_bytes,
        max_source_chars=settings.agenda_import_max_source_chars,
        max_blocks=settings.agenda_import_max_blocks,
        max_chunk_chars=settings.agenda_import_max_chunk_chars,
        max_chunks=settings.agenda_import_max_chunks,
        max_items=settings.agenda_import_max_items,
        confidence_threshold=settings.agenda_import_confidence_threshold,
        default_duration_minutes=settings.agenda_import_default_duration_minutes,
        default_vote_seconds=settings.agenda_import_default_vote_seconds,
    )
    events = MeetingEventStore(database.sessions)
    panel_bindings = MeetingPanelBindingService(sessions=sessions, events=events)
    participants = ParticipantRepository(database.sessions)
    guild_consents = GuildConsentRepository(database.sessions)
    consent_changes = ConsentChangeRepository(database.sessions)
    agenda = AgendaService(AgendaRepository(database.sessions), events)
    speaking = SpeakingQueueService(participants=participants, events=events)
    timer_repository = TimerRepository(database.sessions)
    voting = VotingService(
        votes=VoteRepository(database.sessions),
        timers=timer_repository,
        participants=participants,
        events=events,
    )
    timers = TimerService(
        timers=timer_repository,
        events=events,
        vote_closer=voting,
    )
    mute_leases = MuteLeaseRepository(database.sessions)
    mute_enforcement = MuteEnforcementService(events=events, timers=timers)
    pronunciation_aliases = PronunciationAliasRepository(database.sessions)
    speaking_flow = ControlledSpeakingFlowService(
        participants=participants,
        events=events,
        timers=timers,
    )
    runtime = MeetingRuntime(
        sessions=sessions,
        events=events,
        snapshots=MeetingSnapshotStore(database.sessions),
        snapshot_interval=settings.meeting_event_snapshot_interval,
    )
    controller = MeetingController(
        sessions=sessions,
        events=events,
        session_service=MeetingSessionService(sessions, SessionFiles(settings.data_root)),
        runtime=runtime,
        participants=participants,
        guild_consents=guild_consents,
        consent_changes=consent_changes,
        agenda=agenda,
        speaking=speaking,
        timers=timers,
        voting=voting,
        speaking_flow=speaking_flow,
        default_vote_seconds=settings.default_vote_seconds,
        jobs=ProcessingJobRepository(database.sessions),
        agenda_import_guard=agenda_imports,
        mute_enforcement=mute_enforcement,
    )
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key is not None else None
    provider = (
        OpenAICompatibleStructuredIntentProvider(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=api_key,
            timeout_seconds=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
            max_utterance_chars=settings.llm_max_utterance_chars,
            expected_model_digest=settings.llm_model_digest or None,
        )
        if settings.llm_intents_enabled
        else None
    )
    intents = MeetingIntentService(
        controller=controller,
        candidates=IntentCandidateRepository(database.sessions),
        rules=RuleIntentResolver(_TextCommandEchoDetector()),
        provider=provider,
        rule_intents_enabled=settings.rule_intents_enabled,
        llm_intents_enabled=settings.llm_intents_enabled,
        confidence_threshold=settings.llm_intent_confidence_threshold,
    )
    return MeetingApplication(
        database=database,
        sessions=sessions,
        participants=participants,
        runtime=runtime,
        controller=controller,
        intents=intents,
        timers=timers,
        agenda_imports=agenda_imports,
        panel_bindings=panel_bindings,
        mute_leases=mute_leases,
        mute_enforcement=mute_enforcement,
        pronunciation_aliases=pronunciation_aliases,
    )
