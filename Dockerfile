ARG PYTHON_BASE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

FROM ${PYTHON_BASE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.9.2

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN UV_COMPILE_BYTECODE=0 uv sync --frozen --no-dev

FROM ${PYTHON_BASE} AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        espeak-ng \
        ffmpeg \
        libopus0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 lingxu \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin lingxu

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY scenarios ./scenarios

RUN mkdir -p \
        /data/glossaries \
        /data/sessions \
        /data/tts_cache \
        /models \
        /runtime \
    && chown -R 10001:10001 /data /models /runtime

USER 10001:10001

ENV PATH="/app/.venv/bin:${PATH}" \
    DATA_ROOT=/data/sessions \
    GLOSSARY_ROOT=/data/glossaries \
    RUNTIME_TMP=/runtime \
    MODEL_CACHE_DIR=/models \
    TTS_CACHE_DIR=/data/tts_cache

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-m", "lingxu.cli.doctor", "--runtime-only"]

CMD ["python", "-m", "lingxu.bot_main"]

FROM runtime-base AS tools

COPY scripts/__init__.py \
     scripts/preview_full_transcript_minutes.py \
     ./scripts/

ARG BUILD_VERSION="0.1.0"
ARG BUILD_REVISION="unknown"
ARG BUILD_SOURCE="https://github.com/Xav13RRR/LINGXU-V2"

LABEL org.opencontainers.image.title="Lingxu" \
      org.opencontainers.image.description="Local-first Discord voice meeting host and worker" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${BUILD_REVISION}" \
      org.opencontainers.image.source="${BUILD_SOURCE}"

FROM runtime-base AS runtime

COPY scripts/__init__.py \
     scripts/analyze_latest_assistant.py \
     scripts/benchmark_tts.py \
     scripts/check_deployment_idle.py \
     scripts/deploy_fixed_scenario.py \
     scripts/download_asr_models.py \
     scripts/download_wakeword_model.py \
     scripts/pre_generate_tts.py \
     scripts/reconcile_discord_publication.py \
     scripts/smoke_agenda_import.py \
     scripts/smoke_host_console.py \
     scripts/smoke_meeting_assistant.py \
     scripts/smoke_structured_intent.py \
     ./scripts/

ARG BUILD_VERSION="0.1.0"
ARG BUILD_REVISION="unknown"
ARG BUILD_SOURCE="https://github.com/Xav13RRR/LINGXU-V2"

LABEL org.opencontainers.image.title="Lingxu" \
      org.opencontainers.image.description="Local-first Discord voice meeting host and worker" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${BUILD_REVISION}" \
      org.opencontainers.image.source="${BUILD_SOURCE}"
