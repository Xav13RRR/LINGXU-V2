# Local rollback runbook

Rollback is a controlled replacement of Lingxu containers and, only when required, restoration of
the independent Lingxu data volume. It never reads from or writes to Lingji.

## Before every upgrade

1. Confirm there is no active Meeting. If a Meeting is active, finish it or schedule downtime.
2. Record current image IDs:

   ```text
   docker image inspect lingxu-app:latest --format "{{.Id}}"
   docker image inspect lingxu-lingxu-tts:latest --format "{{.Id}}"
   ```

3. Apply durable rollback tags to those exact images:

   ```text
   docker image tag lingxu-app:latest lingxu-app:rollback
   docker image tag lingxu-lingxu-tts:latest lingxu-lingxu-tts:rollback
   ```

4. Stop Bot and Worker cleanly and verify exit code zero. Leave Ollama running if only application
   code is changing.
5. Back up the complete `lingxu-data` Docker volume with Docker Desktop's volume backup/export
   operation. Record the backup timestamp and current Alembic revision. The volume includes the
   SQLite database, WAL state after clean shutdown, encrypted Sessions, and TTS cache.

Do not back up `/runtime`: it is deliberately ephemeral plaintext tmpfs. The separately named
`lingxu-ollama`, `lingxu-models`, and `lingxu-cosyvoice` volumes need backup only when model assets
are changing.

## Application-only rollback

Use this path when no migration ran or the prior application is documented as compatible with the
current database:

```text
docker compose stop lingxu-bot lingxu-worker
docker image tag lingxu-app:rollback lingxu-app:latest
docker compose up -d --no-build lingxu-worker lingxu-bot
docker compose ps
```

Run health, Discord identity/command-sync, structured-intent, and console authentication checks.
Do not use a prior image against a newer schema unless compatibility was explicitly verified.
Bot, Worker, and migration intentionally share `lingxu-app`; their independent commands and the
one-shot migration authority remain defined by Compose.

## Read-only storage review

Preview Lingxu-scoped image and local backup retention without changing Docker or the filesystem:

```text
uv run python scripts/preview_storage_cleanup.py
```

The tool has no apply/delete mode. It protects container-referenced, `latest`, `rollback`, recent,
and newest-per-repository images, plus recent/newest or non-gzip backups. A `review-candidate` is
only an operator prompt: validate its archive integrity and rollback value before any separately
approved manual removal.

## Data rollback

Use this only when an upgrade changed schema/data incompatibly:

1. Stop Bot and Worker.
2. Export the failed `lingxu-data` volume for incident analysis; do not overwrite the last known
   good backup.
3. Restore the pre-upgrade complete volume backup through Docker Desktop.
4. Restore the rollback-tagged shared application image.
5. Run the one-shot `lingxu-migrate` service and require exit code zero; then start Worker for
   recovery evidence and finally Bot. Bot and Worker only check schema currentness and never own DDL.
6. Verify Alembic revision, WAL mode, event/snapshot recovery, Job lease recovery, health, and zero
   `.part` residue before reopening meetings.

Never run an ad-hoc Alembic downgrade against production. Restore the database and encrypted
Session files as one consistent backup.

## Ollama/model rollback

Stop Bot before model replacement. Restore the `lingxu-ollama` volume backup or pull the previously
approved tag through the downloader profile, then restore the previous exact
`LLM_MODEL_DIGEST`. The Bot refuses inference until the local `/api/tags` digest matches.

## CosyVoice rollback

Stop Bot and TTS before replacement. Restore the previous `lingxu-cosyvoice` volume backup or run
the approved downloader for the prior immutable revision, restore the matching
`TTS_MODEL_REVISION` and `TTS_MODEL_DIGEST`, retag the rollback TTS image, then start TTS before
Bot. TTS and Bot both refuse a model-digest mismatch. For an emergency compatibility rollback,
set `TTS_PROVIDER=espeak-ng` and `TTS_VOICE=cmn`, rebuild/restart only Bot, and retain the
CosyVoice volume for investigation.

## Abort criteria

Keep the deployment offline and investigate if any of these occur:

- nonzero SIGTERM exit;
- migration revision mismatch;
- event/snapshot/materialised-state mismatch;
- expired Job not requeued;
- plaintext media under persistent storage;
- model digest mismatch;
- console reachable beyond `127.0.0.1`;
- unauthenticated API/WebSocket data;
- anonymous vote projection contains voter identity;
- secret scan finding.
