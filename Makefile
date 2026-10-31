UV := uv
PYTHON := $(UV) run python

.PHONY: install lint format-check typecheck complexity-check test test-unit test-integration doctor bot worker api
.PHONY: migrate benchmark-realtime-asr benchmark-tts replay-meeting
.PHONY: scenario-validate scenario-simulate docker-cpu docker-gpu

install:
	$(UV) sync --all-extras

lint:
	$(UV) run ruff check .

format-check:
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy

complexity-check:
	$(UV) run ruff check src --select C901 --config "lint.mccabe.max-complexity=55"

test:
	$(UV) run pytest

test-unit:
	$(UV) run pytest tests/unit

test-integration:
	$(UV) run pytest tests/integration

doctor:
	$(PYTHON) -m lingxu.cli.doctor

bot:
	$(PYTHON) -m lingxu.bot_main

worker:
	$(PYTHON) -m lingxu.worker_main

api:
	@$(PYTHON) -c "raise SystemExit('The Host console is embedded in lingxu-bot; no standalone API service is deployed')"

migrate:
	$(PYTHON) -m lingxu.storage.migrate

benchmark-realtime-asr:
	$(PYTHON) scripts/benchmark_realtime_asr.py

benchmark-tts:
	$(PYTHON) scripts/benchmark_tts.py

replay-meeting:
	$(PYTHON) scripts/smoke_worker_pipeline.py

scenario-validate:
	$(PYTHON) -m lingxu.scenario.cli scenarios/fixed_meeting_v1.yaml

scenario-simulate:
	$(PYTHON) -m lingxu.scenario.simulate_cli scenarios/fixed_meeting_v1.yaml

docker-cpu:
	docker compose up --build lingxu-bot lingxu-worker

docker-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
