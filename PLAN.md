# Lingxu public-source maintenance plan

## Scope

Maintain the independent Lingxu Bot and Worker, with reproducible configuration, synthetic tests,
clear documentation and strict separation of public source from private deployment data.

## Invariants

- Discord is a normal Bot application, never a self-bot. Separate each user's voice track.
- AudioSink is non-blocking; capture and ASR respect purpose-specific consent and withdrawal.
- Append-only events and deterministic Runtime own meeting state. LLMs cannot mutate it.
- Post-meeting processing runs in the independently deployed Lingxu Worker.
- Keep original source text distinct from normalization and model suggestions.
- Provider output requires schema, evidence, consent and authoritative-fact validation.
- Label deterministic basic minutes when an allowed final-model failure occurs; input corruption,
  privacy failures and ambiguous publication still fail closed.
- Checkpoints, atomic files and publication receipts prevent unsafe automatic regeneration/sends.
- Pin dependencies, external Git references, model identities and manifests. Existing selections
  remain as recorded in code and MODELS.md; changes require review and regression testing.
- Preserve Guild isolation, scoped wake phrases and the single heavy-workload capacity boundary.
- Do not add TRPG/mystery-game features or import/call Lingji at runtime.

## Public evidence policy

Public tests use self-contained synthetic data or clearly identified public software/model assets.
Do not include real audio, transcripts, minutes, IDs, timestamps, output hashes, sample-size counts,
run journals, private-fixture dependencies or backups in source, tests, docs, releases or CI artifacts.
Public dependency/model/asset pins and hashes of code-owned schemas/prompts are not meeting evidence.
Review the origin of every literal digest; pattern matching alone is insufficient.

Keep operational evidence outside the public repository. Do not encode private fingerprints into
tests, deny lists or examples. Source-file inventories and synthetic test results may be recorded.
Never mark an unrun test as passed.

## Validation priorities

Complete the outstanding ordinary-voice matrix: host/member roles, leave/rejoin, mute reapplication,
permission loss, restart recovery and final cleanup of Bot-owned mutes. Its status remains
PARTIALLY_RUN_NOT_PASSED until those cases have been observed and accepted.

Maintain static checks, platform tests, package integrity and the public-source privacy boundary.
Use [TEST_REPORT.md](docs/TEST_REPORT.md) for coverage and [SECURITY.md](SECURITY.md) for release checks.

## References

- [Architecture](docs/ARCHITECTURE.md)
- [Current decisions](docs/DECISIONS.md)
- [Privacy](docs/PRIVACY.md)
- [Security](SECURITY.md)
- [Synthetic validation](docs/TEST_REPORT.md)
- [Publication](docs/PUBLICATION_REVIEW.md)
