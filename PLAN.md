# Lingxu public-source maintenance plan

## Scope and authority

Maintain the independent Lingxu Bot and Worker. This repository is a clean public source baseline,
not an archive of real meetings, private fixtures or production execution evidence. The owner
authorized deletion/recreation of Xav13RRR/LINGXU-V2 after a complete privacy review (ADR-0173).
The original LINGXU and Lingji repositories remain private; existing services must not change.

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
  remain as recorded in code and MODELS.md; no provider/model change or live call is authorized here.
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

## Current validation and sequencing

Inherited engineering Gates remain in STATE.md. The strict ordinary-voice Task 16 matrix remains
PARTIALLY_RUN_NOT_PASSED: host/member roles, leave/rejoin, mute reapplication, permission loss,
restart recovery and final owned-mute cleanup still require their explicit manual acceptance.
The owner's source-publication exception does not convert that Gate and does not authorize deployment.

For this publication: review every allowlisted file; inspect test data provenance and historical
documents; privately cross-check retained evidence fingerprints/content without printing them;
run static checks, tests and package checks; create a fresh Git root with no old objects; verify
the exact staged and remote trees; confirm the original repository remains private.

## References

- [Architecture](docs/ARCHITECTURE.md)
- [Current decisions](docs/DECISIONS.md)
- [Privacy](docs/PRIVACY.md)
- [Security](SECURITY.md)
- [Synthetic validation](docs/TEST_REPORT.md)
- [Publication](docs/PUBLICATION_REVIEW.md)
