# Partner Collaboration Rules

- Repository files are the only source of truth.
- Read `PLAN.md`, `STATE.md`, `AGENTS.md`, `docs/DECISIONS.md`, and
  `docs/TEST_REPORT.md` before changing code.
- Work only in the independent Lingxu repository.
- Treat any Lingji checkout as a read-only source-provenance reference.
- Never import or call Lingji at runtime.
- Migrated code must be recorded in `docs/CODE_PROVENANCE.md`.
- Complete implementation phases in order.
- Never proceed past a failed or awaiting-manual-test gate unless an explicit project-owner
  sequencing exception is recorded in `docs/DECISIONS.md`; an exception never changes the Gate
  result or counts as validation.
- Never mark an unrun test as passed.
- Keep dependency versions pinned.
- Keep Git dependencies pinned to full commit SHAs.
- Do not expose secrets, DAVE/MLS key material, raw audio, transcripts, or meeting content in logs.
- Never put real Meeting execution/generation records, identities, receipts, artifact hashes,
  private review notes, or production backup inventories in tracked documents or test fixtures.
  Keep operational evidence under ignored `private/`; public evidence uses synthetic tests only.
- Public tests must be self-contained: never depend on retained real meetings, embed their artifact
  hashes/counts, or hide them behind a skip when private fixture files are absent.
- Do not rewrite Git history, change the licence, push, publish, or deploy as part of publication
  preparation without separate project-owner confirmation. A clean tree is not a clean history.
- Update `STATE.md` after every meaningful implementation step.
- Record architectural deviations in `docs/DECISIONS.md`.
- Prefer working code and tests over speculative prose.
- Keep Discord user audio tracks separate.
- AudioSink must remain non-blocking.
- Require each Discord user to consent before the probe persists that user's audio.
- LLMs cannot mutate meeting state.
- The append-only event log is the process source of truth.
- The post-meeting pipeline must run in the independently deployed Lingxu Worker.
- Do not implement TRPG or mystery-game features in this plan.
- The original first-delivery limit is superseded by recorded project-owner sequencing decisions.
- Pause before adding any LLM dependency, selecting an LLM provider/model, or enabling an LLM
  call. The project owner must choose the concrete LLM model first.
