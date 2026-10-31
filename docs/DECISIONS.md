# Architecture decisions

## ADR-0001: Independent runtime

Lingxu owns its Bot, Worker, database, model directories and deployment. Lingji is a source-provenance
reference, never a runtime dependency. Dependencies retain their own licences and pinned identities.

## ADR-0146: Source maintenance and manual acceptance

Source maintenance and releases may proceed while the strict ordinary-voice matrix is incomplete.
This sequencing exception does not change
`TASK16_STRICT_ORDINARY_VOICE_MATRIX=PARTIALLY_RUN_NOT_PASSED` or count as validation.
Deployment acceptance still requires the outstanding host/member, rejoin, mute, permission-loss
and restart/cleanup checks described in [TEST_REPORT.md](TEST_REPORT.md).

## ADR-0164: Deterministic basic minutes on final-generation failure

If final-model transport or output validation fails, the independent Worker may render clearly
labelled basic minutes from the same consent-filtered sources and deterministic authority facts.
It must not publish rejected prose, invent confirmations or bypass privacy, storage or
source-integrity checks. Existing publications remain immutable.

## ADR-0165: Structured model output and local composition

Compact consented sources enter a fixed post-meeting inference path. Typed claims and evidence
references are validated locally. Code composes authoritative agenda, speaking and aggregate-vote
facts and the final document. Models cannot change state, consent, source records or publication
authority. Retry generations and attempt journals are deterministic and bounded.

## ADR-0167: Model identity

The remote model is `deepseek-flash`; Qwen, ASR, VAD, KWS and TTS use the pinned identities in
configuration and [MODELS.md](MODELS.md). Changes require model, licence, privacy and regression
review. Policy digests identify request contracts, not immutable weights behind a remote alias.

## ADR-0171: Licence and attribution

Project code, migrated code, related tests, documentation and the banner are distributed under
Apache-2.0. Copyright attribution is Xav13RRR. Third-party components retain their own licences.
See [licence scope](RIGHTS_CLEARANCE.md), [code provenance](CODE_PROVENANCE.md) and [NOTICE](../NOTICE).

## ADR-0173: Public-source privacy boundary

Public tests are self-contained. Source, documentation, test fixtures, releases and CI artifacts
must exclude real meeting content, private sample fingerprints, execution records and measurements.
Model, dependency, asset and code-schema integrity hashes retain their documented technical purpose.
Review both file contents and Git history before release; a clean working tree does not establish
that earlier commits are safe. See [SECURITY.md](../SECURITY.md).
