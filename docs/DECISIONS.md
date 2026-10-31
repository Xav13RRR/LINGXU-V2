# Current architecture and publication decisions

This is a public contract register, not a historical run log. Detailed original engineering
records remain private. No actual meeting IDs, measurements, artifact fingerprints or transcript
samples belong here.

## ADR-0001: Independent runtime

Lingxu owns its application, Bot, Worker, database, model/data directories and deployment.
Lingji is a private source-provenance reference only, never a runtime dependency. External
dependencies retain their original licences and pinned identities.

## ADR-0146: Manual acceptance is not inferred from automated checks

The owner permits separately requested work to proceed while Task 16's ordinary-voice matrix is
deferred. TASK16_STRICT_ORDINARY_VOICE_MATRIX remains PARTIALLY_RUN_NOT_PASSED. Test doubles, public
CI and release success do not replace host/member, rejoin, mute, permission-loss or restart/cleanup
observations. Source publication does not authorize a new meeting or deployment.

## ADR-0164: Deterministic basic minutes on allowed final-generation failure

An approved final-model attempt may fail through transport/output validation. The independent
Worker may then render clearly labelled basic minutes from the same consent-filtered sources and
deterministic authority facts. It must not publish rejected prose, invent confirmations or bypass
privacy/storage/source-integrity checks. Existing publications remain immutable.

## ADR-0165: Structured model output and code-owned composition

Compact complete consented sources enter the fixed post-meeting inference path. Typed claims and
evidence references are validated locally; code composes authoritative agenda/speaking/aggregate-vote
facts and the final document. Models cannot change state, consent, source records or publication
authority. Retry generations and attempt journals remain deterministic and bounded.

## ADR-0167: Existing model selection remains unchanged

The owner-selected remote model is deepseek-flash; local Qwen, ASR, VAD, KWS and TTS identities remain
the pinned selections in configuration and MODELS.md. Licence/privacy remediation changes no model,
prompt, request parameters, dependency or live-call policy.

## ADR-0171: Owner rights and public identity

The owner confirms authority over migrated code and related tests under Apache-2.0 without exceptions,
identifies Lingji as their private repository and the banner as their own material. Attribution uses
the public GitHub handle Xav13RRR without inferring a legal name. Third-party components are not
relicensed. The planned public repository is Xav13RRR/LINGXU-V2. See RIGHTS_CLEARANCE.md and NOTICE.

## ADR-0173: Authorized clean recreation and stronger privacy gate

The owner explicitly permits deletion and recreation of the newly created LINGXU-V2 after a complete
review. Only that repository may be deleted; original LINGXU remains private with unchanged refs and
runtime. Do not reuse the earlier public repository's Git objects or history.

Review the complete source allowlist, including test code and historical prose, for real meeting
content, private sample fingerprints, measurements and execution records. Replace historical
narratives with current contracts. Private-data tests stay in the private archive; the public suite
is self-contained. Model/dependency/approved-asset and code-schema hashes retain their public purpose.

A fresh independently initialized tree must pass staged-content, privacy, dependency, package and
test checks before it is pushed. Verify public visibility, Apache-2.0 detection, remote tree identity,
CI and absence of the previous commit after recreation. Preserve any required local audit privately.
This exception authorizes source publication only, never deployment or mutation of meeting data.
