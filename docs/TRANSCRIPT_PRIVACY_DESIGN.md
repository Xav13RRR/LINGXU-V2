# Transcript and minutes privacy design

This file now documents the current design without private review examples, actual execution
measurements or historical fixture references. It is not a retained meeting report.

## Deterministic boundaries

Consent is purpose-specific and Guild-scoped. The durable timeline owns the valid intervals;
withdrawal stops subsequent processing and re-consent never fills a withdrawn gap. Apply cutoffs
before canonicalization, entity normalization, topic inference and final composition.

Preserve original source text separately. Canonicalization must not merge unrelated speakers or
silently discard short meaningful contributions. Normalize only through explicit, auditable rules.
Model-produced claims need valid source evidence; authority facts come from deterministic events.

## Publication

Only the approved final minutes document is public meeting output. Internal transcript/event JSON,
evidence, model traffic, journals and receipts remain private deployment data. Validate complete
source coverage and deterministic agenda, speaking and aggregate-vote facts before publication.
Never expose voter-to-option mappings or non-consented identities.

An allowed final-generation failure can produce labelled deterministic basic minutes. Corrupt
source/storage state, privacy failures and uncertain sends still block publication. Completed
receipts are immutable and retries require explicit deterministic authority.

## Validation

Use generated synthetic people, utterances, events and consent timelines. Exercise withdrawal,
cross-speaker isolation, evidence binding, malformed provider output, restart recovery and single
publication. Do not copy or describe real meetings, their statistics or their hashes in test source.
No public test requires a private corpus.

See [privacy](PRIVACY.md), [Worker](WORKER.md), and
[current decisions](DECISIONS.md) for the maintained contracts.
