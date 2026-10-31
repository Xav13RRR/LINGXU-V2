# Privacy hardening maintenance checklist

Use this checklist when changing transcript processing or minutes generation.

1. Read the current architecture, privacy contracts and Gate state.
2. Express a defect with a self-contained synthetic regression. Never paste retained meetings,
   actual identifiers, private artifact hashes or observed sample counts into fixtures.
3. Validate consent cutoffs, original/canonical source separation and speaker isolation.
4. Check model schema/evidence boundaries and deterministic authority facts.
5. Exercise failure/restart/currentness and publication idempotency without live services.
6. Run static checks, applicable tests, package/privacy checks and source-file provenance review.
7. Report manual and unrun checks separately. Plan live tests and deployment with the operator,
   including participant consent, data handling and rollback requirements.

Deployment-specific diagnostics and real-data reviews belong in private operator storage, not this
repository, CI logs, issues or release attachments. Public model/dependency identity pins remain
documented in [MODELS.md](MODELS.md) and [DEPENDENCIES.md](DEPENDENCIES.md).

See [current plan](../PLAN.md), [decisions](DECISIONS.md) and
[validation](TEST_REPORT.md).
