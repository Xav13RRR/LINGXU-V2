# Public synthetic validation

This report covers public source/software tests only. It contains no real meeting measurements,
private fixture inventories, transcripts or artifact fingerprints. Historical operational evidence
remains private and is not an input dependency of this test suite.

## Clean recreation gate

A fresh full validation is required after removal of private-data-dependent testing and replacement
of historical documents with current contracts. Results are recorded only after execution.
Fresh complete local run: **1,491 passed, 2 skipped**. Both skips are the Windows symlink
capability cases described below; there is no missing-private-fixture skip.

The first full run after adding Python-syntax privacy inspection failed on five exporter tests:
their synthetic .py placeholders were plain prose, not valid Python. The fixture now emits a
Python comment for .py files. The initial failure is retained as failed evidence; a fresh complete
run subsequently passed as recorded above. No real private fixture is used for the regression.

Ruff, formatting, strict mypy and the offline lock check passed. The 360-file source allowlist
passed the publication scanner and local cross-comparison against retained private credentials,
identities, artifact fingerprints and long-form content, without printing those values. Public
dependency/model/asset pins and code-owned schema/prompt snapshots were reviewed separately.
Package and staged/remote checks are release gates; their final results and Linux CI belong to
the exact published commit's GitHub Actions and the private release audit, not an assumed pass.

Required checks:

- Explicit source allowlist and file-origin review, including every test and document.
- Credential rules, known-private-evidence cross-comparison performed locally without logging values,
  and rejection of private fixture dependencies or pinned meeting-artifact manifests.
- Ruff, formatting, strict mypy, frozen dependency checks and self-contained synthetic tests.
- Source/wheel builds and member/content checks.
- Fresh root commit, exact staged/remote file inventory and unchanged private-archive refs/visibility.
- GitHub CI, including both real-filesystem symlink boundary tests on Linux.

## Platform-dependent checks

The Windows host cannot create the symlinks required by the glossary path-escape and secret-scan
ancestor-link tests. Their skip is an environment-capability limit, not removal of the safeguards.
Linux CI explicitly runs both. Windows-specific permission/link behavior remains a separate
platform acceptance boundary; do not claim an unexecuted case passed.

The public suite must not require retained real meetings or private fixtures. A missing real-data
fixture must not be used to excuse public test coverage; use synthetic fixtures instead.

## Manual acceptance

Task 16's strict ordinary-voice matrix remains PARTIALLY_RUN_NOT_PASSED. Automated checks and source
publication do not alter that status or authorize a production deployment.

CI evidence for each public commit is available in
[GitHub Actions](https://github.com/Xav13RRR/LINGXU-V2/actions/workflows/ci.yml).
