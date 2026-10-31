# Test coverage and validation

## Automated checks

[GitHub Actions](https://github.com/Xav13RRR/LINGXU-V2/actions/workflows/ci.yml) runs:

- Ruff lint and formatting, strict mypy and the complexity regression limit.
- Public-file and credential scans.
- The synthetic test suite on Linux, including explicit filesystem symlink boundary tests.
- Windows configuration, launcher, path and type checks.
- CPU/GPU Compose validation, application-container build, package build and wheel-resource smoke.

The [initial source baseline run](https://github.com/Xav13RRR/LINGXU-V2/actions/runs/41176940672)
passed all four CI jobs: 1,493 Linux tests, two additional explicit symlink tests and 34 Windows
tests. These results apply to that run; use the checks on a specific commit when assessing a change.

To run local checks, follow the [deployment guide](DEPLOYMENT.en.md#12-pre-release-quality-gate).
Tests use self-contained synthetic fixtures. Real meetings and private data are not test dependencies.

## Platform limits

Windows hosts without symlink creation privileges skip the glossary path-escape and secret-scan
ancestor-link tests. Linux CI executes both. A Linux pass does not establish equivalent Windows
link/permission behavior; report skips separately from passes.

## Manual acceptance

`TASK16_STRICT_ORDINARY_VOICE_MATRIX=PARTIALLY_RUN_NOT_PASSED`.

The outstanding ordinary-voice matrix covers host/member roles, leave/rejoin, mute reapplication,
permission loss, restart recovery and cleanup of Bot-owned mutes. Automated tests do not replace
these observations. Configure and validate each deployment using the
[Discord setup guide](DISCORD_SETUP.md#english).

[STATE.md](../STATE.md) retains engineering Gate status. A completed source release does not imply
that every live deployment or manual acceptance case has passed.
