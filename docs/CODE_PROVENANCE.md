# Code provenance / 代码来源

Copyright 2026 Xav13RRR. The Lingji-derived code and related tests distributed in this repository
are licensed under Apache-2.0. Lingxu does not import or call Lingji at runtime.
See [licence scope](RIGHTS_CLEARANCE.md) and [NOTICE](../NOTICE).

本仓库中源自 Lingji 的代码和相关测试采用 Apache-2.0。下表记录来源和改动分类；
Lingxu 运行时不导入或调用 Lingji。

## Retained migration relationships

Paths in the reference column are historical Lingji source paths, not files imported at runtime.
The Lingji source was proprietary; the Lingxu derivatives are distributed under Apache-2.0.
Third-party libraries retain their separate licences.

| Lingxu target | Lingji reference | Classification | Main adaptation / verification |
| --- | --- | --- | --- |
| `src/lingxu/config.py` | `src/lingji/config.py` | Rewritten | Independent settings, paths and credentials; configuration tests |
| `src/lingxu/logging.py` | `src/lingji/logging.py` | Copied and adapted | Namespace isolation and privacy redaction; logging tests |
| `src/lingxu/lifecycle.py` | `src/lingji/lifecycle.py` | Copied and adapted | Independent lifecycle ownership; lifecycle tests |
| `src/lingxu/utils.py` | `src/lingji/utils.py` | Rewritten | Independent identifier utilities; synthetic unit tests |
| `src/lingxu/cli/doctor.py` | `src/lingji/cli/doctor.py` | Inspired | Independent readiness, pin and boundary checks |
| `src/lingxu/voice/frames.py` | `src/lingji/voice/frames.py` | Copied and adapted | Independent frame ownership; frame/sink tests |
| `src/lingxu/voice/consent.py` | `src/lingji/voice/consent.py` | Copied and adapted | Consent/capture ordering; concurrency tests |
| `src/lingxu/voice/dave_adapter.py` | `src/lingji/voice/dave_adapter.py` | Copied and adapted | Independent DAVE adapter and safe diagnostics |
| `src/lingxu/voice/opus_guard.py` | `src/lingji/voice/opus_guard.py` | Copied and adapted | Allowlisted content-free decoder diagnostics |
| `src/lingxu/voice/sink.py` | `src/lingji/voice/sink.py` | Copied and adapted | Independent consent-aware non-blocking capture |
| `src/lingxu/bot/client.py`, `src/lingxu/bot_main.py` | `src/lingji/bot/client.py`, `src/lingji/main.py` | Rewritten | Independent application, services and command surface |
| Related unit/voice integration tests | Historical Lingji test modules | Rewritten and expanded | Synthetic configuration, lifecycle, voice and integration coverage |

Subsequent storage, deterministic Runtime, ASR/TTS integration, Worker, scenario, console, privacy,
publication and repository-safety components were implemented for the independent Lingxu project.

## Notices and verification

Dependencies and external model assets keep their original notices; refer to
[DEPENDENCIES.md](DEPENDENCIES.md) and
[MODELS.md](MODELS.md). A later binary/image distribution needs its own actual-component audit.

Public validation uses configuration, logging, voice, consent, lifecycle, namespace-independence
and synthetic integration tests. Actual meeting fixtures, their hashes/counts and production
execution records must never be added as provenance evidence.
