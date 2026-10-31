# Code provenance / 代码来源

Copyright 2026 Xav13RRR. The owner confirms authority over the migrated code and related tests
and authorizes Apache-2.0 distribution without exceptions. Lingji remains their private repository,
not a runtime dependency. See [authorization](RIGHTS_CLEARANCE.md) and [NOTICE](../NOTICE).

所有者已确认迁移代码和相关测试的授权无例外。下面保留来源关系与改写分类，不公开私有仓库的
提交指纹、实际会议样本、运行证据或统计；这些记录不是程序运行依赖。

## Retained migration relationships

Paths in the reference column are historical Lingji source paths, not files imported at runtime.
Original proprietary status is historical; the owner's subsequent Apache-2.0 grant applies to
the retained Lingxu derivatives. Third-party libraries retain their separate licences.

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
| Related unit/voice integration tests | Historical Lingji test modules | Rewritten and expanded | Owner authorization covers the retained test scope; use synthetic inputs only |

Historical wildcard test provenance is not represented as a line-by-line copyright audit.
Retired probe-specific modules and private-data-dependent tests are not part of this public baseline.
Subsequent storage, deterministic Runtime, ASR/TTS integration, Worker, scenario, console, privacy,
publication and repository-safety components were implemented for the independent Lingxu project.
No new Lingji copying or runtime connection occurs in the publication preparation.

## Notices and verification

The original internal proprietary notice is superseded for the authorized derivatives by the
owner's Apache-2.0 grant. Copyright attribution remains Xav13RRR. Dependencies and external model
assets keep their original notices; refer to [DEPENDENCIES.md](DEPENDENCIES.md) and
[MODELS.md](MODELS.md). A later binary/image distribution needs its own actual-component audit.

Public validation uses configuration, logging, voice, consent, lifecycle, namespace-independence
and synthetic integration tests. Actual meeting fixtures, their hashes/counts and production
execution records must never be added as provenance evidence.
