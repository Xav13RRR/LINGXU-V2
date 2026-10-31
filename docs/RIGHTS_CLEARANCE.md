# 授权记录 / Rights authorization record

**权利及初始公开发布均已获所有者授权。 / Rights and initial public publication are owner-authorized.**

## 所有者确认 / Owner confirmation

所有者在本任务中明确确认：迁移代码和相关测试均有权授权，同意按 Apache-2.0 发布，**没有例外**；
版权人为所有者本人，Lingji 是其私有仓库；Banner 是其自己的素材，没有授权问题。
新仓库名称指定为 **LINGXU-V2**。同时要求只更新本地授权记录和候选，暂不推送或公开。

后续明确指令已解除初始发布暂停：创建公开 LINGXU-V2，提交并推送已验证的干净快照。
原私有档案不公开、不带入旧历史或敏感数据；不改动本地运行版本。所有者进一步授权清理后
删除并重新创建本仓库，使用全新 Git 历史；仅限 LINGXU-V2。参见 ADR-0173。

The owner explicitly confirmed authority over the migrated code and related tests, authorized
Apache-2.0 distribution with **no exceptions**, identified themselves as copyright holder and
Lingji as their private repository, and confirmed the banner is their own material.
The new repository name is **LINGXU-V2**. This instruction authorizes local preparation only,
not remote creation, push or public release.

The subsequent explicit instruction authorizes creating public LINGXU-V2 and committing/pushing
the verified clean snapshot. The private archive, old history, secrets and meeting data stay out;
existing runtime installations must not change. The owner further authorized deletion and clean
recreation of only LINGXU-V2 with fresh Git history. This supersedes the initial hold (ADR-0173).

- 公开署名 / Public attribution: **Copyright 2026 Xav13RRR**.
- 署名采用现有 GitHub 所有者标识，不推断或披露真实姓名；不把版权归属等同于每行代码作者身份。
  Attribution uses the existing GitHub owner handle, not an inferred legal name or authorship claim.
- 目标地址 / Intended destination: `https://github.com/Xav13RRR/LINGXU-V2`.
- 权利依据为所有者的明确确认，不声称完成了独立法律尽调。
  This is an owner-attested authorization record, not an independent legal ownership investigation.

## A. 迁移代码与测试 / Migrated code and tests

[CODE_PROVENANCE.md](CODE_PROVENANCE.md) 保留历史来源、修改分类和原始许可证。
历史的 Proprietary 字段不再作为当前候选的待确认项；后续所有者授权已明确。
Lingji 原仓库本身仍保持私有，不将其完整源码、Git 历史或运行数据导入新仓库。

| 历史分类 / Historical classification | 当前目标 / Current targets | 当前状态 / Status |
| --- | --- | --- |
| Copied and adapted | `src/lingxu/logging.py`, `src/lingxu/lifecycle.py` | 所有者确认 Apache-2.0，无例外 / Owner-confirmed, no exceptions |
| Copied and adapted | `src/lingxu/voice/frames.py`, `consent.py`, `dave_adapter.py`, `opus_guard.py`, `sink.py`（同一 voice 目录） | 同上 / Same |
| Rewritten | `src/lingxu/config.py`, `src/lingxu/utils.py`, `src/lingxu/bot/client.py`, `src/lingxu/bot_main.py` | 同上 / Same |
| Inspired | `src/lingxu/cli/doctor.py` | 同上 / Same |
| Rewritten and expanded tests | 原始记录中的 `tests/unit/test_*.py`、`tests/integration/test_*voice*.py` | 所有者确认相关测试的完整授权范围，无例外；不冒充逐行来源审计 |

Historical provenance remains intact; subsequent owner authorization covers the retained migrated
code and related tests without exceptions, including the historically wildcarded test scope.
Wildcard provenance is not retroactively represented as a precise line-by-line audit.
Lingji itself remains private and is not a runtime dependency or part of this export.

## B. Banner / Promotional artwork

- 文件 / File: `docs/assets/lingxu-banner.png`.
- 权利人 / Rightsholder: Xav13RRR（所有者自述 / owner confirmation）。
- 候选许可 / Candidate licence: **Apache-2.0**, under the selected project licence.
- 处理 / Treatment: 原样纳入白名单并恢复 README 展示，不重绘、不替换、不推断制作方式。
  Include the original bytes and README embed; do not infer an artist, tool or creation process.
- 来源与文件指纹 / Source and identity: [ASSET_PROVENANCE.md](ASSET_PROVENANCE.md).

这记录的是所有者对素材权利的确认，不声称图内每个视觉元素经过独立版权审查。
第三方依赖、模型及其标志不因本项目许可而被重新许可。
This records the owner's assurance, not an independent audit of every visual element. Third-party
dependencies, models and their marks retain their own rights and licences.

## C. 发布边界 / Publication boundary

已解决：代码／测试再许可权、版权署名、Banner 纳入、新仓库名称与目标地址。
后续指令已授权本次初始建库、提交、推送和公开，`publication_authorized=true`；不授权部署。

Resolved: code/test authorization, copyright attribution, banner inclusion, repository name and
intended URL. The subsequent owner instruction authorizes **this initial publication only**, not deployment.

继续保留原私有档案及旧历史。初始源码不含密钥、会议数据或旧 Git 历史。
本次实际创建、提交、推送和公开已获明确指令；生产部署或其它外部操作不在授权范围内。
Keep the original private archive and old history. Exclude credentials, meeting data and old Git
history from the source. This initial remote creation/commit/push/publication is now explicitly
authorized; production deployment and unrelated external operations remain out of scope.
