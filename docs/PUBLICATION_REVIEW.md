# 初始源码发布 / Initial source publication

所有者已授权清理后重新创建公开 Xav13RRR/LINGXU-V2，将已验证的干净快照提交并推送。
代码、相关测试及 Banner 使用 Apache-2.0，署名 Copyright 2026 Xav13RRR。
见 [授权记录](RIGHTS_CLEARANCE.md)、[NOTICE](../NOTICE) 及 ADR-0173。

The owner authorized clean recreation, commit and push of this Apache-2.0 source repository.
This is source publication, not production deployment. The existing archive remains private.

## 数据与历史 / Data and history

按逐文件白名单和哈希复制候选，在独立目录初始化全新 Git 仓库。不带入旧 `.git`、分支、
标签、旧提交、密钥、真实会议记录、数据库、录音、逐字稿、纪要、缓存或模型权重。
原私有档案的引用及可见性不变；本机运行中的服务不变。

Only allowlisted, hash-verified source files enter the new root commit. Old Git objects, private
working directories, credentials and real meeting/runtime data are excluded. The original archive's
refs and visibility, and existing runtime services, remain unchanged.

## 跳过的检查 / Skipped checks

Windows 清理后完整测试为 1,491 项通过、两项符号链接能力相关跳过：术语表链接的路径逃逸拒绝，
以及密钥扫描器在读取前拒绝指向仓库外的父目录链接。跳过不等于禁用防护，但该 Windows
环境没有实际验证这两种链接行为。Linux CI 单独执行它们并运行全套测试，须观察实际结果。
Task 16 人工语音矩阵不因发布或 CI 自动成为通过。

The clean Windows suite passed 1,491 tests with two real-filesystem symlink cases skipped because link
creation was unavailable. They cover glossary path escape and secret-scan ancestor-link rejection,
not meeting functionality. Companion deterministic tests pass. Linux CI explicitly executes both;
Linux success does not retrospectively prove Windows-specific link/permission behavior.

## 后续维护 / Future maintenance

PUBLICATION_STATUS.json 只记录本次初始发布授权，不授权生产部署或后续自动发布。
prepare_publication.py 保留为未发布候选工具：它不是推送器，会拒绝以已经授权发布的状态
作为新的准备输入。未来候选应有自己的准备状态和审核，不复用本次授权。

The retained candidate-preparation tool intentionally rejects an already-authorized release status
as input. Future candidates need their own review state; initial publication authority is not
automatic permission for future releases. Third-party licences and any future binary/image
redistribution obligations remain separate from this source-only publication.
