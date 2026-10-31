# 源码发布检查 / Source release checks

## 中文

发布前检查以下内容：

1. 核对 `.publication-files.txt`，只包含源码、合成测试、文档和许可素材。
2. 运行[安全检查](../SECURITY.md)，检查工作区、暂存内容和待发布的 Git 历史。
   不得包含密钥、会议数据、运行记录或私有样本的哈希与统计。
3. 核对 [LICENSE](../LICENSE)、[NOTICE](../NOTICE)、代码及素材来源和第三方许可证。
4. 运行静态检查、自动测试和构建，检查源码包与 wheel 的文件清单和内容。
5. 核对远端提交、文件清单及 CI 结果。测试覆盖范围和人工验收限制见[测试说明](TEST_REPORT.md)。

`scripts/prepare_publication.py` 用于在本地 `private/` 下生成白名单源码快照和校验清单，
不执行提交、推送或部署。它要求输入的 `PUBLICATION_STATUS.json` 中
`publication_authorized=false`；已发布仓库的状态不适合作为新候选的准备输入。
检查导出结果后，再按发布流程提交。

## English

Before releasing source:

1. Review `.publication-files.txt`: include only source, synthetic tests, documentation and licensed assets.
2. Run the [security checks](../SECURITY.md) against the working tree, index and history to be published.
   Exclude credentials, meeting data, execution records and private sample hashes or statistics.
3. Check [LICENSE](../LICENSE), [NOTICE](../NOTICE), code/asset provenance and third-party licences.
4. Run static checks, automated tests and builds; inspect source-distribution and wheel contents.
5. Verify the remote commit, file inventory and CI results. See [test coverage](TEST_REPORT.md)
   for platform limitations and manual acceptance requirements.

`scripts/prepare_publication.py` creates an allowlisted source snapshot and integrity inventory under
local `private/`; it does not commit, push or deploy. Its input `PUBLICATION_STATUS.json` must have
`publication_authorized=false`. A released repository's status is not preparation input for a new
candidate. Review the exported files before submitting them for release.
