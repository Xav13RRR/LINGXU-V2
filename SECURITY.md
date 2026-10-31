# 安全与数据边界 / Security and data boundaries

## 中文

### 禁止提交的内容

- 任何实际 API Key、Discord Token、文件加密密钥、DPAPI 密文或 DAVE/MLS 材料。
- 真实会议音频、逐字稿、纪要、数据库、模型请求／响应、身份与回执。
- 本地会议运行／生成记录，包括真实 Session/Job/服务器/频道/消息标识、产物哈希、
  Token 用量、录音与发言数量、实际运行报告和备份清单；“不含正文”不代表可以公开。
- 私有评审、备份或诊断压缩包。请勿把它们附到 Issue、PR、Release 或 Actions 制品中。

真实证据统一留在被忽略的本机 `private/`、`.runtime/`、`data/`、`backups/` 或
`tests/private_fixtures/` 中。测试代码可以提交，但必须使用合成输入和占位身份；不要复制真实
会议内容作为回归 fixture。公开文档只保留工程约束、合成测试结论和已记录的聚合 Gate 状态。

测试代码本身也不得固定真实样本的哈希、计数或私有目录依赖；不以“缺文件就跳过”掩盖这些
元数据。扫描器检查 Python 语法树中的私有 fixture 依赖与会议产物指纹表，并检查文档中的
私有会议统计。公开模型／依赖／授权素材与代码 schema 的完整性指纹不属于真实会议证据。

`.gitignore` 只防止通常的误添加；它不会移除已跟踪文件、旧提交、其它克隆或远端缓存。
当前文档清理不等于 Git 历史清理。已选 Apache-2.0 和新建无旧历史仓库方案；旧仓库保持私有。
迁移代码／测试授权、Banner 及 `Xav13RRR/LINGXU-V2` 名称已由所有者确认；
ADR-0172 已授权本次初始建库／提交／推送／公开，不授权任何生产部署。

### 提交前检查

在根目录安装固定开发环境，然后检查：

```powershell
uv sync --all-extras --frozen
uv run python scripts/scan_secrets.py
uv run python -m scripts.check_public_files
# 在准备好要提交的文件并自行暂存后，检查整个暂存快照：
uv run python -m scripts.check_public_files --index
git diff --cached --stat
```

可选启用本仓库的本地提交钩子（先检查是否已配置其它钩子，不要直接覆盖）：

```powershell
git config --get core.hooksPath
git config --local core.hooksPath .githooks
```

类 Unix 系统另需让 `.githooks/pre-commit` 可执行。钩子在缺少虚拟环境时拒绝提交，并检查实际
暂存对象；CI 再检查已提交的文件。**CI 在推送之后执行，不能撤销一次泄露。** 钩子可以被绕过，
规则也不能理解所有语义敏感内容，所以仍需人工审核和单独的全历史审查。不要使用 `git add -f`
绕过私有文件保护，不要把本机工作目录直接打包上传。

### 本机保护和漏洞报告

限制 `.env`、私有数据目录、备份和 Docker 管理权限；不要给予 Everyone、共享账户或云同步
服务不必要的访问。DPAPI 与运行账户绑定，但注入 Worker 的密钥仍可被 Docker 管理员读取。
参见 [API 配置](docs/CONFIGURATION.zh-CN.md)。不要在配置更新时更换旧数据的加密根密钥。

若发现漏洞／疑似泄露，先用已有的维护者私密渠道联系；如尚无私密渠道，只请求提供联系渠道，
不要公开漏洞详情或敏感附件。启用 GitHub Private Vulnerability Reporting 后可使用仓库
Security → Report a vulnerability；当前文档不声称该远端设置已经启用。
不要将真实密钥交给维护者复现。暴露的访问凭据应在供应商平台撤销／轮换；历史清理不能代替
撤销。数据加密密钥需单独迁移，避免丢失已有数据。

## English

### Never commit

- Actual API keys, Discord tokens, file-encryption roots, DPAPI ciphertext or DAVE/MLS material.
- Real audio, transcripts, minutes, databases, model traffic, identities or publication receipts.
- Real meeting execution/generation records, including Session/Job/Guild/channel/message IDs,
  artifact hashes, token usage, audio/utterance counts, run reports and backup inventories.
  Content-free operational telemetry is not automatically appropriate for public distribution.
- Private reviews, backups or diagnostic archives, including issue/PR attachments, releases and
  Actions artifacts.

Keep actual evidence under ignored local `private/`, `.runtime/`, `data/`, `backups/` or
`tests/private_fixtures/`. Test source is welcome, but use synthetic inputs and placeholder identities;
never turn a real meeting into a public regression fixture. Public docs retain engineering contracts,
synthetic-test results and already-recorded aggregate Gate states only.

Test source must not pin real sample hashes/counts or depend on private fixture directories, even
behind a missing-file skip. The scanner inspects Python syntax for those dependencies/manifests and
flags private meeting statistics in documents. Public dependency/model/approved-asset and code-schema
integrity pins are distinct from retained meeting evidence. Manual provenance review remains required.

`.gitignore` is an accident barrier, not removal of tracked files, commits, other clones or remote
caches. Current-document cleanup does not clean history. Apache-2.0 and a new history-free repository
are selected; the original stays private. Migrated code/tests, the banner and the planned
`Xav13RRR/LINGXU-V2` identity are owner-confirmed. ADR-0172 separately authorizes this initial
repository creation/commit/push/publication, never production deployment.

### Before committing

Install the pinned development environment and run from the repository root:

```powershell
uv sync --all-extras --frozen
uv run python scripts/scan_secrets.py
uv run python -m scripts.check_public_files
# After deliberately staging the intended changes, inspect the complete index:
uv run python -m scripts.check_public_files --index
git diff --cached --stat
```

Optionally enable the repository's local hook; check any existing hook configuration before replacing it:

```powershell
git config --get core.hooksPath
git config --local core.hooksPath .githooks
```

On Unix also make `.githooks/pre-commit` executable. The hook refuses a missing virtual environment
and inspects actual staged blobs; CI checks committed files again. **CI runs after upload and cannot
undo a disclosure.** Hooks can be bypassed and pattern checks cannot identify every semantic private
record. Manual review and a separate complete-history audit remain mandatory. Never force-add private
files or upload an archive of the live working directory.

### Local protection and reporting

Restrict `.env`, private data, backups and Docker administration. Avoid unnecessary Everyone,
shared-account or cloud-sync access. DPAPI is account-bound, but Docker administrators can read
credentials injected into Worker. See [API setup](docs/CONFIGURATION.en.md). Do not replace the
encryption root for existing data during an API-key update.

Report suspected vulnerabilities/leaks through an established private maintainer channel. If none
exists, request a private channel without posting exploit details or sensitive attachments. Once
GitHub Private Vulnerability Reporting is enabled, use Security → Report a vulnerability; this
document does not claim that remote setting is enabled. Never send actual credentials to reproduce
an issue. Revoke/rotate exposed access credentials at their provider: history cleanup is not a
substitute. The data-encryption root requires a separate migration to preserve existing data.
