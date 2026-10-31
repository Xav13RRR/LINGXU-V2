# API 与凭据配置

[English](CONFIGURATION.en.md) · [部署指南](DEPLOYMENT.zh-CN.md) · [Discord 配置](DISCORD_SETUP.md)

本文适用于自己的独立部署，不需要也不应取得维护者的密钥或会议数据。现有部署不需要为了
文档更新而重新生成密钥、重新下载模型或重启。不要覆盖已有 `.env`。

## 1. 首次配置清单

仅首次安装时，在仓库根目录复制 `.env.example` 为 `.env`；使用本地编辑器填写值。
不要将密钥写进终端命令、聊天、Issue、截图、录屏或 shell 历史。

| 配置项 | 是否必需 | 获取方式与保存位置 |
| --- | --- | --- |
| `DISCORD_TOKEN` | 是 | Developer Portal → 自己的 Application → Bot → Reset Token；只存本地 `.env` |
| `DISCORD_APPLICATION_ID` | 是，非密钥 | Developer Portal → General Information → Application ID；存 `.env` |
| `FILE_ENCRYPTION_KEY` | 是 | 每个新部署生成一次 URL-safe Base64 编码的 32 字节随机密钥；存 `.env`，另做安全备份 |
| `SUMMARY_LLM_API_KEY` | 是 | 自己的 DeepSeek API 账户；Windows 推荐使用下面的掩码输入脚本，`.env` 中留空 |
| `DISCORD_TEST_GUILD_ID` | 否 | 正式部署留空；仅开发服务器立即同步命令时填写 Guild ID |
| `HOST_CONSOLE_ENABLED` / `HOST_CONSOLE_GUILD_ID` | 可选 | 需要 localhost 主持台才启用并指定一个 Guild；不开放公网 |
| `HOST_CONSOLE_TOKEN` | 可选 | 留空时从文件加密密钥派生；参见部署指南，不上传或共享 |
| `LLM_API_KEY` / `MEETING_ASSISTANT_LLM_API_KEY` / `AGENDA_IMPORT_LLM_API_KEY` | 默认不需要 | 固定本地 Ollama 方案留空；不要填入 DeepSeek Key |

DeepSeek 的变量名在本项目中是 **`SUMMARY_LLM_API_KEY`**，不是官方示例中的
`DEEPSEEK_API_KEY`。保持 `.env.example` 中的 `SUMMARY_LLM_BASE_URL=https://api.deepseek.com`、
`SUMMARY_LLM_MODEL=deepseek-flash`、模型指纹及其它参数不变。本指南不授权切换模型或服务商。

## 2. 获取自己的 DeepSeek API Key

1. 从[官方 API 文档](https://api-docs.deepseek.com/)进入其 API Key 入口，登录
   [DeepSeek 平台](https://platform.deepseek.com/)，为本部署创建独立密钥。
2. 检查 API 账户可用余额和用量限制；聊天产品订阅不应被当作 API 可用额度的证明。
3. 先了解[隐私边界](PRIVACY.md)：会后 Worker 会把经过同意过滤的会议文字和允许的身份标签
   发往远端 DeepSeek，并产生 API 费用。它不是完全离线的纪要功能。
4. 不要用真实会议验证新密钥。首次端到端测试使用所有参与者明确同意的合成内容，测试结果
   仅留本地，不能上传到仓库。配置预检不等于 API 鉴权、余额或推理成功。

## 3. Windows 推荐方式：DPAPI + 维护的启动器

使用同一个 Windows 运行账户执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/set_deepseek_api_key.ps1
```

在掩码提示中输入 Key。密文保存在仓库外的
`%LOCALAPPDATA%\Lingxu\deepseek-api-key.dpapi`，与当前 Windows 用户绑定。
该脚本不启动服务，也不修改 `.env`。首次安装时让 `.env` 的 `SUMMARY_LLM_API_KEY` 留空，
不要另外设置同名进程环境变量：启动器优先使用已有环境变量或 `.env` 值，再读取 DPAPI。

完成部署指南中的模型下载后，执行配置预检：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_lingxu.ps1 -DryRun
```

它不会重建或替换 Lingxu 服务，也不会发起模型推理，但必要时会启动 Docker Desktop。
预检通过后，首次部署才运行 `启动聆序.cmd`（CPU 使用 `-Cpu`）。已有部署应遵循空闲检查和
备份升级流程，不要为验证密钥随意重启。

**不要在 DPAPI 配置后直接改用 `docker compose up`。** Compose 不会解密这个文件。
启动器会把密钥传给 Worker 容器；退出时清除的是它自己注入的宿主进程变量，**不是**运行中
容器的环境。具有 Docker 管理权限的人仍可查看容器配置，应按凭据管理员保护 Docker 访问。

## 4. 手动 Compose / 非 Windows

DPAPI 脚本仅适用于 Windows。Linux/macOS 的手动部署需由自己的秘密管理系统向 Compose
进程提供 `SUMMARY_LLM_API_KEY`，或用本地编辑器填入被忽略且受限的 `.env`。后者是明文存储，
不要把它描述为加密。Unix 上用 `chmod 600 .env` 限制访问；Windows 使用文件安全属性限制
到实际运行账户和必要管理员。不要放在共享或自动云同步目录。

只有密钥已提供、模型已下载、部署空闲并完成备份后，才能使用部署指南的手动命令。
Compose 的进程环境变量优先于 `.env`；排错时检查是否存在旧覆盖值，但不要打印值。
仅支持当前固定服务商和模型；非 Windows 全新部署仍需独立验收，不等于已通过本机 GPU 验收。

## 5. 生成加密密钥与更新凭据

首次部署可使用 PowerShell 内置随机数生成器，无需安装完整 Python 依赖：

```powershell
$keyBytes = New-Object byte[] 32
$keyRng = [Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $keyRng.GetBytes($keyBytes)
    [Convert]::ToBase64String($keyBytes).Replace('+', '-').Replace('/', '_')
} finally {
    [Array]::Clear($keyBytes, 0, $keyBytes.Length)
    $keyRng.Dispose()
}
```

输出只复制到本地 `.env` 的 `FILE_ENCRYPTION_KEY`，不要在录屏、共享终端或转录会话中运行。
丢失密钥会使旧加密数据无法恢复；更改此值不等于完成数据密钥轮换。

- 更新 DeepSeek Key：重跑掩码脚本，移除旧环境覆盖，再按维护窗口重建相关容器。
  脚本本身不会更新运行中的容器。验证新凭据后，在供应商平台撤销旧 Key。
- 更新 Discord Token：在 Bot 页面重置，更新本地配置，按维护流程更新 Bot/Worker。
- 不要为更新 API Key 生成新的 `FILE_ENCRYPTION_KEY`；已有数据需要保留原值。
- 怀疑泄露时应先限制访问，并在对应平台撤销受影响的访问凭据。数据加密根密钥需要专门迁移，
  不应直接覆盖。

## 6. 安全排错与提交检查

| 现象 | 检查方法 |
| --- | --- |
| DeepSeek credential is missing | 同一 Windows 账户重跑掩码脚本；确认通过启动器启动，或已提供手动 Compose 凭据 |
| DPAPI 无法解密 | 切回原账户，或用新账户重新输入自己的 Key；不要复制别人的 DPAPI 文件 |
| API 鉴权／额度／限流失败 | 在供应商平台核对密钥、余额与限制；不要粘贴请求头、响应正文或会议内容 |
| 模型身份不匹配 | 检查默认配置和旧覆盖项；不要删除校验、随意改指纹或自动切模型 |
| 纪要无法发布 | 检查目标频道的 View Channel、Send Messages、Attach Files 及频道覆盖权限 |

```powershell
git check-ignore .env
uv run python scripts/scan_secrets.py
uv run python -m scripts.check_public_files
```

只使用 `docker compose config --quiet`；不要分享完整 Compose 配置、`docker inspect`、环境变量
列表、`.env`、DPAPI 文件、数据库、音频、逐字稿、纪要或运行记录。
检查通过只覆盖声明的范围，不能替代历史审查和人工检查。参见 [SECURITY.md](../SECURITY.md)。
