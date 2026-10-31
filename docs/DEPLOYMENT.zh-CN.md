# 聆序 LINGXU 本地部署指南

本文档对应仓库内当前验证的本机版本：Windows、Docker Desktop、Discord 普通语音频道、
本地 Ollama `qwen3.5:9b`、本地 SenseVoice ASR、本地 CosyVoice 中文女声、独立会后
Worker，以及项目所有者批准的官方 DeepSeek 最终纪要 API。持久数据库、音频和模型卷留在
本机；已授权的完整逐字稿、固定模板、权威议程和聚合投票事实会由 Worker 发往 DeepSeek。
默认不会把 Ollama 或 TTS 端口发布到宿主机。

[English version](DEPLOYMENT.en.md) · [返回 README](../README.md) · [中文使用指南](USAGE.zh-CN.md)

## 1. 部署组成

| 服务/卷 | 作用 | 网络与数据边界 |
| --- | --- | --- |
| `lingxu-bot` | Discord Gateway、DAVE、会议 Runtime、实时 ASR、语音问答、主持台 | 连接 Discord；挂载 `/data`、`/models` 和 tmpfs `/runtime` |
| `lingxu-worker` | 会后录音校验、转写补全、逐字稿、DeepSeek 最终纪要、附件发布、恢复 | 与 Bot 共享独立 Lingxu 数据卷；连接 Discord 与官方 DeepSeek API，不调用其它项目 Worker |
| `lingxu-ollama` | `qwen3.5:9b` 本地结构化意图、议程导入和短回答 | 仅 `lingxu-llm` 内部网络，无宿主机端口 |
| `lingxu-tts` | CosyVoice `中文女` 语音合成 | 仅 `lingxu-tts` 内部网络，无宿主机端口 |
| `lingxu-data` | SQLite/WAL、加密 Session、TTS cache | 生产数据；升级前必须备份 |
| `lingxu-models` | SenseVoice、FSMN-VAD、sherpa-onnx KWS | 与应用镜像分离 |
| `lingxu-ollama` 卷 | Ollama 模型 | 仅下载器和 Ollama 服务访问 |
| `lingxu-cosyvoice` 卷 | CosyVoice 与 Wetext 文件 | 服务只读挂载 |

内部技术标识保留小写 `lingxu`，公开项目名为“聆序 LINGXU”。不要重命名现有 Docker 卷、
Python 包或数据库表，否则会破坏升级与恢复兼容性。

## 2. 前置条件

### Windows GPU 方案（当前本机验证方案）

- 64 位 Windows 10/11；
- Docker Desktop，启用 WSL 2 后端和 Docker Compose v2；
- 支持容器计算的 NVIDIA 驱动；
- Git；
- 一个独立 Discord Application、Bot Token，以及至少一个有权安装 Bot 的目标服务器；
- 足够的磁盘空间。仅固定 Qwen Ollama manifest 即约 6.59 GB，其它 ASR/TTS/KWS 模型和
  Docker 镜像还需要额外空间。

用以下命令确认环境：

```powershell
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
git --version
```

如果不具备 NVIDIA GPU，可使用本文第 10 节的 CPU 兼容方案；功能相同，但 Qwen 和 TTS
冷启动及推理明显更慢。

### 可选开发/验收工具

本地质量命令需要 Python 3.12 和 `uv 0.9.2`。仅运行 Docker 生产服务时不要求在宿主机
安装全部 Python 依赖。

## 3. 创建 Discord Application

1. 打开 [Discord Developer Portal](https://discord.com/developers/applications)，创建专用于
   聆序 LINGXU 的 Application。
2. 在 **Bot** 页面创建 Bot 并生成 Token。Token 只写入本地 `.env`，不要发到聊天、截图、
   Issue 或 Git。
3. 不需要启用 privileged Message Content Intent。代码只使用 Guilds、Voice States 和
   标准 Guild Messages；用户直接真实 Mention Bot 时，Discord 会提供该消息的附件信息。
4. 在安装设置中启用 `bot` 与 `applications.commands` scopes。
5. 发布文字频道允许 View Channels、Send Messages、Attach Files；语音频道允许 View Channel、
   Connect、Speak。强制有序／演示议程另需可选 Mute Members。不要授予 Administrator，
   同时检查角色权限与频道覆盖权限。
6. 使用 Developer Portal 的安装链接把同一个 Bot 安装到每个目标 Guild。安装者需要该
   服务器的相应管理权限。复制 Application ID → `DISCORD_APPLICATION_ID`。
7. 正式多服务器部署将 `DISCORD_TEST_GUILD_ID` 留空。只有开发调试时才填写一个 Guild ID，
   让命令只同步到该测试服务器。

详细权限说明见 [DISCORD_SETUP.md](DISCORD_SETUP.md)。不要复用其它项目或 Bot 的 Token。

## 4. 获取代码并创建本地配置

使用独立公开仓库 `Xav13RRR/LINGXU-V2`，不要把私有工程档案当成分发源。
请配置自己的凭据，已有 `.env` 不应覆盖：

```powershell
git clone https://github.com/Xav13RRR/LINGXU-V2.git LINGXU
Set-Location LINGXU
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

先阅读 [API 与凭据配置](CONFIGURATION.zh-CN.md)，了解必填／可选项、自己的 API Key 获取、
Windows DPAPI 与手动 Compose 的区别、凭据更新和安全排错。
打开被 Git 忽略的 `.env`，至少填写：

```dotenv
DISCORD_TOKEN=<Discord Bot Token>
DISCORD_APPLICATION_ID=<Application ID>
DISCORD_TEST_GUILD_ID=
FILE_ENCRYPTION_KEY=<URL-safe base64 32-byte key>
```

留空的 `DISCORD_TEST_GUILD_ID` 会同步仅限 Guild 安装/上下文的全局 Slash Commands，因而后续
新安装的服务器不需要改 `.env` 或重启独立实例。全局命令同步后可能需要短暂等待 Discord
客户端刷新。若启用本地主持台，另行设置一个固定 `HOST_CONSOLE_GUILD_ID`；主持台仍只控制
这一台服务器。

本机部署仍只有一个重型服务槽：正式会议（包括会后 Worker 发布）、常驻助手和 Markdown
议程模型处理会原子竞争该槽。其它 Guild 的重型入口会收到固定的“服务容量已满”提示，响应中
不会披露占用方。会议创建、状态查看等轻量确定性配置可以先完成，开始重型流程时才占槽。

不要修改仓库内已固定的模型名、revision 或 digest，除非完成新的模型审查和全量回归。
生产本机 GPU 方案可以在 `.env` 设置：

```dotenv
COMPOSE_FILE=docker-compose.yml;docker-compose.gpu.yml
```

如果本机 Compose 不接受分号，可不设置此项；一键启动器会显式加载两个 Compose 文件。

### 生成文件加密密钥

仅使用 Docker 的用户可采用[配置指南中的 PowerShell 内置生成器](CONFIGURATION.zh-CN.md#5-生成加密密钥与更新凭据)，
无需安装完整 Python 依赖。另一种方式是在安装 Python 3.12 和 uv 后，在仓库根目录运行：

```powershell
uv sync --frozen --no-dev
uv run python -c "from lingxu.storage.encryption import FileCipher; print(FileCipher.generate_key())"
```

只把输出写入 `.env` 的 `FILE_ENCRYPTION_KEY`。该密钥用于 AES-256-GCM Session 文件；
丢失密钥后已有加密录音无法恢复。密钥应与数据备份分开安全保存，但绝不能提交到 Git。

## 5. 首次构建和固定模型下载

下载器是显式 `model-download` profile；常规启动不会每次联网下载模型。所有下载步骤只在
首次安装或经过批准的模型替换时运行。

### 5.1 构建应用和 TTS 镜像

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build `
  lingxu-bot lingxu-worker lingxu-tts
```

### 5.2 下载 Ollama Qwen 模型

服务容器本身位于无外网内部网络，下载由独立一次性容器完成：

```powershell
docker compose stop lingxu-ollama
docker compose --profile model-download run --rm lingxu-ollama-downloader
docker compose up -d --no-deps lingxu-ollama
docker compose exec -T lingxu-ollama ollama show qwen3.5:9b
```

生产固定 manifest digest：
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`。
Bot 在每次模型调用前核验本地 inventory；不匹配时拒绝推理。

### 5.3 下载 CosyVoice 中文女声

```powershell
docker compose --profile model-download run --rm lingxu-tts-downloader
docker compose up -d lingxu-tts
```

下载器验证固定 `FunAudioLLM/CosyVoice-300M-SFT` revision、模型文件聚合摘要以及离线
Wetext 规范化文件。TTS 服务拒绝不完整或被替换的权重。

### 5.4 下载唤醒词模型

```powershell
docker compose --profile model-download run --rm lingxu-wakeword-downloader
```

该步骤下载 sherpa-onnx 中英 KWS 模型并验证官方压缩包及实际运行文件摘要。默认唤醒词为
“呼叫指挥中心”，保留受限的声学近似路径。Manage Guild 成员可用 `/wakeup set` 为各服务器
独立选择连续 4-16 个汉字；命令在持久化前用已固定的 `tokens.txt` 预检读音，自定义词使用
精确声学路径，活动会议或常驻助手无需重启即可切换。`/wakeup reset` 恢复默认值。

### 5.5 下载并验证 SenseVoice 与 FSMN-VAD

下面的一次性命令把 ASR 权重写入共享 `lingxu-models` 卷，并在返回前核验两个 `model.pt`
的 SHA-256：

```powershell
docker compose --profile model-download run --rm lingxu-asr-downloader
```

若下载中断，重新运行同一命令；摘要不匹配时不要绕过错误，应删除错误的模型卷内容并按
[MODELS.md](MODELS.md)重新核对来源。Bot 与 Worker 运行时只从上述已验证本地快照加载，
不会在会议启动或会后处理中查询 ModelScope。

## 6. 启动当前本机 GPU 版本

首次模型和 `.env` 配置完成后，以掩码方式输入一次 DeepSeek API 密钥。脚本只在仓库外
保存 Windows 当前用户可解密的 DPAPI 密文，不把密钥写入 `.env`：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/set_deepseek_api_key.ps1
```

完成后，最简单的启动方式是双击仓库根目录：

```text
启动聆序.cmd
```

启动器会在 Docker Desktop 未运行时自动启动并等待，把 DPAPI 密钥仅解密到当前启动进程，
静默验证必需配置，只通过 `lingxu-migrate` 构建一次 Bot/Worker/迁移共享应用镜像，并仅在
Dockerfile 或 TTS 服务源码指纹变化时重建 TTS；随后再次执行只返回聚合计数的空闲检查，
以 `--no-build` 启动 Bot、Worker、Ollama 和 TTS，等待四项健康检查，再验证 Discord
`bot_ready`。结束验证时会清除进程环境中的密钥；它不会打印
凭据、逐字稿或会议内容。从 Git 工作区启动时，它还会在 Compose 校验前自动绑定完整 HEAD：
干净工作区使用精确 SHA，存在修改时追加 `-dirty`；已经显式设置的进程级
`LINGXU_BUILD_REVISION` 不会被覆盖。没有 `.git` 的源码包使用 Compose 回退值。

PowerShell 等价命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lingxu.ps1
```

只检查、不改变服务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lingxu.ps1 -DryRun
```

需要无条件重建 TTS 时显式添加 `-RebuildAll`；正常启动无需重复构建未变化的 TTS 镜像。

根目录 CMD 也会转发可选参数，因此可在终端直接验证真实包装器路径：

```text
启动聆序.cmd -DryRun
```

手工启动仅适用于已经通过 Compose 进程环境变量或受限本地 `.env` 提供
`SUMMARY_LLM_API_KEY` 的用户；**Compose 不会解密 DPAPI 文件**。Windows DPAPI 用户请使用
上面的启动器。详见[手动凭据配置](CONFIGURATION.zh-CN.md#4-手动-compose--非-windows)。
已有部署先执行第 9 节的空闲检查及备份流程。

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## 7. 验证部署

```powershell
docker compose ps
docker compose exec -T lingxu-bot python -m lingxu.cli.doctor --strict-discord --runtime-only
docker compose exec -T lingxu-bot python scripts/smoke_meeting_assistant.py
docker compose exec -T lingxu-bot python scripts/smoke_agenda_import.py
```

健康状态应满足：

- 一次性 `lingxu-migrate` 容器以状态码 0 退出；Bot 和 Worker 仅只读核对 Alembic 当前版本，
  不再并发执行 DDL；
- `lingxu-bot`、`lingxu-worker`、`lingxu-ollama`、`lingxu-tts` 均为 `healthy`；
- Bot 日志出现 `application_commands_synced`；默认常驻配置下在 Discord 验证 `/meeting`、
  `/assistant`、`/wakeup`，不要再以旧版 count 为 2 作为验收条件；
- Bot 日志出现 `bot_ready`；
- Doctor 的 Discord、加密、ASR、KWS、Ollama、TTS 和 DAVE 检查通过；
- `http://127.0.0.1:8765/healthz` 仅在启用主持台后本机可访问；
- 宿主机没有对外发布 Ollama 11434 或 TTS 服务端口。

不要把完整 `docker compose config` 输出复制到公共位置，因为渲染后的 environment 可能
包含秘密。只运行 `docker compose config --quiet`。

Compose 将凭据按进程能力分配：Bot 获得本地意图/会议助手/议程导入/主持台凭据，Worker
获得最终纪要凭据；Discord 令牌和文件加密密钥仅由确实需要它们的 Bot 与 Worker 共享。
`lingxu-migrate` 不接收这些秘密。若在不重建镜像的情况下恢复旧数据库，请先执行
`docker compose run --rm lingxu-migrate`，再启动 Bot/Worker。

## 8. 可选本地主持台

在 `.env` 中设置：

```dotenv
HOST_CONSOLE_ENABLED=true
HOST_CONSOLE_GUILD_ID=<允许访问的单一 Guild ID>
```

未设置 `HOST_CONSOLE_TOKEN` 时，系统从文件加密根密钥进行域分离派生。启动后在本机终端
查看登录令牌：

```powershell
docker compose exec -T lingxu-bot python -m lingxu.console.token
```

浏览器打开 `http://127.0.0.1:8765`。不要截图、记录或转发令牌。控制台仅绑定 localhost
和一个 Guild，并使用 HttpOnly session、CSRF 与同源校验。

## 9. 停止、重启、升级和备份

正常停止：

```powershell
docker compose stop
```

Bot 和 Worker 必须以 exit code 0 退出。不要在会议录音最终化或 Worker 发布阶段强制删除
容器。

升级前：

1. 确认没有活动会议和 `PENDING/RUNNING` Worker Job；
2. 记录共享应用镜像与 TTS 当前 image ID 并添加回滚 tag；
3. 正常停止 Bot 与 Worker；
4. 完整备份 `lingxu-data` 卷；
5. 拉取新代码、运行质量门并重建；
6. 先要求一次性 `lingxu-migrate` 成功退出，再启动 Worker 恢复流程和 Bot；
7. 复核 revision、Doctor、命令同步和四项健康状态。

不要提交或上传 `backups/`。详细步骤见 [ROLLBACK.md](ROLLBACK.md) 和
[FAILURE_RECOVERY.md](FAILURE_RECOVERY.md)。

## 10. CPU 兼容部署

CPU Profile 不请求 NVIDIA 设备。以下手动命令需要按第 6 节另行提供 Compose 凭据；
DPAPI 用户使用后面的启动器 `-Cpu` 方式：

```powershell
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml build lingxu-migrate lingxu-tts
docker compose -f docker-compose.yml up -d --no-build
```

或运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lingxu.ps1 -Cpu
```

Qwen 模型、ASR 与 TTS 的功能边界不变，但实时短回答和动态 TTS 延迟可能显著增加。CPU
模式主要用于兼容和恢复，不是当前本机的低延迟推荐方案。

## 11. 固定模型来源

| 用途 | 模型 | 官方/上游链接 |
| --- | --- | --- |
| 本地 LLM | `qwen3.5:9b` / Qwen3.5-9B | [Ollama](https://ollama.com/library/qwen3.5:9b) · [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B) |
| 最终会议纪要 | 官方 DeepSeek API / `deepseek-flash` | [DeepSeek API](https://api-docs.deepseek.com/) |
| ASR | `iic/SenseVoiceSmall` | [ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall) |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | [ModelScope](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch) |
| 唤醒词 | sherpa-onnx KWS Zipformer | [模型索引](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html) · [Releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) |
| TTS | `FunAudioLLM/CosyVoice-300M-SFT` | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT) · [GitHub](https://github.com/FunAudioLLM/CosyVoice) |
| 文本规范化 | `pengzhendong/wetext` | [ModelScope](https://modelscope.cn/models/pengzhendong/wetext) |

所有生产 revision、文件摘要、manifest digest、许可证和替换要求见
[MODELS.md](MODELS.md)。不要仅凭可变的 `master` 或 `latest` 标签认定生产身份。

## 12. 发布前质量门

开发者在构建或升级前运行：

```powershell
uv sync --all-extras --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
uv run python scripts/scan_secrets.py
uv run python -m scripts.check_public_files
uv build
docker compose config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
```

应用镜像以固定的 Python 3.12.11 多架构 manifest digest 为基础，并使用 builder/runtime
多阶段构建；Git 与 uv 只存在于 builder。运行镜像写入 OCI `version`、`revision` 和 `source`
标签，CI 将提交 SHA 作为 revision。Compose 可通过 `LINGXU_BUILD_VERSION`、
`LINGXU_BUILD_REVISION`、`LINGXU_BUILD_SOURCE` 提供同一元数据；未提供 revision 时明确为
`unknown`，不能把它当作可追溯发布。

项目所有者已依据多轮真实正式会议，验收双向 DAVE、Phase 1 E2E 和 outcome extraction
Gate。已退役的独立探针和私有 annotation preparation 不再是部署验收指令。
