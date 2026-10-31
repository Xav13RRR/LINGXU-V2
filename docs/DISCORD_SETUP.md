# Discord Bot 配置 / Discord Bot setup

[中文部署](DEPLOYMENT.zh-CN.md) · [English deployment](DEPLOYMENT.en.md) ·
[API 配置](CONFIGURATION.zh-CN.md) · [API setup](CONFIGURATION.en.md)

## 中文

### 1. 创建自己的应用

1. 打开 [Discord Developer Portal](https://discord.com/developers/applications)，创建独立
   Application。不要使用用户账号 Token、self-bot 或其它项目的 Bot Token。
2. General Information → Application ID，复制到本地 `DISCORD_APPLICATION_ID`。它不是 Bot Token。
3. Bot → 创建 Bot（若尚未创建）→ Reset Token，把新 Token 只写入被忽略且受限的本地
   `DISCORD_TOKEN`。不要截图、贴到聊天、Issue 或 Git。
4. Installation → 允许 **Guild Install**。安装 scopes 选择 `bot` 和
   `applications.commands`，使用该应用提供的安装链接邀请到目标服务器。
5. 安装者需要服务器管理／添加应用权限。Bot 本身不需要 Administrator 或 Manage Guild。
   `/assistant join` 和 `/wakeup set|reset` 的调用者需要 Manage Guild，这是用户权限。
6. 不需要开启 Message Content、Server Members 或 Presence 特权 Intent。代码使用 Guilds、
   Voice States 和标准 Messages；真实直接 Mention Bot 的附件导入不依赖 Message Content。

门户标签可能调整，以官方 [Application 配置说明](https://docs.discord.com/developers/quick-start/getting-started)
和 [安装说明](https://docs.discord.com/developers/resources/application#installation-context) 为准。

### 2. 按频道配置最小权限

| 范围 | 权限 | 用途 |
| --- | --- | --- |
| 发布文字频道 | View Channel、Send Messages、**Attach Files** | 主持面板及最终 `minutes.md` 附件；缺少附件权限会阻断发布 |
| 普通语音频道 | View Channel、Connect、Speak | 加入语音、接收与播报 |
| 可选强制发言模式 | Mute Members | 主持人启用强制有序／演示议程时需要；自由／建议模式不要求 |
| 主持人／管理员 | Manage Guild，或对应会议的创建者权限 | 管理命令按实际命令检查；不是给 Bot 的全局管理权限 |

同时检查服务器角色和**目标频道的权限覆盖**。不要通过 Administrator 绕过排错。
使用普通语音频道；不要把 Stage、论坛或线程当作已验证的发布／语音路径。
权限定义见 [Discord 官方文档](https://docs.discord.com/developers/topics/permissions)。

### 3. 本地配置与启动

按 API 配置指南填写 `DISCORD_TOKEN`、`DISCORD_APPLICATION_ID`、
`FILE_ENCRYPTION_KEY`，配置自己的 DeepSeek Key 并下载固定模型。
不要只运行 Bot 而遗漏独立 Worker；使用维护的部署启动器。

生产环境将 `DISCORD_TEST_GUILD_ID` 留空，使用全局、仅服务器可用的命令。
开发调试时可填单一测试 Guild ID，使用 Discord 客户端 Developer Mode → Copy Server ID 获取。
全局命令传播／客户端刷新可能需要等待，不要反复重建 Bot 或泄露 Token 排查。

默认启用常驻助手时应有三个顶层命令组：

- `/meeting`：会议、议程、同意、发言、投票、主持控制、重试与删除。
- `/assistant`：常驻语音问答的 `join|status|stop`，不产生会议录音或纪要。
- `/wakeup`：`show|set|reset`，每服务器独立配置，默认“呼叫指挥中心”。

一个 Bot 可安装到多个服务器，但本部署同时只有一个重型工作槽。会议直到 Worker 发布结束、
常驻助手直到退出，都可能占用该槽；这不是安装或 Token 错误。

### 4. 首次验收与排错

1. 验证 Bot/Worker 和本地模型服务健康，Discord 出现上述命令。
2. 在准备接收纪要的普通文字频道创建会议。参与者按使用指南明确同意录音、实时 ASR 和纪要；
   提前告知会后文字会发送到 DeepSeek。使用合成内容进行首次验收。
3. 测试语音加入／播报、停止会议，再确认 Worker 最终发送的 `minutes.md` 附件可下载。
   只验证 Bot 在线或语音可用，不等于验证会后发布。
4. 没有命令：检查安装 scopes、服务器上下文、测试 Guild 配置和全局传播；不要开启无关特权。
5. 不能加入语音：检查频道覆盖权限；强制发言模式检查 Mute Members。
6. 投票私信失败：检查接收者隐私设置，并按使用指南使用服务器命令降级。
7. 已处理但未发布：检查原创建频道是否仍存在，以及 Attach Files / Send Messages 权限。
   不要盲目重跑、手工重复发送或更改数据库。
8. 所有验收音频、文字、标识、回执和运行记录仅存本地私有目录，不上传 Issue 或仓库。

完整会议、Markdown 议程导入与故障恢复见 [中文使用](USAGE.zh-CN.md)。
导入需主持人在会议创建的文字频道中真实 `@Bot` 并附一个 `.md` 文件；纯文本写出名字
不算 Mention。成员标识以 Discord ID 为准，不按昵称猜测。

## English

### 1. Create your own application

1. Open the [Discord Developer Portal](https://discord.com/developers/applications) and create a
   dedicated Application. Never use user-account tokens, self-bots or another project's token.
2. Copy General Information → Application ID into local `DISCORD_APPLICATION_ID`; this is not a token.
3. On Bot, create the Bot if needed, then Reset Token. Store it only as `DISCORD_TOKEN` in the
   ignored, access-restricted local `.env`. Never put it in chat, screenshots, issues or Git.
4. Enable **Guild Install** under Installation. Select `bot` and `applications.commands` scopes
   and use the application's installation link to invite it to the intended server.
5. The installer needs server-management/application-install authority. The Bot does not need
   Administrator or Manage Guild. The caller of `/assistant join` and `/wakeup set|reset` needs
   Manage Guild; that is a user permission, not a permission to grant the Bot.
6. Do not enable privileged Message Content, Server Members or Presence intents. Runtime uses
   Guilds, Voice States and standard Messages; a genuine direct Bot mention exposes the attachment
   used by agenda intake without privileged Message Content access.

Portal labels may change; consult the official
[getting-started guide](https://docs.discord.com/developers/quick-start/getting-started) and
[installation context](https://docs.discord.com/developers/resources/application#installation-context).

### 2. Grant channel-scoped permissions

| Scope | Permission | Purpose |
| --- | --- | --- |
| Publication text channel | View Channel, Send Messages, **Attach Files** | Panels and the final `minutes.md` attachment; missing file permission blocks delivery |
| Ordinary voice channel | View Channel, Connect, Speak | Join, receive and play voice |
| Optional enforced floor | Mute Members | Enforced ordered/presentation agendas; free/advisory mode does not require it |
| Human host/administrator | Manage Guild, or meeting-creator authority as applicable | Commands enforce their own caller checks; not a Bot-wide administration grant |

Check server roles **and target-channel overrides**. Do not use Administrator as a troubleshooting
shortcut. Use ordinary voice and text channels; Stage, forum and thread paths are not claimed as
validated here. See the [official permission definitions](https://docs.discord.com/developers/topics/permissions).

### 3. Configure and start

Follow the API guide to set `DISCORD_TOKEN`, `DISCORD_APPLICATION_ID`, `FILE_ENCRYPTION_KEY`,
your own DeepSeek key and pinned models. Use the maintained deployment launcher; starting only Bot
omits the independent Worker required for post-meeting processing.

Leave `DISCORD_TEST_GUILD_ID` blank for production global Guild-only commands. Set one development
Guild only for test sync; use Discord Developer Mode → Copy Server ID to obtain it.
Allow for global-command propagation/client refresh instead of repeatedly rebuilding or exposing tokens.

With the default resident assistant enabled, expect three top-level command groups:

- `/meeting`: meetings, agendas, consent, speaking, voting, host controls, retries and deletion.
- `/assistant`: resident Q&A `join|status|stop`, without meeting recordings or minutes.
- `/wakeup`: `show|set|reset`, independently configured per Guild; default “呼叫指挥中心”.

One Bot can be installed in multiple Guilds, but a deployment has one heavy-workload slot. A meeting
keeps it through Worker publication; resident mode keeps it until stopped. This is not an install
or token failure.

### 4. First acceptance and troubleshooting

1. Verify healthy Bot/Worker/model services and the commands above.
2. Create a meeting in the ordinary text channel that should receive minutes. Obtain explicit
   recording/ASR/minutes consent and disclose remote DeepSeek text processing before testing.
   Use synthetic content for initial acceptance.
3. Test voice join/playback and meeting stop, then confirm the Worker's final `minutes.md` attachment
   can be downloaded. Bot online/voice success alone is not publication acceptance.
4. Missing commands: check scopes, Guild context, test-Guild settings and propagation; do not enable
   unrelated privileged intents.
5. Voice join fails: check channel overrides; enforced floor also requires Mute Members.
6. Voting DMs fail: check recipient privacy settings and use the documented Guild-command fallback.
7. Processing finished without delivery: check the original channel still exists and permits Attach
   Files / Send Messages. Do not blindly retry, manually duplicate delivery or edit the database.
8. Keep all acceptance audio, text, identities, receipts and execution records in local private
   storage, not issues or the repository.

See [English usage](USAGE.en.md) for full meeting, agenda-import and recovery workflows. Markdown
intake requires the host's genuine Bot mention with one `.md` attachment in the creation channel;
typed nickname text is not a mention. Identity uses Discord IDs, never guessed display names.
