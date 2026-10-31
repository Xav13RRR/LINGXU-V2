# 聆序 LINGXU 使用指南

本指南说明当前 Discord 版本的两组入口：正式会议 `/meeting` 和常驻语音问答
`/assistant`。同一 Guild 同时只能由一个模式占用 Bot 的语音连接；先结束当前会议或助手，
再启动另一个模式。

[English version](USAGE.en.md) · [返回 README](../README.md) · [中文部署指南](DEPLOYMENT.zh-CN.md)

## 1. 使用前检查

- Bot 在线，并能在目标语音频道 View Channel、Connect、Speak；
- Bot 能在相关文字频道 View Channel、Send Messages；
- Slash Commands 中可见 `/meeting`、`/assistant`；
- 主持人必须位于计划使用的普通语音频道；
- 需要接收投票私信的参与者应允许服务器成员私信，关闭私信时可使用服务器内
  `/meeting vote cast` 降级入口；
- 真实录音、ASR 和纪要需要参与者在本 Guild 明确运行 `/meeting consent`。

## 2. 完整会议流程

### 2.1 创建会议

主持人在会议准备文字频道或目标语音频道的内置文字聊天运行：

```text
/meeting create
```

Bot 创建一个会议 Session 并发布主持控制面板。每个 Guild 同时只有一个活动会议；会议
创建者成为主持人。若创建时主持人已在语音频道，当前非 Bot 成员会进入参与者状态。

### 2.2 添加议程：Slash Command

```text
/meeting agenda add title:<议程名称> duration:<分钟> requires_vote:<true|false> language:<auto|zh|yue|en|ja|ko>
```

- `title`：Bot 进入该议程时会按所选语言播报；
- `duration`：该议程讨论阶段的总分钟数，不包括投票；留空表示无固定时长，主持人手动
  使用 `/meeting next`；
- `requires_vote=false`：议程完成后不进入投票；
- `requires_vote=true`：Discord 弹出表单，必须填写投票内容和 10–3600 秒投票时长；
- `language=auto`：自动选择；也可明确指定中文、粤语、英文、日文或韩文。

管理议程：

```text
/meeting agenda list
/meeting agenda remove position:<序号>
/meeting agenda reorder from_position:<原序号> to_position:<新序号>
```

只能在会议尚未开始时删除和调整待处理议程。

内置受控议程模板：

```text
/meeting agenda template kind:<round_robin|standup|pre_vote_statements|alternating_sides|presentation|presentation_q_and_a>
```

圆桌、站会和表决前陈述按 Discord 显示名的 NFKC/case-fold 顺序建立快照；`alternating_sides`
要求选择两名不同成员；演讲模板要求选择一至两名演讲者。可设置总时长、每轮秒数、准备
秒数、补充/答疑环节与严格静音。只有显式选择 `presentation_q_and_a` 或 `followup` 时才加入
后续环节。若启用严格静音，开始前必须给聆序 `Mute Members` 权限；普通语音频道是唯一
支持的频道模式。主持人可以是被选成员或演讲者，但主持权限本身不会自动获得发言权。

### 2.3 添加议程：Markdown 文档

会议创建后，也可以在执行 `/meeting create` 的同一频道中：

1. 使用 Discord 真实 Mention 选择 Bot；
2. 在同一条消息中只附加一份 `.md` 文件；
3. 等待 Bot 回复“已收到议程文档”；
4. 处理成功后 Bot 回复“准备完毕”、议程数量和总时长。

该入口只接受会议主持人、空议程、尚未开始的会议以及固定大小范围内的单一 Markdown
附件。Bot 使用不可变用户 ID 识别 Mention，修改 Bot 的全局名称或服务器昵称不会影响。
纯手打的 `@名字` 不是 Discord Mention。

本地 `qwen3.5:9b` 按议程导入规则归纳标题、顺序、语言、明确的投票语义和时长。文档未给出
讨论时长时使用固定 15 分钟，明确需要投票但未给出投票时长时使用固定 60 秒。模型无
Discord、数据库或工具权限；所有候选经过来源、范围、数量、语言、投票和置信度验证后才
一次性写入，不会部分导入。

若要让导入文件直接扩展上述模板、完全不调用议程模型，可在 `.md` 中加入唯一一个封闭
JSON 区块：

导入前仅在本地文件中将 `PRESENTER_USER_ID` 和 `OTHER_USER_ID` 替换为目标成员的 ID。
它们是占位符而非可用 Mention；填写真实身份后的文件不得提交。

````markdown
```lingxu-agenda
{
  "schema_version": 1,
  "identity_map": {"主讲": "<@PRESENTER_USER_ID>"},
  "items": [
    {
      "template": "PRESENTATION_Q_AND_A",
      "title": "产品演讲",
      "duration_minutes": 20,
      "presenters": ["主讲"],
      "enforce_mute": true
    },
    {
      "template": "ALTERNATING_SIDES",
      "members": ["主讲", "<@OTHER_USER_ID>"],
      "turn_seconds": 120
    }
  ]
}
```
````

成员只接受真实 Discord mention，或由主持人在 `identity_map` 显式映射到 mention；自然姓名
永远不会自动匹配。聆序在提交前向 Discord 验证每个引用确属当前服务器且不是 Bot。未知
字段、未映射姓名、无效/重复成员或不适用于模板的参数都会使整份导入失败，不会部分写入。

### 2.4 参与者同意与撤回

每位参与者首次在一个 Guild 中运行：

```text
/meeting consent
```

该命令同时明确同意实时 ASR、持续录音和会后纪要三项用途。授权仅在当前 Discord Guild
内跨后续会议复用，不会带到其它服务器。

随时撤回：

```text
/meeting withdraw
```

撤回立即阻止该用户后续帧进入对应处理，并清除本 Guild 的授权；即使当前没有会议也可
撤回。撤回不会伪造或删除已发生的追加式审计事件。

每位用户也可管理仅用于本 Guild 会议 TTS 的姓名读音：

```text
/meeting pronunciation set alias:<希望聆序读出的名称>
/meeting pronunciation show
/meeting pronunciation clear
```

读音别名不会改变 Discord 身份、显示名、首字母排序、公开状态、逐字稿说话人或纪要事实；
文字回退与持久化的 Bot 提示仍使用规范 Discord 显示名。

### 2.5 开始会议

主持人进入目标普通语音频道后运行：

```text
/meeting start
```

Bot 按以下顺序处理：

1. 校验主持人、会议、议程、权限和未完成的议程导入；
2. 将当前非 Bot 语音成员同步为参与者；
3. 进入主持人所在频道并确认入站 DAVE；
4. 启动分用户录音、实时 ASR 和语音助手；
5. 检查当前参与者 consent；
6. 播报“本次会议将于十秒后开始”；
7. 十秒后自动播报议程数量、标题、语言和阶段时长并进入第一项。

如果有人未授权，状态保持 `CONSENT_CHECK`。所有当前参与者完成授权，或未授权成员离开
频道后，系统重新计算并自动推进。

### 2.6 自动流程和主持控制

有固定时长的流程到期后自动进入下一阶段。无固定时长时，Bot 会提示主持人手动推进。
仅原始总时长大于 120 秒的流程在约剩余 60 秒时提醒一次；短流程、投票和开场不会密集
播报倒计时。

主持人命令：

```text
/meeting status
/meeting pause
/meeting resume
/meeting next
/meeting back
/meeting skip
/meeting stop
```

主持面板也提供暂停、继续、下一步和停止按钮。`back` 通过新事件表达回退，不删除历史；
LLM 不可直接执行任何状态变更，所有操作仍经过确定性 Controller 和主持权限守卫。

### 2.7 发言队列

参与者申请发言：

```text
/meeting speak
```

重复申请不会重复入队。主持人选择下一位：

```text
/meeting speaker next
```

当前发言者或主持人结束本轮：

```text
/meeting speaker finish
```

主持人移除等待者：

```text
/meeting speaker remove member:<成员>
```

### 2.8 议程内私信投票

当 `requires_vote=true` 的议程讨论结束：

1. Bot 播报投票内容和有效时长；
2. 对投票开始时仍在频道内的每名非 Bot 参与者建立选民快照；
3. 向每名选民私信发送“是/否”按钮；
4. 截止前可再次点击修改选择；
5. 全员完成或时间到期后关闭投票，未响应者计为弃权；
6. Bot 播报是、否、弃权人数和“是/否/平票”结果；
7. 约五秒后自动进入下一议程。

关闭私信或按钮不可用时，在会议 Guild 内运行：

```text
/meeting vote cast option:<yes|no>
```

该入口仍验证选民快照。加入过晚的成员不能进入本轮；已在快照中的成员离开频道后仍可
投票。最终纪要只展示汇总和结论，不展示个人选择映射。

### 2.9 独立公开投票

主持人也可手工创建公开投票：

```text
/meeting vote open title:<问题> duration_seconds:<10-3600>
/meeting vote status
/meeting vote close
```

公开投票卡包含赞成、反对、弃权按钮；此功能独立于议程内私信 Yes/No 投票。
独立公开投票仍开启时，`next`、`back` 和 `skip` 会拒绝改变会议流程；主持人须先运行
`/meeting vote close`，或等待该投票到期关闭。

### 2.10 会议中的实时语音问答

唤醒词按 Discord 服务器隔离。在该服务器的任意频道运行：

```text
/wakeup show
/wakeup set phrase:<连续 4-16 个汉字>
/wakeup reset
```

所有服务器成员都可以使用 `show`；`set` 和 `reset` 需要 Manage Guild。默认值是
“呼叫指挥中心”，`reset` 会删除本服务器覆盖并恢复默认值。设置会先规范化并通过固定本地
KWS 模型的读音/词表预检，然后同时用于该服务器的会议助手和常驻助手；活动会话无需重启，
其它服务器不受影响。命令响应为私密消息，运行日志不记录唤醒词值。

频道内任意非 Bot 成员随后可以说：

```text
<本服务器当前唤醒词>，<问题>
```

唤醒词可以在句首或句中，ASR 文本确认允许受限近似音节以适应发音不清。也可只说当前
唤醒词，再在四秒内补充问题。系统会在声学命中后等待同一用户说完，再进行 ASR、调用本地
`qwen3.5:9b` 并播报最多两句、最长约 15 秒的回答。

- 同一时间只有一个回答回合；
- 模型处理或 Bot 播报期间的新唤醒被丢弃，不排队；
- 真人开始讲话不会中断 Bot，Bot 播完后才重新接受调用；
- 该即时问答路径不要求 consent，因为 PCM、问题和回答均不持久化；
- 正式录音、逐字稿和纪要仍各自遵循 consent；
- 模型没有工具、投票、计时、Discord 或会议状态修改权限。

### 2.11 结束与会议材料

最后一项完成后系统自动进入结束流程，也可以由主持人运行：

```text
/meeting stop
```

Bot 播报会议结束和总结将发送至语音频道文字聊天，然后依次停止接收、排空 ASR、完成
加密 Segment、断开语音并创建独立 Worker Job。当前正常发布合同向 Discord 发送且只发送
一个附件：

- `minutes.md`：适合阅读的会议纪要，内含完整可读的规范化逐字稿。

Worker 仍在本地登记 `minutes.json`、`transcript.txt`、规范化逐字稿 JSON、事件和
`evidence.json`，用于一致性校验、恢复与审计；它们不是当前正常 Discord 附件。

项目所有者批准的官方 DeepSeek API 接收完整已授权记录的紧凑数据包：真人发言、系统上下文、
说话人字典，以及议程和聚合投票背景。模型返回结构化内容与证据编号，由 Worker 校验后在本地
排版为 `minutes.md`；未修订的来源记录原样保留，完整权威表格不再依赖模型照抄。
这会减少格式负担，但不能代替事实核对。发言流程表由 append-only 事件确定性计算，
列出计划/延长时长、扣除会议暂停/权限恢复/主持介入后的实际发言地板占用，以及提前结束、
跳过、全部或部分缺席、超时和已配置的补充发言/答疑参与。确认“下一项”或停止会议时会先在
事件流中闭合当前地板；受控议程进行中拒绝通用的返回/跳过状态命令。未获 `MINUTES` 授权的身份会在保留流程事实的
同时删除 Discord ID 和显示名。发布前，本地代码逐行校验模板、说话者清单、议程表、权威发言
表和投票表；模型不能增加、删除、改序或修改这些事实，也不能覆盖会议状态。Provider、超时
或输出合同失败时，Worker 会自动生成明确标注的基础纪要，继续完成校验和发布。
基础纪要包含议程、投票、发言流程、原始发言摘录和完整已授权记录；决策、共识和行动事项
标记为待人工核对，不采用失败的模型文本。原始模型失败仍保留在内部审计记录中，不会自动
再次请求模型。重启后复用已验证的基础纪要，已发布文件不重复发送。
数据损坏、隐私校验失败、存储故障或 Discord 发布异常仍可能阻止交付；修复原因后，主持人
可对失败任务运行 `/meeting retry`。已完成发布的基础纪要不会由此自动重新生成。

会后 Job 失败时由主持人运行：

```text
/meeting retry
```

删除最新会议及其数据：

```text
/meeting delete
```

删除会真正移除数据库图和精确 Session 目录。操作不可恢复，重要材料应先由管理员备份。

## 3. 常驻语音指挥中心

常驻模式不运行任何会议功能。具有 Manage Guild 权限的成员先进入目标语音频道，再运行：

```text
/assistant join
```

之后频道内任意非 Bot 成员都可用本服务器当前唤醒词提问；可用 `/wakeup show` 查看。
行为与会议问答相同，但模型只收到当前问题和固定模式标记，不收到会议状态、频道消息、
成员列表或逐字稿。会议模式和常驻模式始终共用同一个服务器级设置。

```text
/assistant status
/assistant stop
```

- `/assistant status` 私密显示内容安全的会话和频道状态；
- 启动者或 Manage Guild 成员可停止；
- Bot 被移动或移出频道时会自动停止，而不会跟随到新频道；
- 正常进程重启会根据仅含 Guild/频道/创建者/生命周期的记录恢复；
- 不录音、不生成逐字稿、纪要、议程、投票或 Worker Job；
- PCM、问题、回答和模型输出只存在于内存；动态 WAV 位于 tmpfs，播完删除。

## 4. 本地主持台

启用后在 Bot 主机浏览器打开：

```text
http://127.0.0.1:8765
```

主持台提供单一 Guild 的实时状态、议程、参与者同意、发言队列、计时器、投票、事件、
手工 TTS 和场景模拟。令牌只能在本机终端使用
`docker compose exec -T lingxu-bot python -m lingxu.console.token` 查看。不要公开令牌或把
主持台绑定到公网接口。

## 5. 常见问题

### 看不到 Slash Commands

确认 `.env` 中 `DISCORD_APPLICATION_ID` 正确，Bot 安装包含 `applications.commands` scope，
然后检查日志中的 `application_commands_synced` 和 `bot_ready`。正式多服务器部署应将
`DISCORD_TEST_GUILD_ID` 留空并看到 `scope=global`；开发环境填写 Guild ID 时，命令只会出现
在该测试服务器。全局同步后可刷新 Discord 客户端并短暂等待。

### 显示“服务容量已满”

当前本机正在为另一 Discord 服务器运行会议/会后发布、常驻助手或议程模型处理。
等待其完成或由对应主持人停止后重试。为保护隐私，提示不会显示占用服务器或用户。你仍可
先创建会议和查看轻量状态；`/meeting start` 等重型入口会原子争用唯一槽位。

### Bot 无法进入语音频道

确认调用者本人在普通语音频道，Bot 有 View Channel、Connect、Speak，并且同 Guild 没有
另一个 `/meeting` 或 `/assistant` 占用语音连接，同时本机槽位未被其它 Guild
占用。

### 会议卡在 consent_check

让所有当前非 Bot 参与者在本 Guild 运行 `/meeting consent`，或让不参与且未授权的成员
离开频道。授权按 Guild 隔离；其它服务器中的授权不能复用。

### 唤醒没有响应

先运行 `/wakeup show` 确认本服务器当前值。避免在 Bot 自己播报或播报结束后的短隔离窗口
中呼叫，并完整说出当前唤醒词；可以句中使用，但只说其中一部分不会被当作完整唤醒。
检查模型、ASR、KWS Doctor 结果以及内容安全的聚合计数，不要记录或上传用户音频。若自定义
读音不被固定模型词表支持，`/wakeup set` 会拒绝保存；可用 `/wakeup reset` 恢复默认值。

### 没有收到纪要

检查会议是否已进入会后处理、Worker 是否 healthy、是否存在失败 Job；主持人可运行
`/meeting retry`。附件发布到会议实际使用的语音频道内置文字聊天，而不是随意选择的频道。

更多故障恢复见 [OPERATIONS.md](OPERATIONS.md)、[FAILURE_RECOVERY.md](FAILURE_RECOVERY.md)
和 [TEST_REPORT.md](TEST_REPORT.md)。
