# Approved Models

These models implement Phase 4 realtime ASR. They are not LLMs and do not select or imply the
later Phase 12 LLM provider/model decision.

| Role | Model | Hub revision | Implementation-time repository HEAD | Weight SHA-256 | Licence |
| --- | --- | --- | --- | --- | --- |
| Multilingual sentence ASR | `iic/SenseVoiceSmall` | `master` | `7bf452403abd7353a300cd760f7adae7701c92c1` | `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea` | Apache-2.0 |
| Voice activity detection | `fsmn-vad` / `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | `v2.0.4` | Tagged revision | `b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5` | Apache-2.0 |

SenseVoiceSmall exposes no release tag in ModelScope. Lingxu therefore records the remote Git HEAD
observed at implementation time and refuses to use a downloaded `model.pt` whose SHA-256 differs
from the approved value. A future upstream change requires an explicit model review, hash update,
licence recheck, and full multilingual/latency replay.

The official model repository supplies the acceptance fixtures:

| Fixture | SHA-256 |
| --- | --- |
| `example/zh.mp3` | `0e64de19e4ff9a02e682955c9112f32d2317cfdbb5bc2f3504664044c993f195` |
| `example/en.mp3` | `f10378336a4e584f3f63799e62f99d5add3c2a401b51d3abe7d3a3a82f255ada` |

The mixed fixture is constructed in memory by concatenating the decoded Chinese fixture, 500 ms of
silence, and the decoded English fixture. Neither fixture audio nor transcript text is committed to
the repository or written to application logs.

## Meeting wake-word model

| Role | Model | Runtime | Approved archive SHA-256 | Selected runtime-file digest | Licence |
| --- | --- | --- | --- | --- | --- |
| Open-vocabulary Mandarin/English keyword spotting | `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20` | `sherpa-onnx==1.13.4` + `sherpa-onnx-core==1.13.4`, CPU, int8 chunk-8 encoder/joiner | `68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6` | `9604d8412a76c099ef8eecedb95fa04d523cb16df2f6a6be724bc956a92870ef` | sherpa-onnx Apache-2.0 release asset |

The selected digest binds the chunk-8 int8 encoder, fp32 decoder, int8 joiner, and token file by
filename and bytes. The immutable default manifest contains `呼叫指挥中心` plus only its six
one-syllable-deletion variants. Each Guild may persist one normalized custom phrase through
`/wakeup`; before persistence, runtime derives its Mandarin token path and rejects any token absent
from the pinned vocabulary. A custom phrase uses one exact acoustic path in a digest-named tmpfs
manifest. Runtime verifies the model digest before creating or replacing a spotter, and an active
assistant converges without restart. The downloader refuses partial or changed assets and never
overwrites a mismatched installed model.
Unconsented caller audio reaches this local detector and, only after a valid hit, the existing
SenseVoice provider in memory; it is never persisted by the assistant path.

## Local runtime LLM

| Role | Model | Runtime format | Parameters / quantization | Ollama manifest digest | Licence |
| --- | --- | --- | --- | --- | --- |
| Allowlisted meeting-intent candidate | `qwen3.5:9b` | GGUF through Ollama `0.21.2` | 9.7B / Q4_K_M | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` | Apache-2.0 |
| Meeting realtime short answer | `qwen3.5:9b` | GGUF through Ollama `0.21.2` native structured output | 9.7B / Q4_K_M, 64-token maximum | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` | Apache-2.0 |
| Resident voice-channel short answer | `qwen3.5:9b` | GGUF through Ollama `0.21.2` native structured output | 9.7B / Q4_K_M, 64-token maximum | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` | Apache-2.0 |
| M1+P2 Markdown agenda import | `qwen3.5:9b` | GGUF through Ollama `0.21.2` native structured output | 9.7B / Q4_K_M, 1,536-token maximum per bounded chunk | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` | Apache-2.0 |

The local manifest is 6,594,474,711 bytes. Production sets one loaded model, one parallel request,
4,096-token context, flash attention, cloud disabled, and no published API port. The Bot verifies
the full manifest digest before every provider request. A model update requires explicit owner
approval, licence review, new digest, schema/injection regression, real local inference smoke, and
deployment health validation.

For structured intent, the model receives only one untrusted utterance, public Meeting state, and
the request-specific actor-authorized intent enum. The local model has no post-meeting summary,
topic, outcome, or final-minutes role. It has no tools and is not used for vote tallying, state
selection, database writes, TTS, or raw transcript replacement.

For M1+P2 Markdown import, the Bot sends only bounded, line-identified blocks from the host's
attachment. The closed schema requires source-block references and per-field provenance. Lingxu
rejects unknown references, low confidence, conflicting structure, unsupported multi-option votes,
invalid languages, excess items, and incomplete vote definitions. Deterministic P2 defaults replace
model-supplied defaults: 15 minutes per agenda and 60 seconds per vote. The model proposes a complete
candidate list but cannot write agenda rows; the repository performs one guarded atomic commit.

For realtime answers, the Bot sends one ephemeral question and bounded public Meeting status. The
schema permits only an answer and response language; thinking and tools are disabled. Post-model
limits enforce two sentences, at most 60 CJK characters or 35 English words, and at most 15 seconds
of synthesized playback. The model cannot invoke the Controller, event store, database, timers,
votes, or Discord APIs.

The resident answer role reuses the exact same digest-pinned provider and output limits but receives
only one ephemeral question and the fixed marker `resident_voice_qa`. It receives no Meeting state,
channel messages, member list, transcript, audio, or repository data. Its separate prompt forbids
tools, Discord operations, mode control, and state mutation.

## Official final-minutes model

| Role | Provider/model | Approved identity digest | Request policy | Data boundary |
| --- | --- | --- | --- | --- |
| Post-meeting topic confirmation and outcome candidates | Official DeepSeek API / `deepseek-flash` | `4a87243abc84b3147f3e9c2fc3416e539ae1ebd634706612996fc62018d32a0f` | closed JSON; bounded non-thinking initial request; high-thinking bounded repair only | Minutes-consented transcript partitions and content-minimal authoritative Meeting facts |
| Complete-transcript final minutes | Official DeepSeek API / `deepseek-flash` | `4a87243abc84b3147f3e9c2fc3416e539ae1ebd634706612996fc62018d32a0f` | thinking-high; JSON-object response; temperature omitted; explicit `max_tokens=131072`; 1,200-second total timeout; zero automatic retries | Compact complete consented source rows, speaker dictionary, separate system context, agenda/aggregate-vote context and closed output schema; local Markdown/authority composition |

This is the Worker's only post-meeting model and the final-publication path; there is no provider
selector or local-Qwen summary implementation. ADR-0164 adds a deterministic basic-minutes renderer
for final-generation failures; it does not invoke another model. The digest
is the repository-approved policy identity used by configuration and provider validation; it is not
a downloadable local weight hash. The API key is loaded by the Windows launcher from current-user
DPAPI storage into the child process environment and is never committed or logged.

As of 2026-09-12, `deepseek-flash` is the official API ID for DeepSeek V4.1 Flash (ADR-0167).
The policy digest identifies this approved contract, not immutable remote weights; the provider
controls the alias. Existing prompts, endpoint, thinking/token/timeout limits and retry counts are
unchanged. Historical records retain their original model labels. If a local environment overrides
`SUMMARY_LLM_MODEL` or `SUMMARY_LLM_MODEL_DIGEST`, update both to the values above before startup.
See the [official model mapping](https://api-docs.deepseek.com/quick_start/pricing/).

Requests exclude raw audio, credentials, DAVE/MLS material, local paths, role-private content,
individual voter mappings, and anonymous-vote tokens. Local validation requires the exact template,
speaker inventory, agenda rows, and aggregate vote rows before publication. Provider or contract
failure is recorded in the attempt journal and produces labelled basic minutes from saved facts and
consented records. There is no automatic model retry. Input/privacy/storage corruption and Discord
publication ambiguity still prevent delivery; a host can explicitly retry a failed Job.

## Local Mandarin TTS

| Role | Model | Hub revision | Upstream code commit | Aggregate approved-file digest | Default voice | Licence |
| --- | --- | --- | --- | --- | --- | --- |
| Meeting speech synthesis | `FunAudioLLM/CosyVoice-300M-SFT` | `fbb71de2afe387ed854eebd80b9f3d078c6b9869` | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` | `3a80af814cfd832dadbf1fe677adf08e30f64b8bdcd52b997184515fb33d2198` | `中文女` | Apache-2.0 |

The aggregate digest covers the exact name, byte size, and SHA-256 of `campplus.onnx`,
`cosyvoice.yaml`, `flow.pt`, `hift.pt`, `llm.pt`, `speech_tokenizer_v1.onnx`, and `spk2info.pt`.
The downloader accepts only the pinned repository and immutable revision; the serving container
verifies every file before model load. Lingxu exposes only the built-in `中文女` SFT speaker.
Changing the model, snapshot, upstream code, approved files, or voice requires explicit owner
approval, licence review, new digests, intelligibility replay, and deployment health validation.

CosyVoice's approved offline text frontend is the `pengzhendong/wetext` Chinese/English
normalisation inventory labelled `master` with aggregate digest
`a2e3a67aab0e2f9457ee61c1d3b40203d24e30d08cfdeceefdb7042f8a24a6d5`. Because that upstream
label is mutable, Lingxu does not trust the label alone: it verifies these exact files before model
load and redirects Wetext to the verified local directory so serving performs no network download.

| Text-normalisation file | Bytes | SHA-256 |
| --- | ---: | --- |
| `en/tn/tagger.fst` | 5,389,722 | `ce8b28103ccf2d936d057a5cd5621ec29ccfedb5f737e48ac8096e7d6af7b2ad` |
| `en/tn/verbalizer.fst` | 2,194,398 | `16a6a2ce28b60975e501e965396edbf94a591216a644abd27c8607a2a9aa24cd` |
| `zh/tn/tagger.fst` | 1,075,582 | `4b8b6504c7effcaa067e2fd14ecac3ae2edf9130d229e11abd0ff9b85cbe0390` |
| `zh/tn/verbalizer.fst` | 87,702 | `9fcadad76cafddb2e96b92d892d18249f0ba5fbe60e4d4fc51f466cb6bc7ade9` |

The TTS service refuses startup unless the frontend reports `wetext` and its content-free numeric
self-test converts an Arabic two-digit count to spoken Chinese. Bot-side deterministic number
verbalisation remains a defence in depth and is applied before cache-key calculation.

Sources:

- `https://modelscope.cn/models/iic/SenseVoiceSmall`
- `https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `https://github.com/modelscope/FunASR`
- `https://huggingface.co/Qwen/Qwen3.5-9B`
- `https://ollama.com/library/qwen3.5:9b`
- `https://github.com/FunAudioLLM/CosyVoice`
- `https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT`
- `https://modelscope.cn/models/pengzhendong/wetext`
- `https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html`
- `https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models`
- `https://api-docs.deepseek.com/`
