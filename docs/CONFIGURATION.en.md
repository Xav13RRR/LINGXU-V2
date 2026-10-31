# API and credential configuration

[中文](CONFIGURATION.zh-CN.md) · [Deployment](DEPLOYMENT.en.md) · [Discord setup](DISCORD_SETUP.md)

Use your own independent deployment, credentials and data. You do not need the maintainer's keys
or meetings. Never overwrite an existing `.env` or regenerate the key used by existing encrypted data.

## 1. First-install checklist

For a new installation only, copy `.env.example` to `.env` in the repository root. Edit values in
a local editor, not in terminal commands, chat, issues, screenshots, recordings or shell history.

| Setting | Required? | Source and storage |
| --- | --- | --- |
| `DISCORD_TOKEN` | Yes | Developer Portal → your Application → Bot → Reset Token; local `.env` only |
| `DISCORD_APPLICATION_ID` | Yes; not secret | Developer Portal → General Information → Application ID; `.env` |
| `FILE_ENCRYPTION_KEY` | Yes | Generate a URL-safe Base64-encoded random 32-byte key once per new deployment; `.env` plus a separate secure backup |
| `SUMMARY_LLM_API_KEY` | Yes | Your DeepSeek API account; on Windows use the masked script below and leave this blank in `.env` |
| `DISCORD_TEST_GUILD_ID` | No | Blank for production; one Guild ID only for immediate development-Guild command sync |
| `HOST_CONSOLE_ENABLED` / `HOST_CONSOLE_GUILD_ID` | Optional | Enable the localhost console for one explicit Guild only; do not expose it publicly |
| `HOST_CONSOLE_TOKEN` | Optional | Derived from the file-encryption key when blank; see deployment instructions; never upload or share |
| `LLM_API_KEY` / `MEETING_ASSISTANT_LLM_API_KEY` / `AGENDA_IMPORT_LLM_API_KEY` | Not for the default | Leave blank for the pinned local Ollama profile; do not put the DeepSeek key here |

This application uses **`SUMMARY_LLM_API_KEY`**, not the `DEEPSEEK_API_KEY` name in upstream examples.
Keep `SUMMARY_LLM_BASE_URL=https://api.deepseek.com`, `SUMMARY_LLM_MODEL=deepseek-flash`, the model
digest and other defaults from `.env.example`; the application validates this model configuration.

## 2. Obtain your own DeepSeek API key

1. Follow the API-key link in the [official documentation](https://api-docs.deepseek.com/) to the
   [DeepSeek platform](https://platform.deepseek.com/). Create a separate key for your deployment.
2. Check the API account balance and usage limits; do not assume a chat-product subscription proves
   API credit is available.
3. Read the [privacy boundary](PRIVACY.md): the Worker sends consent-filtered meeting text and
   permitted speaker labels to remote DeepSeek and incurs API charges. Minutes are not fully offline.
4. Do not test a new key with a real meeting. Use synthetic content with every participant's consent
   for the first end-to-end test, and keep results local. Configuration preflight does not prove API
   authentication, credit or successful inference.

## 3. Recommended Windows path: DPAPI and the maintained launcher

Run under the same Windows account that will operate Lingxu:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/set_deepseek_api_key.ps1
```

Enter the key at the masked prompt. The account-bound ciphertext is stored outside the repository
at `%LOCALAPPDATA%\Lingxu\deepseek-api-key.dpapi`. The script neither starts services nor edits
`.env`. For a new installation, leave `SUMMARY_LLM_API_KEY` blank in `.env` and unset in the process
environment: the launcher uses an existing environment/`.env` value before consulting DPAPI.

After downloading the models described in the deployment guide, run configuration preflight:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_lingxu.ps1 -DryRun
```

It does not rebuild/replace Lingxu services or run inference, but may start Docker Desktop if needed.
For first deployment, then run `启动聆序.cmd` (use `-Cpu` for CPU mode). Existing installations must
follow the idle-check, backup and upgrade procedure; do not restart merely to check a key.

**Do not switch directly to `docker compose up` after configuring DPAPI.** Compose cannot decrypt
that file. The launcher passes the credential to the Worker container; its exit cleanup clears only
the host-process variable it injected, **not** the running container environment. Docker administrators
can inspect container configuration and must be trusted with credentials.

## 4. Manual Compose / non-Windows hosts

The DPAPI script is Windows-only. For manual Linux/macOS deployment, have your own secret manager
inject `SUMMARY_LLM_API_KEY` into the Compose process, or enter it using a local editor in the ignored,
access-restricted `.env`. The latter is plaintext, not encrypted storage. Use `chmod 600 .env` on Unix;
on Windows restrict file access to the operating account and necessary administrators. Avoid shared
or automatically cloud-synchronized directories.

Use the deployment guide's manual commands only after providing credentials, downloading models,
checking for active work and backing up existing data. Compose process variables override `.env`;
check for stale overrides without printing values. Only the pinned provider/model is supported.
A fresh non-Windows installation still needs independent acceptance; workstation GPU validation is
not evidence of validation on another host.

## 5. Generate the encryption key and update credentials

For a new deployment, PowerShell's built-in random generator avoids installing the Python stack:

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

Copy the output only into local `.env` as `FILE_ENCRYPTION_KEY`. Do not run in a recorded, shared or
transcribed terminal. Losing this key makes old encrypted data unrecoverable; changing the setting
does not rotate existing data.

- DeepSeek update: rerun the masked script, remove stale overrides, then recreate affected containers
  during a maintenance window. The script does not update running containers. Validate the new
  credential before revoking the old key in the provider console.
- Discord update: reset the Bot token, update local configuration, and update Bot/Worker through the
  maintenance procedure.
- Do not regenerate `FILE_ENCRYPTION_KEY` when updating an API key; retain it for existing data.
- On suspected compromise, restrict access and revoke affected access credentials with the provider.
  The data-encryption root requires a separate migration, not an in-place setting replacement.

## 6. Safe troubleshooting and submission checks

| Symptom | Check |
| --- | --- |
| DeepSeek credential is missing | Run the masked script under the operating account and use the launcher, or provide manual Compose credentials |
| DPAPI decryption fails | Use the original account, or enter your own key under the new account; do not copy someone else's ciphertext |
| API authentication, credit or rate-limit failure | Check the provider console; never paste headers, response bodies or meeting content |
| Model identity mismatch | Check defaults and stale overrides; do not bypass checks, invent digests or automatically change models |
| Minutes cannot be published | Check View Channel, Send Messages, Attach Files and target-channel permission overrides |

```powershell
git check-ignore .env
uv run python scripts/scan_secrets.py
uv run python -m scripts.check_public_files
```

Use only `docker compose config --quiet`. Never share rendered Compose configuration, `docker inspect`,
environment dumps, `.env`, DPAPI files, databases, audio, transcripts, minutes or execution records.
Passing checks cover only their stated scope, not all history or semantic private content. See
[SECURITY.md](../SECURITY.md).
