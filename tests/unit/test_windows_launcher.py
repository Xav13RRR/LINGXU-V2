from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_windows_one_click_launcher_has_safe_validated_service_contract() -> None:
    command = (ROOT / "启动聆序.cmd").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "start_lingxu.ps1").read_text(encoding="utf-8")
    credential_script = (ROOT / "scripts" / "set_deepseek_api_key.ps1").read_text(encoding="utf-8")
    failure_probe = (ROOT / "tests" / "fixtures" / "test_docker_probe_failure.ps1").read_text(
        encoding="utf-8"
    )

    assert "start_lingxu.ps1" in command
    assert "ExecutionPolicy Bypass" in command
    assert 'start_lingxu.ps1" %*' in command
    assert "[DONE] Lingxu launcher completed successfully." in command
    assert "Lingxu is online" not in command
    assert "Start-Process -FilePath $desktop -WindowStyle Hidden" in script
    docker_probe = script.split("function Test-DockerEngine", 1)[1].split(
        "function Start-DockerDesktopIfNeeded", 1
    )[0]
    assert "try {" in docker_probe
    assert "catch {" in docker_probe
    assert "return $false" in docker_probe
    assert 'Join-Path $ProjectRoot ".env"' in script
    assert '"Lingxu\\deepseek-api-key.dpapi"' in script
    assert "Add-Type -AssemblyName System.Security" in script
    assert "ProtectedData]::Unprotect" in script
    assert "DataProtectionScope]::CurrentUser" in script
    assert 'SetEnvironmentVariable("SUMMARY_LLM_API_KEY", $credential, "Process")' in script
    assert "Remove-Item Env:SUMMARY_LLM_API_KEY" in script
    assert "function Set-BuildRevisionFromRepository" in script
    assert "if (-not (Test-Path -LiteralPath $gitDirectory))" in script
    assert "Get-Command git -CommandType Application -ErrorAction SilentlyContinue |" in script
    assert "Select-Object -First 1" in script
    assert "rev-parse --verify HEAD" in script
    assert "status --porcelain --untracked-files=normal" in script
    assert 'SetEnvironmentVariable("LINGXU_BUILD_REVISION", $revisionLabel, "Process")' in script
    assert (
        "Set-BuildRevisionFromRepository"
        in script.split('Invoke-Docker -Arguments ($ComposeArguments + @("config", "--quiet"))', 1)[
            0
        ]
    )
    assert "Remove-Item Env:LINGXU_BUILD_REVISION" in script
    assert '"DISCORD_TOKEN"' in script
    assert '"FILE_ENCRYPTION_KEY"' in script
    required_settings = script.split("$RequiredSettings = @(", 1)[1].split(")", 1)[0]
    assert "DISCORD_TEST_GUILD_ID" not in required_settings
    assert '"docker-compose.gpu.yml"' in script
    assert "[switch]$RebuildAll" in script
    assert "function Get-TtsBuildFingerprint" in script
    assert "function Test-TtsImageCurrent" in script
    tts_probe = script.split("function Test-TtsImageCurrent", 1)[1].split(
        "function Set-BuildRevisionFromRepository", 1
    )[0]
    assert "{{json .Config.Labels}}" in tts_probe
    assert 'PSObject.Properties["io.lingxu.tts.build-fingerprint"]' in tts_probe
    assert "{{ index .Config.Labels" not in tts_probe
    assert '@("build", "lingxu-migrate")' in script
    assert '@("build", "lingxu-tts")' in script
    assert '@("up", "-d", "--no-build")' in script
    assert '@("up", "-d", "--build")' not in script
    assert '"lingxu-bot"' in script
    assert '"lingxu-worker"' in script
    assert '"lingxu-ollama"' in script
    assert '"lingxu-tts"' in script
    assert '"--strict-discord"' in script
    assert '"bot_ready"' in script
    assert "logs --no-color --since $startedAt lingxu-bot" in script
    assert "simulated missing dockerDesktopLinuxEngine pipe" in failure_probe
    assert "if (Test-DockerEngine)" in failure_probe
    assert "docker compose logs" not in command
    assert 'Read-Host "DeepSeek API key" -AsSecureString' in credential_script
    assert "Add-Type -AssemblyName System.Security" in credential_script
    assert "ProtectedData]::Protect" in credential_script
    assert "DataProtectionScope]::CurrentUser" in credential_script
    assert "WriteAllText($credentialFile, $hex" in credential_script
    assert "Write-Host $credential" not in credential_script
    assert "function Assert-DeploymentIdle" in script
    assert "function Assert-RunningDeploymentIdleIfAvailable" not in script
    assert "function Get-ExistingDeploymentActivity" in script
    assert "function Test-DeploymentActivity" in script
    assert "function Resume-ExistingDeployment" in script
    assert '"source=lingxu-data,target=/data"' in script
    assert "$containerIds.Count -gt 1" in script
    assert "$properties.Count -ne 2" in script
    assert '"System.Int32", "System.Int64"' in script
    assert 'Invoke-Docker -Arguments @("start", $containerId)' in script
    assert '($ComposeArguments + @("start") + $ExpectedServices)' not in script
    assert "existing deployment was recovered without rebuild or replacement" in script
    assert (
        "Deployment became active during image preparation; refusing service replacement." in script
    )
    assert '"/app/scripts/check_deployment_idle.py"' in script
    activity_probe = script.index("$existingActivity = Get-ExistingDeploymentActivity")
    app_build = script.index('@("build", "lingxu-migrate")')
    idle_guard = script.index("Assert-DeploymentIdle\n", app_build)
    service_start = script.index('@("up", "-d", "--no-build")')
    assert activity_probe < app_build
    assert app_build < idle_guard < service_start
    invoke_docker = script.split("function Invoke-Docker", 1)[1].split(
        "function Test-DockerEngine", 1
    )[0]
    assert '$ErrorActionPreference = "Continue"' in invoke_docker
    assert "Docker command failed with exit code $exitCode." in invoke_docker
    assert "Remove-Item Env:LINGXU_TTS_BUILD_FINGERPRINT" in script
    dry_run = script.split("if ($DryRun)", 1)[1].split("Assert-DeploymentIdle", 1)[0]
    assert "return" in dry_run
