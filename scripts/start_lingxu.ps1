[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$DryRun,
    [switch]$RebuildAll,
    [ValidateRange(30, 900)]
    [int]$TimeoutSeconds = 420
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Security
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$ExpectedServices = @(
    "lingxu-bot",
    "lingxu-worker",
    "lingxu-ollama",
    "lingxu-tts"
)
$RequiredSettings = @(
    "DISCORD_TOKEN",
    "DISCORD_APPLICATION_ID",
    "FILE_ENCRYPTION_KEY"
)
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$BaseCompose = Join-Path $ProjectRoot "docker-compose.yml"
$GpuCompose = Join-Path $ProjectRoot "docker-compose.gpu.yml"
$EnvironmentFile = Join-Path $ProjectRoot ".env"
$DeepSeekCredentialFile = Join-Path $env:LOCALAPPDATA "Lingxu\deepseek-api-key.dpapi"
$script:ImportedDeepSeekCredential = $false
$script:InjectedBuildRevision = $false
$script:InjectedTtsBuildFingerprint = $false
$ComposeArguments = @("compose", "-f", $BaseCompose)
if (-not $Cpu) {
    $ComposeArguments += @("-f", $GpuCompose)
}

function Write-Step {
    param([string]$Message)

    Write-Host "[Lingxu] $Message" -ForegroundColor Cyan
}

function Invoke-Docker {
    param(
        [string[]]$Arguments,
        [switch]$Quiet
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $exitCode = 1
    try {
        # Windows PowerShell can promote a native stderr record to a terminating error before the
        # explicit exit-code check below. Keep native execution non-terminating so callers always
        # receive one content-safe, actionable Docker exit code.
        $ErrorActionPreference = "Continue"
        if ($Quiet) {
            & docker @Arguments *> $null
        }
        else {
            & docker @Arguments
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Docker command failed with exit code $exitCode."
    }
}

function Test-DockerEngine {
    try {
        & docker info --format "{{.ServerVersion}}" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        # A missing Docker Desktop named pipe is the expected pre-start state after a reboot.
        # Convert PowerShell's NativeCommandError into a normal negative probe so the caller can
        # launch Docker Desktop and wait for the engine instead of aborting the whole launcher.
        return $false
    }
}

function Start-DockerDesktopIfNeeded {
    if (Test-DockerEngine) {
        return
    }

    $candidates = @()
    if ($env:ProgramFiles) {
        $candidates += Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    }
    if ($env:LOCALAPPDATA) {
        $candidates += Join-Path $env:LOCALAPPDATA "Docker\Docker Desktop.exe"
    }
    $desktop = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $desktop) {
        throw "Docker Desktop is not running and Docker Desktop.exe was not found."
    }

    Write-Step "Starting Docker Desktop..."
    Start-Process -FilePath $desktop -WindowStyle Hidden | Out-Null
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        Start-Sleep -Seconds 2
        if (Test-DockerEngine) {
            Write-Step "Docker Desktop is ready."
            return
        }
    }
    throw "Docker Desktop did not become ready within $TimeoutSeconds seconds."
}

function Get-DotEnvValues {
    param([string]$Path)

    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($key) {
            $values[$key] = $value
        }
    }
    return $values
}

function Assert-EnvironmentReady {
    if (-not (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf)) {
        throw "Missing .env. Copy .env.example to .env and configure independent Lingxu credentials."
    }
    $values = Get-DotEnvValues -Path $EnvironmentFile
    $missing = @(
        $RequiredSettings | Where-Object {
            -not $values.ContainsKey($_) -or [string]::IsNullOrWhiteSpace($values[$_])
        }
    )
    if ($missing.Count -gt 0) {
        throw "Required .env settings are blank: $($missing -join ', ')."
    }
}

function Get-EffectiveSetting {
    param(
        [hashtable]$DotEnvValues,
        [string]$Name,
        [string]$DefaultValue
    )

    $processValue = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not [string]::IsNullOrWhiteSpace($processValue)) {
        return $processValue
    }
    if ($DotEnvValues.ContainsKey($Name) -and -not [string]::IsNullOrWhiteSpace($DotEnvValues[$Name])) {
        return $DotEnvValues[$Name]
    }
    return $DefaultValue
}

function Get-TtsBuildFingerprint {
    $inputs = @(
        Get-Item -LiteralPath (Join-Path $ProjectRoot "Dockerfile.cosyvoice")
        Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "services\cosyvoice") -File -Recurse |
            Sort-Object -Property FullName
    )
    $stream = New-Object IO.MemoryStream
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try {
        foreach ($inputFile in $inputs) {
            $relativePath = $inputFile.FullName.Substring($ProjectRoot.Length).TrimStart("\").Replace("\", "/")
            $pathBytes = [Text.Encoding]::UTF8.GetBytes($relativePath)
            $contentBytes = [IO.File]::ReadAllBytes($inputFile.FullName)
            $pathLength = [BitConverter]::GetBytes([int]$pathBytes.Length)
            $contentLength = [BitConverter]::GetBytes([long]$contentBytes.Length)
            $stream.Write($pathLength, 0, $pathLength.Length)
            $stream.Write($pathBytes, 0, $pathBytes.Length)
            $stream.Write($contentLength, 0, $contentLength.Length)
            $stream.Write($contentBytes, 0, $contentBytes.Length)
        }
        $stream.Position = 0
        $digest = $sha256.ComputeHash($stream)
        return ([BitConverter]::ToString($digest)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

function Set-TtsBuildFingerprint {
    $existing = [Environment]::GetEnvironmentVariable("LINGXU_TTS_BUILD_FINGERPRINT", "Process")
    if (-not [string]::IsNullOrWhiteSpace($existing)) {
        return
    }
    $fingerprint = Get-TtsBuildFingerprint
    [Environment]::SetEnvironmentVariable("LINGXU_TTS_BUILD_FINGERPRINT", $fingerprint, "Process")
    $script:InjectedTtsBuildFingerprint = $true
}

function Get-TtsImageReference {
    $values = Get-DotEnvValues -Path $EnvironmentFile
    $image = Get-EffectiveSetting -DotEnvValues $values -Name "LINGXU_TTS_IMAGE" -DefaultValue "lingxu-lingxu-tts"
    $tag = Get-EffectiveSetting -DotEnvValues $values -Name "LINGXU_TTS_IMAGE_TAG" -DefaultValue "latest"
    return "${image}:${tag}"
}

function Test-TtsImageCurrent {
    param([string]$ExpectedFingerprint)

    $imageReference = Get-TtsImageReference
    $labelsJson = & docker image inspect `
        --format '{{json .Config.Labels}}' `
        $imageReference 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    try {
        $labels = ([string]($labelsJson | Select-Object -First 1)) | ConvertFrom-Json
        $property = $labels.PSObject.Properties["io.lingxu.tts.build-fingerprint"]
        if ($null -eq $property) {
            return $false
        }
        return ([string]$property.Value).Trim() -eq $ExpectedFingerprint
    }
    catch {
        return $false
    }
}

function Set-BuildRevisionFromRepository {
    $existing = [Environment]::GetEnvironmentVariable("LINGXU_BUILD_REVISION", "Process")
    if (-not [string]::IsNullOrWhiteSpace($existing)) {
        return
    }

    $gitDirectory = Join-Path $ProjectRoot ".git"
    if (-not (Test-Path -LiteralPath $gitDirectory)) {
        Write-Step "Git metadata is unavailable; image revision will use the Compose fallback."
        return
    }

    $git = Get-Command git -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $git) {
        throw "Git is required to identify this repository build revision."
    }

    $revisionOutput = & $git.Source -C $ProjectRoot rev-parse --verify HEAD 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "The repository build revision could not be resolved."
    }
    $revision = ([string]($revisionOutput | Select-Object -First 1)).Trim().ToLowerInvariant()
    if ($revision -notmatch '\A[0-9a-f]{40}\z') {
        throw "The repository build revision is not a full Git commit SHA."
    }

    $dirtyOutput = @(& $git.Source -C $ProjectRoot status --porcelain --untracked-files=normal 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "The repository working-tree status could not be inspected."
    }
    $revisionLabel = if ($dirtyOutput.Count -gt 0) { "$revision-dirty" } else { $revision }
    [Environment]::SetEnvironmentVariable("LINGXU_BUILD_REVISION", $revisionLabel, "Process")
    $script:InjectedBuildRevision = $true
    if ($dirtyOutput.Count -gt 0) {
        Write-Step "Bound image metadata to the repository revision with a dirty-tree marker."
    }
    else {
        Write-Step "Bound image metadata to the clean repository revision."
    }
}

function Import-DeepSeekCredentialIfNeeded {
    $values = Get-DotEnvValues -Path $EnvironmentFile
    $summaryEnabled = (Get-EffectiveSetting -DotEnvValues $values -Name "SUMMARY_LLM_ENABLED" -DefaultValue "true").ToLowerInvariant()
    if ($summaryEnabled -notin @("1", "true", "yes", "on")) {
        throw "SUMMARY_LLM_ENABLED must remain true; the Worker has one fixed DeepSeek post-meeting path."
    }

    $existingKey = Get-EffectiveSetting -DotEnvValues $values -Name "SUMMARY_LLM_API_KEY" -DefaultValue ""
    if (-not [string]::IsNullOrWhiteSpace($existingKey)) {
        return
    }
    if (-not (Test-Path -LiteralPath $DeepSeekCredentialFile -PathType Leaf)) {
        throw "DeepSeek credential is missing. Run scripts\set_deepseek_api_key.ps1 once, then retry."
    }

    [byte[]]$protectedBytes = @()
    [byte[]]$clearBytes = @()
    try {
        $hex = [IO.File]::ReadAllText($DeepSeekCredentialFile).Trim()
        if ($hex.Length -eq 0 -or $hex.Length % 2 -ne 0 -or $hex -notmatch '\A[0-9A-Fa-f]+\z') {
            throw "The encrypted DeepSeek credential has an invalid format."
        }
        $protectedBytes = New-Object byte[] ($hex.Length / 2)
        for ($index = 0; $index -lt $protectedBytes.Length; $index++) {
            $protectedBytes[$index] = [Convert]::ToByte($hex.Substring($index * 2, 2), 16)
        }
        $clearBytes = [Security.Cryptography.ProtectedData]::Unprotect(
            $protectedBytes,
            $null,
            [Security.Cryptography.DataProtectionScope]::CurrentUser
        )
        $credential = [Text.Encoding]::Unicode.GetString($clearBytes).TrimEnd([char]0)
        if ([string]::IsNullOrWhiteSpace($credential)) {
            throw "The encrypted DeepSeek credential decrypted to an empty value."
        }
        [Environment]::SetEnvironmentVariable("SUMMARY_LLM_API_KEY", $credential, "Process")
        $credential = $null
        $script:ImportedDeepSeekCredential = $true
        Write-Step "Loaded the DeepSeek credential from Windows current-user encrypted storage."
    }
    finally {
        if ($protectedBytes.Length -gt 0) {
            [Array]::Clear($protectedBytes, 0, $protectedBytes.Length)
        }
        if ($clearBytes.Length -gt 0) {
            [Array]::Clear($clearBytes, 0, $clearBytes.Length)
        }
    }
}

function Get-ComposeRows {
    $raw = & docker @ComposeArguments ps --format json 2>$null
    if ($LASTEXITCODE -ne 0) {
        return @()
    }
    $rows = @()
    foreach ($line in @($raw)) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $rows += $line | ConvertFrom-Json
        }
        catch {
            continue
        }
    }
    return $rows
}

function Get-ServiceStatusText {
    param([object[]]$Rows)

    $byService = @{}
    foreach ($row in $Rows) {
        $byService[$row.Service] = $row
    }
    $parts = foreach ($service in $ExpectedServices) {
        if (-not $byService.ContainsKey($service)) {
            "$service=missing"
            continue
        }
        $row = $byService[$service]
        $health = if ([string]::IsNullOrWhiteSpace($row.Health)) { "none" } else { $row.Health }
        "$service=$($row.State)/$health"
    }
    return $parts -join "; "
}

function Wait-LingxuHealthy {
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $lastReport = -15.0
    while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $rows = @(Get-ComposeRows)
        $healthy = @(
            $rows | Where-Object {
                $_.Service -in $ExpectedServices -and
                $_.State -eq "running" -and
                $_.Health -eq "healthy"
            } | Select-Object -ExpandProperty Service -Unique
        )
        if (@($ExpectedServices | Where-Object { $_ -notin $healthy }).Count -eq 0) {
            return
        }
        if ($watch.Elapsed.TotalSeconds - $lastReport -ge 15) {
            Write-Step ("Waiting for services: " + (Get-ServiceStatusText -Rows $rows))
            $lastReport = $watch.Elapsed.TotalSeconds
        }
        Start-Sleep -Seconds 3
    }
    $finalRows = @(Get-ComposeRows)
    throw "Services did not become healthy: $(Get-ServiceStatusText -Rows $finalRows)"
}

function Wait-DiscordReady {
    $containerId = & docker @ComposeArguments ps -q lingxu-bot 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($containerId)) {
        throw "The Lingxu Bot container was not found after startup."
    }
    $startedAt = & docker inspect --format "{{.State.StartedAt}}" $containerId 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($startedAt)) {
        throw "The Lingxu Bot startup timestamp could not be inspected."
    }
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt 60) {
        $logs = & docker @ComposeArguments logs --no-color --since $startedAt lingxu-bot 2>$null
        if ($LASTEXITCODE -eq 0 -and $logs -match '"event":\s*"bot_ready"') {
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Bot health passed, but Discord bot_ready was not observed in recent startup logs."
}

function Get-ComposeContainerId {
    param(
        [string]$Service,
        [switch]$IncludeStopped
    )

    $arguments = $ComposeArguments + @("ps")
    if ($IncludeStopped) {
        $arguments += "--all"
    }
    $arguments += @("--quiet", $Service)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& docker @arguments 2>$null)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Existing container lookup failed with Docker exit code $exitCode."
    }
    $containerIds = @(
        $output |
            Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
            ForEach-Object { ([string]$_).Trim() } |
            Select-Object -Unique
    )
    if ($containerIds.Count -gt 1) {
        throw "Existing service $Service unexpectedly has multiple containers."
    }
    if ($containerIds.Count -eq 0) {
        return $null
    }
    return $containerIds[0]
}

function Invoke-DeploymentActivityProbe {
    param([string[]]$Arguments)

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& docker @Arguments 2>$null)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -notin @(0, 2)) {
        throw "Deployment activity probe failed with Docker exit code $exitCode."
    }
    $line = @(
        $output | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) }
    ) | Select-Object -Last 1
    if ($null -eq $line) {
        throw "Deployment activity probe returned no content-safe result."
    }
    try {
        $payload = ([string]$line) | ConvertFrom-Json
        $properties = @($payload.PSObject.Properties)
        $propertyNames = @($properties | Select-Object -ExpandProperty Name)
        if (
            $properties.Count -ne 2 -or
            "active_meetings" -notin $propertyNames -or
            "executable_jobs" -notin $propertyNames
        ) {
            throw "invalid property set"
        }
        $integerTypes = @("System.Int32", "System.Int64")
        if (
            $null -eq $payload.active_meetings -or
            $payload.active_meetings.GetType().FullName -notin $integerTypes -or
            $null -eq $payload.executable_jobs -or
            $payload.executable_jobs.GetType().FullName -notin $integerTypes
        ) {
            throw "invalid count types"
        }
        $activeMeetings = [int64]$payload.active_meetings
        $executableJobs = [int64]$payload.executable_jobs
    }
    catch {
        throw "Deployment activity probe returned an invalid content-safe result."
    }
    if ($activeMeetings -lt 0 -or $executableJobs -lt 0) {
        throw "Deployment activity probe returned negative counts."
    }
    $expectedExitCode = if ($activeMeetings -gt 0 -or $executableJobs -gt 0) { 2 } else { 0 }
    if ($exitCode -ne $expectedExitCode) {
        throw "Deployment activity probe result did not match its exit code."
    }
    return [pscustomobject]@{
        active_meetings = $activeMeetings
        executable_jobs = $executableJobs
    }
}

function Get-ExistingDeploymentActivity {
    $containerId = $null
    foreach ($service in @("lingxu-bot", "lingxu-worker")) {
        $containerId = Get-ComposeContainerId -Service $service -IncludeStopped
        if (-not [string]::IsNullOrWhiteSpace($containerId)) {
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($containerId)) {
        return $null
    }

    $status = & docker inspect --format "{{.State.Status}}" $containerId 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($status)) {
        throw "Existing application container state could not be inspected."
    }
    if (([string]$status).Trim() -eq "running") {
        Write-Step "Checking the existing running deployment before image builds..."
        return Invoke-DeploymentActivityProbe -Arguments @(
            "exec",
            $containerId,
            "python",
            "/app/scripts/check_deployment_idle.py"
        )
    }

    $imageId = & docker inspect --format "{{.Image}}" $containerId 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($imageId)) {
        throw "Existing application image identity could not be inspected."
    }
    Write-Step "Checking the stopped deployment with its immutable application image..."
    return Invoke-DeploymentActivityProbe -Arguments @(
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "--mount",
        "source=lingxu-data,target=/data",
        ([string]$imageId).Trim(),
        "/app/scripts/check_deployment_idle.py"
    )
}

function Test-DeploymentActivity {
    param([AllowNull()][object]$Activity)

    return $null -ne $Activity -and (
        [int]$Activity.active_meetings -gt 0 -or [int]$Activity.executable_jobs -gt 0
    )
}

function Test-ExistingDeploymentComplete {
    foreach ($service in $ExpectedServices) {
        $containerId = Get-ComposeContainerId -Service $service -IncludeStopped
        if ([string]::IsNullOrWhiteSpace($containerId)) {
            return $false
        }
    }
    return $true
}

function Resume-ExistingDeployment {
    param([object]$Activity)

    if (-not (Test-ExistingDeploymentComplete)) {
        throw (
            "Active Meeting or executable Worker Job detected, but the complete existing " +
            "deployment is unavailable. Refusing to rebuild or replace it."
        )
    }
    Write-Step (
        "Active work detected " +
        "(Meetings=$($Activity.active_meetings), Jobs=$($Activity.executable_jobs)); " +
        "recovering the existing deployment without image replacement..."
    )
    foreach ($service in @("lingxu-ollama", "lingxu-tts", "lingxu-worker", "lingxu-bot")) {
        $containerId = Get-ComposeContainerId -Service $service -IncludeStopped
        $status = & docker inspect --format "{{.State.Status}}" $containerId 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($status)) {
            throw "Existing $service container state could not be inspected."
        }
        switch (([string]$status).Trim()) {
            "running" { continue }
            "paused" {
                Write-Step "Unpausing the existing $service container..."
                Invoke-Docker -Arguments @("unpause", $containerId)
            }
            { $_ -in @("created", "exited") } {
                Write-Step "Starting the existing $service container..."
                Invoke-Docker -Arguments @("start", $containerId)
            }
            default {
                throw "Existing $service container is in unsupported recovery state: $_."
            }
        }
    }
    Wait-LingxuHealthy

    Write-Step "Running strict local readiness checks on the recovered deployment..."
    Invoke-Docker -Arguments (
        $ComposeArguments +
        @("exec", "-T", "lingxu-bot", "python", "-m", "lingxu.cli.doctor", "--strict-discord")
    ) -Quiet
    Wait-DiscordReady

    $rows = @(Get-ComposeRows)
    Write-Step ("All services are healthy: " + (Get-ServiceStatusText -Rows $rows))
    Write-Host (
        "[READY] The existing deployment was recovered without rebuild or replacement; " +
        "active work was preserved."
    ) -ForegroundColor Green
}

function Assert-DeploymentIdle {
    Write-Step "Checking that no Meeting or executable Worker Job is active..."
    $activity = Invoke-DeploymentActivityProbe -Arguments (
        $ComposeArguments +
        @(
            "run",
            "--rm",
            "--no-deps",
            "--entrypoint",
            "python",
            "lingxu-migrate",
            "/app/scripts/check_deployment_idle.py"
        )
    )
    if (Test-DeploymentActivity -Activity $activity) {
        throw "Deployment became active during image preparation; refusing service replacement."
    }
}

$LaunchExitCode = 0
try {
    Set-Location -LiteralPath $ProjectRoot
    Write-Step "Validating local deployment..."
    if (-not (Get-Command docker -CommandType Application -ErrorAction SilentlyContinue)) {
        throw "Docker CLI is not installed or is not on PATH."
    }
    Start-DockerDesktopIfNeeded
    Invoke-Docker -Arguments @("compose", "version") -Quiet
    Assert-EnvironmentReady
    Set-BuildRevisionFromRepository
    Set-TtsBuildFingerprint
    Import-DeepSeekCredentialIfNeeded
    Invoke-Docker -Arguments ($ComposeArguments + @("config", "--quiet")) -Quiet

    $profile = if ($Cpu) { "CPU" } else { "NVIDIA GPU" }
    if ($DryRun) {
        Write-Step "Dry run passed. Profile: $profile. No services were changed."
        $LaunchExitCode = 0
        return
    }

    $existingActivity = Get-ExistingDeploymentActivity
    if (Test-DeploymentActivity -Activity $existingActivity) {
        Resume-ExistingDeployment -Activity $existingActivity
        $LaunchExitCode = 0
        return
    }

    Write-Step "Building the shared application image..."
    Invoke-Docker -Arguments ($ComposeArguments + @("build", "lingxu-migrate"))

    $ttsFingerprint = [Environment]::GetEnvironmentVariable("LINGXU_TTS_BUILD_FINGERPRINT", "Process")
    if ($RebuildAll -or -not (Test-TtsImageCurrent -ExpectedFingerprint $ttsFingerprint)) {
        Write-Step "Building the CosyVoice TTS image..."
        Invoke-Docker -Arguments ($ComposeArguments + @("build", "lingxu-tts"))
    }
    else {
        Write-Step "CosyVoice TTS image matches its current source fingerprint; build skipped."
    }

    Assert-DeploymentIdle
    Write-Step "Starting Bot, Worker, Ollama, and CosyVoice TTS with the $profile profile..."
    Invoke-Docker -Arguments ($ComposeArguments + @("up", "-d", "--no-build"))
    Wait-LingxuHealthy

    Write-Step "Running strict local readiness checks..."
    Invoke-Docker -Arguments (
        $ComposeArguments +
        @("exec", "-T", "lingxu-bot", "python", "-m", "lingxu.cli.doctor", "--strict-discord")
    ) -Quiet
    Wait-DiscordReady

    $rows = @(Get-ComposeRows)
    Write-Step ("All services are healthy: " + (Get-ServiceStatusText -Rows $rows))
    Write-Host "[READY] Discord application commands are available." -ForegroundColor Green
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "The launcher did not print credentials, meeting text, or transcript content."
    Write-Host "For content-safe status, run: docker compose ps"
    $LaunchExitCode = 1
}
finally {
    if ($script:ImportedDeepSeekCredential) {
        Remove-Item Env:SUMMARY_LLM_API_KEY -ErrorAction SilentlyContinue
        $script:ImportedDeepSeekCredential = $false
    }
    if ($script:InjectedBuildRevision) {
        Remove-Item Env:LINGXU_BUILD_REVISION -ErrorAction SilentlyContinue
        $script:InjectedBuildRevision = $false
    }
    if ($script:InjectedTtsBuildFingerprint) {
        Remove-Item Env:LINGXU_TTS_BUILD_FINGERPRINT -ErrorAction SilentlyContinue
        $script:InjectedTtsBuildFingerprint = $false
    }
}
exit $LaunchExitCode
