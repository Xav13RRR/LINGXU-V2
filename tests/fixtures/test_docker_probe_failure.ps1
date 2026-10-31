Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$launcher = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path (
    "scripts\start_lingxu.ps1"
)
$source = Get-Content -LiteralPath $launcher -Raw -Encoding UTF8
$start = $source.IndexOf("function Test-DockerEngine")
$end = $source.IndexOf("function Start-DockerDesktopIfNeeded")
if ($start -lt 0 -or $end -le $start) {
    throw "Docker engine probe function could not be isolated."
}
Invoke-Expression $source.Substring($start, $end - $start)

function docker {
    Write-Error "simulated missing dockerDesktopLinuxEngine pipe" -ErrorAction Stop
}

if (Test-DockerEngine) {
    throw "Expected the failed Docker named-pipe probe to return false."
}
Write-Host "Docker named-pipe failure was converted to a negative engine probe."
