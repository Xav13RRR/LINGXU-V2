[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Security

$credentialDirectory = Join-Path $env:LOCALAPPDATA "Lingxu"
$credentialFile = Join-Path $credentialDirectory "deepseek-api-key.dpapi"
$secureCredential = Read-Host "DeepSeek API key" -AsSecureString
$bstr = [IntPtr]::Zero
[byte[]]$clearBytes = @()
[byte[]]$protectedBytes = @()

try {
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureCredential)
    $credential = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($credential)) {
        throw "The DeepSeek API key cannot be blank."
    }

    $clearBytes = [Text.Encoding]::Unicode.GetBytes($credential)
    $credential = $null
    $protectedBytes = [Security.Cryptography.ProtectedData]::Protect(
        $clearBytes,
        $null,
        [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    $hex = [BitConverter]::ToString($protectedBytes).Replace("-", "")
    [Directory]::CreateDirectory($credentialDirectory) | Out-Null
    [IO.File]::WriteAllText($credentialFile, $hex, [Text.UTF8Encoding]::new($false))
    Write-Host "DeepSeek credential saved in Windows current-user encrypted storage."
}
finally {
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
    if ($clearBytes.Length -gt 0) {
        [Array]::Clear($clearBytes, 0, $clearBytes.Length)
    }
    if ($protectedBytes.Length -gt 0) {
        [Array]::Clear($protectedBytes, 0, $protectedBytes.Length)
    }
    Remove-Variable secureCredential -ErrorAction SilentlyContinue
}
