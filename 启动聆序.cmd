@echo off
setlocal
chcp 65001 >nul
title Lingxu One-Click Launcher
cd /d "%~dp0"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_lingxu.ps1" %*
set "LINGXU_LAUNCH_EXIT=%ERRORLEVEL%"

echo.
if "%LINGXU_LAUNCH_EXIT%"=="0" (
    echo [DONE] Lingxu launcher completed successfully.
) else (
    echo [FAILED] Lingxu did not become ready. Review the error above.
)
echo Press any key to close this window.
pause >nul
exit /b %LINGXU_LAUNCH_EXIT%
