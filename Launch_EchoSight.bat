@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
if defined LOCALAPPDATA (
    set "RUNTIME_ROOT=%LOCALAPPDATA%\EchoSight\2.0"
) else (
    set "RUNTIME_ROOT=%USERPROFILE%\AppData\Local\EchoSight\2.0"
)
set "PYTHON=%RUNTIME_ROOT%\.venv\Scripts\python.exe"
set "PYTHONPATH=%PROJECT_ROOT%src"

if not exist "%PYTHON%" (
    echo EchoSight 2.0 is not set up.
    echo Run SETUP.ps1 from this shared folder first.
    exit /b 1
)

pushd "%PROJECT_ROOT%"
"%PYTHON%" -m echosight2
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo EchoSight 2.0 exited with code %EXIT_CODE%.
    echo Run DIAGNOSE.ps1 for environment details.
)
echo.
pause
exit /b %EXIT_CODE%
