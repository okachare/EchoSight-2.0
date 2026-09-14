$ErrorActionPreference = 'Continue'
$ProjectRoot = $PSScriptRoot
$LocalRoot = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE 'AppData\Local' }
$RuntimeRoot = Join-Path $LocalRoot 'EchoSight\2.0'
$VenvPython = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $ProjectRoot 'src'

Write-Host 'EchoSight 2.0 diagnostics' -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Local runtime: $RuntimeRoot"
Write-Host "PowerShell: $($PSVersionTable.PSVersion)"

if (Test-Path $VenvPython) {
    & $VenvPython -m echosight2.diagnostics
    $ExitCode = $LASTEXITCODE
    Write-Host ''
    Read-Host 'Press Enter to close this diagnostics window'
    exit $ExitCode
}

Write-Warning "Virtual environment not found: $VenvPython"
Write-Host 'Run SETUP.ps1 first.'
Read-Host 'Press Enter to close this diagnostics window'
exit 1
