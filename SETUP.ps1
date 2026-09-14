$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$LocalRoot = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE 'AppData\Local' }
$RuntimeRoot = Join-Path $LocalRoot 'EchoSight\2.0'
$VenvPath = Join-Path $RuntimeRoot '.venv'
$LogDirectory = Join-Path $RuntimeRoot 'logs'
$LogPath = Join-Path $LogDirectory ("setup-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
$PythonCommand = $null
$SetupExitCode = 0

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
Start-Transcript -Path $LogPath -Force | Out-Null

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host $Description -ForegroundColor Yellow
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

try {
    Write-Host 'EchoSight 2.0 setup' -ForegroundColor Cyan
    Write-Host "Project: $ProjectRoot"
    Write-Host "Log: $LogPath"

    foreach ($candidate in @('py -3.9')) {
        $parts = $candidate.Split(' ')
        try {
            & $parts[0] $parts[1] --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PythonCommand = $candidate
                break
            }
        } catch {
            continue
        }
    }

    if (-not $PythonCommand) {
        throw 'Python 3.9 is required. Install Python 3.9 and ensure the py launcher can find it.'
    }

    if (-not (Test-Path $VenvPath)) {
        Write-Host "Creating virtual environment with $PythonCommand..." -ForegroundColor Yellow
        Invoke-Expression "$PythonCommand -m venv `"$VenvPath`""
        if ($LASTEXITCODE -ne 0) {
            throw "Virtual environment creation failed with exit code $LASTEXITCODE."
        }
    }

    $VenvPython = Join-Path $VenvPath 'Scripts\python.exe'
    if (-not (Test-Path $VenvPython)) {
        throw "Virtual environment Python was not created: $VenvPython"
    }

    Write-Host 'Using the pip bundled with the Python 3.9 virtual environment.' -ForegroundColor Yellow
    Invoke-Checked { & $VenvPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt') } 'Installing runtime dependencies...'
    $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
    Invoke-Checked { & $VenvPython -m echosight2.diagnostics } 'Running diagnostics...'

    Write-Host "Local runtime: $RuntimeRoot" -ForegroundColor Green
    Write-Host 'Setup complete. Use Launch_EchoSight.bat from the shared folder to start.' -ForegroundColor Green
} catch {
    Write-Host ''
    Write-Host 'SETUP FAILED' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Full log: $LogPath" -ForegroundColor Yellow
    $SetupExitCode = 1
} finally {
    try { Stop-Transcript | Out-Null } catch { }
    Write-Host ''
    Read-Host 'Press Enter to close this setup window'
}

exit $SetupExitCode
