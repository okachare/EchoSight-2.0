param(
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ArtifactRoot = Join-Path $ProjectRoot 'artifacts'
$StagingRoot = Join-Path $env:TEMP ("EchoSight2-build-{0}" -f [guid]::NewGuid().ToString('N'))
$WorkPath = Join-Path $StagingRoot 'work'
$DistPath = Join-Path $StagingRoot 'dist'

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

if (-not (Test-Path $Python)) {
    throw "EchoSight virtual environment not found: $Python. Run SETUP.ps1 first."
}

$Version = (& $Python -c "from echosight2 import __version__; print(__version__)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $Version) {
    throw 'Could not determine the EchoSight version.'
}

$PackageName = "EchoSight2-$Version-windows-x64"
$FinalDirectory = Join-Path $ArtifactRoot $PackageName
$ArchivePath = Join-Path $ArtifactRoot "$PackageName.zip"
$ManifestPath = Join-Path $ArtifactRoot "$PackageName.build.json"

try {
    New-Item -ItemType Directory -Path $WorkPath, $DistPath, $ArtifactRoot -Force | Out-Null
    Push-Location $ProjectRoot
    try {
        if (-not $SkipTests) {
            Invoke-Checked { & $Python -m pytest -q } 'Running release tests...'
        }
        Invoke-Checked {
            & $Python -m PyInstaller --clean --noconfirm --workpath $WorkPath --distpath $DistPath (Join-Path $ProjectRoot 'EchoSight2.spec')
        } 'Building standalone EchoSight 2.0...'
    } finally {
        Pop-Location
    }

    $BuiltDirectory = Join-Path $DistPath 'EchoSight2'
    $BuiltExecutable = Join-Path $BuiltDirectory 'EchoSight2.exe'
    if (-not (Test-Path $BuiltExecutable)) {
        throw "PyInstaller did not create the expected executable: $BuiltExecutable"
    }

    if (Test-Path $FinalDirectory) {
        Remove-Item $FinalDirectory -Recurse -Force
    }
    $StagedExecutable = Join-Path $BuiltDirectory 'EchoSight2.exe'
    Invoke-Checked { & $StagedExecutable --diagnostics } 'Running packaged dependency diagnostics...'

    Write-Host 'Creating portable ZIP archive...' -ForegroundColor Yellow
    $StagedArchive = Join-Path $StagingRoot "$PackageName.zip"
    Compress-Archive -Path (Join-Path $BuiltDirectory '*') -DestinationPath $StagedArchive -CompressionLevel Fastest

    Copy-Item $BuiltDirectory $FinalDirectory -Recurse
    if (Test-Path $ArchivePath) {
        Remove-Item $ArchivePath -Force
    }
    Copy-Item $StagedArchive $ArchivePath

    $FinalExecutable = Join-Path $FinalDirectory 'EchoSight2.exe'

    $Manifest = [ordered]@{
        product = 'EchoSight 2.0'
        version = $Version
        created_at = (Get-Date).ToString('o')
        platform = 'windows-x64'
        executable = $FinalExecutable
        executable_sha256 = (Get-FileHash $FinalExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
        archive = $ArchivePath
        archive_sha256 = (Get-FileHash $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        archive_size_bytes = (Get-Item $ArchivePath).Length
        tests_run = (-not $SkipTests)
        packaged_diagnostics = 'passed'
    }
    $Manifest | ConvertTo-Json | Set-Content -Path $ManifestPath -Encoding UTF8

    Write-Host ''
    Write-Host 'Portable build complete.' -ForegroundColor Green
    Write-Host "Directory: $FinalDirectory"
    Write-Host "Archive: $ArchivePath"
    Write-Host "Manifest: $ManifestPath"
} finally {
    if (Test-Path $StagingRoot) {
        Remove-Item $StagingRoot -Recurse -Force
    }
}
