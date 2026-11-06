param(
    [string]$PythonExe = 'D:\Anaconda202410\python.exe',
    [string]$OutputDir = ''
)
$ErrorActionPreference = 'Stop'
$packageDir = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $packageDir ('Reproduced_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
if (Test-Path -LiteralPath $OutputDir) { throw 'Choose a new output directory.' }
& $PythonExe (Join-Path $PSScriptRoot 'build_figure.py') --package $packageDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Figure reproduction failed.' }
Write-Output $OutputDir
