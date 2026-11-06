param([ValidateSet('figure','all')][string]$Mode = 'figure')
$ErrorActionPreference = 'Stop'
$analysisPython = 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe'
$analysisStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
if ($Mode -eq 'figure') {
    $analysisDestination = Join-Path $PSScriptRoot ('reproductions\figure_' + $analysisStamp)
    if (Test-Path -LiteralPath $analysisDestination) { throw 'Reproduction directory already exists.' }
    & $analysisPython (Join-Path $PSScriptRoot 'code\plot_shap_v001.py') --output-root $analysisDestination
    if ($LASTEXITCODE -ne 0) { throw 'Figure reproduction failed.' }
} else {
    $analysisDestination = Join-Path (Split-Path -Parent $PSScriptRoot) ('TabM_K08_SHAP_Reproduction_' + $analysisStamp)
    if (Test-Path -LiteralPath $analysisDestination) { throw 'Reproduction directory already exists.' }
    New-Item -ItemType Directory -Path (Join-Path $analysisDestination 'code') | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'PROTOCOL.md') -Destination $analysisDestination
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'code') -Filter '*.py' | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $analysisDestination 'code')
    }
    foreach ($analysisStep in @('calculate_shap_v001.py','verify_shap_v001.py','summarize_shap_v001.py','plot_shap_v001.py')) {
        & $analysisPython (Join-Path $analysisDestination ('code\' + $analysisStep))
        if ($LASTEXITCODE -ne 0) { throw ('Reproduction failed at ' + $analysisStep + '; existing evidence is preserved.') }
    }
}
Write-Output ('New reproduction: ' + $analysisDestination)
