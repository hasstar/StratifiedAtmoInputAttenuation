param([switch]$Execute)

$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$codeRoot = Join-Path $project 'Code\Hourly_Attenuation_Experiments'
$outputRoot = Join-Path $project 'Outputs\Experiments'
$figureRoot = Join-Path $project 'Outputs\Figures'
$keepCode = 'Revision_014'
$keepOutput = 'Hourly_Attenuation_Revision_014_SelfContained'
$latest = Join-Path $outputRoot $keepOutput

$codeNames = @(
    'Revision_001', 'Revision_002', 'Revision_003', 'Revision_004',
    'Revision_005', 'Revision_006', 'Revision_007', 'Revision_008',
    'Revision_010', 'Revision_011', 'Revision_012', 'Revision_013'
)
$outputNames = @(
    'Hourly_Attenuation_Full348_Diagnostics_Revision_001',
    'Hourly_Attenuation_Revision_001', 'Hourly_Attenuation_Revision_002',
    'Hourly_Attenuation_Revision_003',
    'Hourly_Attenuation_Revision_004_MLP_Layers',
    'Hourly_Attenuation_Revision_005_RFE',
    'Hourly_Attenuation_Revision_007_Cleaned',
    'Hourly_Attenuation_Revision_008_XGBoost_Test',
    'Hourly_Attenuation_Revision_010_Frozen_Masking',
    'Hourly_Attenuation_Revision_011_Residual_Layers',
    'Hourly_Attenuation_Revision_012_HonestSplit',
    'Hourly_Attenuation_Revision_013_ValidationStopped'
)
$figureNames = @('Local_XGBoost_Test_Revision_001',
                 'Local_XGBoost_Weather_Revision_001')

function Require-Pass([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required final QA file missing: $path"
    }
    $qa = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if ($qa.status -ne 'PASS') { throw "Final QA is not PASS: $path" }
}

Require-Pass (Join-Path $latest 'model_selection_qa.json')
Require-Pass (Join-Path $latest 'layers\validation_qa.json')
Require-Pass (Join-Path $latest 'layers\final_qa.json')
Require-Pass (Join-Path $latest 'REPRODUCTION_CHECK.json')
foreach ($file in @('run_manifest.json', 'model_selection.json', 'MODEL_REPORT.md',
                    'ten_model_rounds.parquet', 'training_parameters.json',
                    'RESULTS.md', 'layers\run_manifest.json',
                    'layers\layer_rounds.parquet',
                    'layers\layer_selection.json', 'layers\final_test.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $latest $file) -PathType Leaf)) {
        throw "Latest experiment incomplete: $file"
    }
}
$subsetCount = (Get-ChildItem -LiteralPath (Join-Path $latest 'layers\subsets') -File -Filter 'mask_*.json' | Measure-Object).Count
if ($subsetCount -ne 1024) { throw "Expected 1024 latest masks, found $subsetCount" }

$codeResolved = (Resolve-Path -LiteralPath $codeRoot).Path.TrimEnd('\')
$outputResolved = (Resolve-Path -LiteralPath $outputRoot).Path.TrimEnd('\')
$figureResolved = (Resolve-Path -LiteralPath $figureRoot).Path.TrimEnd('\')
$targets = @()
foreach ($batch in @(@($codeResolved, $codeNames),
                     @($outputResolved, $outputNames),
                     @($figureResolved, $figureNames))) {
    $parent = [string]$batch[0]
    foreach ($name in [string[]]$batch[1]) {
        $path = Join-Path $parent $name
        if (-not (Test-Path -LiteralPath $path -PathType Container)) { continue }
        $resolved = (Resolve-Path -LiteralPath $path).Path.TrimEnd('\')
        if ([IO.Path]::GetDirectoryName($resolved) -ne $parent -or
            [IO.Path]::GetFileName($resolved) -ne $name -or
            $name -eq $keepCode -or $name -eq $keepOutput) {
            throw "Unsafe cleanup target: $resolved"
        }
        $item = Get-Item -LiteralPath $resolved -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing reparse-point target: $resolved"
        }
        $size = (Get-ChildItem -LiteralPath $resolved -Recurse -File -ErrorAction Stop |
                 Measure-Object -Property Length -Sum).Sum
        $targets += [pscustomobject]@{Path=$resolved; Bytes=[int64]$size}
    }
}

$running = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'Hourly_Attenuation_Experiments\\Revision_0(0[1-8]|10|11|12|13)\\' })
if ($running.Count -gt 0) {
    throw "Older experiment Python process still running: $($running.ProcessId -join ',')"
}

Write-Output ("Targets: {0} directories, {1:N2} MiB" -f
              $targets.Count, (($targets | Measure-Object -Property Bytes -Sum).Sum / 1MB))
$targets | Format-Table -AutoSize
if (-not $Execute) {
    Write-Output 'PREVIEW ONLY. Pass -Execute only after numerical comparison and final QA.'
    return
}

foreach ($target in $targets) {
    Remove-Item -LiteralPath $target.Path -Recurse -Force -ErrorAction Stop
    if (Test-Path -LiteralPath $target.Path) {
        throw "Deletion did not complete: $($target.Path)"
    }
}
$log = [ordered]@{
    completed_utc = [DateTime]::UtcNow.ToString('o')
    retained_code = (Join-Path $codeResolved $keepCode)
    retained_results = $latest
    removed_directories = @($targets)
    removed_count = $targets.Count
    recovery = 'Permanently removed by PowerShell Remove-Item; no archive created'
    untouched = 'Original H5, ERA5, Run_011, and all Outputs/Datasets'
}
$log | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $latest 'CLEANUP_LOG.json') -Encoding UTF8
Write-Output "CLEANUP COMPLETE: $($targets.Count) directories"
