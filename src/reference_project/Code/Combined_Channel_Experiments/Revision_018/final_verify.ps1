$ErrorActionPreference = 'Stop'
$rootPath = 'E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction'
$outputPath = Join-Path $rootPath 'Outputs\Experiments\Combined_Channel_Revision_018'
$attemptPath = Join-Path $outputPath 'attempt_003'
$finalPath = Join-Path $outputPath 'final_verification.json'
if (Test-Path -LiteralPath $finalPath) { throw 'Final record already exists; do not rerun.' }
$qa = Get-Content -LiteralPath (Join-Path $attemptPath 'independent_qa.json') -Raw | ConvertFrom-Json
if ($qa.status -ne 'PASS_EXTENDED_FROZEN_TEMPORAL_DIAGNOSTIC' -or $qa.verified_hours -ne 3027) { throw 'Independent QA gate' }
$publication = Get-Content -LiteralPath (Join-Path $outputPath 'publication_qa.json') -Raw | ConvertFrom-Json
if ($publication.status -ne 'PASS_REPORT_FROM_INDEPENDENTLY_VERIFIED_SCORES') { throw 'Publication gate' }
$sources = Get-Content -LiteralPath (Join-Path $attemptPath 'sources_before.json') -Raw | ConvertFrom-Json
$sourceCount = 0
foreach ($entry in $sources.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Protected SHA mismatch: $($entry.Name)" }
    $sourceCount += 1
}
$manifestPath = Join-Path $outputPath 'delivery_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$deliveryCount = 0
foreach ($entry in $manifest.files.PSObject.Properties) {
    $targetPath = Join-Path $rootPath $entry.Name
    if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) { throw "Delivery SHA mismatch: $($entry.Name)" }
    $deliveryCount += 1
}
function Get-RelevantProcesses {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and ($_.Name -match '^(python|pythonw|wrf|gsdsu|wsl).*\.exe$' -or
        $_.CommandLine -match 'weather_download|Combined_Channel_Experiments|Fast_Fading_Experiments')
    } | Select-Object ProcessId,Name,CreationDate,CommandLine,KernelModeTime,UserModeTime)
}
$firstUtc = [DateTime]::UtcNow.ToString('o')
$first = Get-RelevantProcesses
Start-Sleep -Seconds 2
$secondUtc = [DateTime]::UtcNow.ToString('o')
$second = Get-RelevantProcesses
if (@($first).Count -gt 0 -or @($second).Count -gt 0) { throw 'A relevant process remains; inspect before finalizing.' }
$locks = @((Join-Path $attemptPath 'execution.lock'), (Join-Path $rootPath 'Outputs\Experiments\Fast_Fading_Revision_002\worker.lock'))
foreach ($lockPath in $locks) {
    $stream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::ReadWrite)
    try { $stream.Lock(0,1); $stream.Unlock(0,1) } finally { $stream.Dispose() }
}
$result = [ordered]@{
    status = 'PASS_EXTENDED_UK_SHARED_FAST_TEMPORAL_DIAGNOSTIC'
    utc = [DateTime]::UtcNow.ToString('o')
    source_files_verified = $sourceCount
    delivery_files_verified = $deliveryCount
    delivery_manifest_sha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    independent_qa_sha256 = (Get-FileHash -LiteralPath (Join-Path $attemptPath 'independent_qa.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    process_sample_1_utc = $firstUtc
    process_sample_1 = @($first)
    process_sample_2_utc = $secondUtc
    process_sample_2 = @($second)
    relevant_OS_locks = 'RELEASED'
    logical_scope = 'Revision018 released; old source files untouched'
    ledger_station_hours = 5664
    scored_station_hours = 3027
    reused_unchanged_station_hours = 932
    new_generation_station_hours = 2095
    new_main_realizations = 134080
    independent_QA_realizations = $qa.independent_replay_realizations
    new_training_calls = 0
    new_weather_requests = 0
    new_hourly_predictions = 0
    STOP_unchanged = $true
    all_seven_families_complete = $false
    full_MTS_complete = $false
    remaining_blockers = @('E-SST original formula and required inputs', 'RMOP GSDSU implementation/configuration and 3D inputs', 'NWP PRM configuration and licensed MTS event library')
}
$json = $result | ConvertTo-Json -Depth 8
$outStream = [System.IO.File]::Open($finalPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
try {
    $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($json)
    $outStream.Write($bytes, 0, $bytes.Length)
} finally { $outStream.Dispose() }
$json
