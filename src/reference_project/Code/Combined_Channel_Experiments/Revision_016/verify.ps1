# Independent implementation of file/state QA; does not verify model performance.
$ErrorActionPreference = 'Stop'
$comparisonRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$comparisonOut = Join-Path $comparisonRoot 'Outputs\Experiments\Combined_Channel_Revision_016'

function Read-ComparisonJson([string]$Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable
}
function Get-ComparisonSha([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Write-ComparisonRecord([string]$Name, $Value) {
    $comparisonBytes = [System.Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 20) + "`n")
    $comparisonStream = [System.IO.File]::Open((Join-Path $comparisonOut $Name), [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try { $comparisonStream.Write($comparisonBytes, 0, $comparisonBytes.Length) } finally { $comparisonStream.Dispose() }
}

if ((Test-Path -LiteralPath (Join-Path $comparisonOut 'independent_qa.json')) -or (Test-Path -LiteralPath (Join-Path $comparisonOut 'final_verification.json'))) { throw 'Do not rerun this one-shot check' }
$comparisonProtected = Read-ComparisonJson (Join-Path $comparisonOut 'sources_before.json')
$comparisonManifest = Read-ComparisonJson (Join-Path $comparisonOut 'delivery_manifest.json')
foreach ($comparisonPath in $comparisonProtected.Keys) {
    if ((Get-ComparisonSha $comparisonPath) -ne $comparisonProtected[$comparisonPath]) { throw "Protected source changed: $comparisonPath" }
}
foreach ($comparisonRelative in $comparisonManifest.Keys) {
    if ((Get-ComparisonSha (Join-Path $comparisonRoot $comparisonRelative)) -ne $comparisonManifest[$comparisonRelative]) { throw "Delivery changed: $comparisonRelative" }
}
$comparisonState = Read-ComparisonJson (Join-Path $comparisonOut 'readiness.json')
$comparisonLookup = Read-ComparisonJson (Join-Path $comparisonOut 'search_log.json')
$comparisonQa = Read-ComparisonJson (Join-Path $comparisonOut 'provenance_qa.json')
if ($comparisonQa.status -ne 'PASS' -or $comparisonState.status -ne 'BLOCKED_NOT_COMPUTED') { throw 'Invalid evidence/readiness state' }
if ($null -ne $comparisonState.real_dynamic_outputs -or $null -ne $comparisonState.real_dynamic_metrics -or $comparisonState.seven_family_complete) { throw 'Unexpected result/completion claim' }
if ($comparisonState.new_training -ne 0 -or $comparisonState.new_inference -ne 0 -or $comparisonState.simulation_runs_executed -ne 0 -or $comparisonState.weather_acquisition_authorized -or $comparisonState.author_contact_or_purchase_authorized) { throw 'Execution/authority mismatch' }
if ($comparisonLookup.asset_obtained -or $comparisonLookup.downloaded_weather_or_event_files -ne 0 -or $comparisonLookup.external_messages_sent -ne 0) { throw 'Acquisition/contact mismatch' }
if ($comparisonState.gates.Count -ne 6 -or $comparisonLookup.sources.Count -ne 3) { throw 'Missing evidence/gates' }
$comparisonDoi = @($comparisonLookup.sources | ForEach-Object { $_.doi } | Sort-Object)
if (($comparisonDoi -join '|') -ne '10.1002/2014RS005541|10.1002/sat.70007|10.1109/TAP.2024.3455567') { throw 'Incorrect source identities' }
$comparisonPriorQa = Read-ComparisonJson (Join-Path $comparisonRoot 'Outputs\Experiments\Combined_Channel_Revision_012\verification.json')
if ($comparisonPriorQa.status -ne 'PASS' -or $comparisonPriorQa.verified_run_metrics -ne 64 -or $comparisonPriorQa.common_uk_hours -ne 5538 -or $comparisonPriorQa.local_test_hours -ne 687) { throw 'Prior scored scope/QA mismatch' }
$comparisonStop = Join-Path $comparisonRoot 'Outputs\Experiments\Combined_Channel_Revision_001\external\weather_gbs_extend_v011\STOP_REQUESTED.json'
if ((Get-ComparisonSha $comparisonStop) -ne '0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a') { throw 'STOP changed' }
$comparisonLock = [System.IO.File]::Open((Join-Path $comparisonOut 'execution.lock'), [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::ReadWrite)
try { $comparisonLock.Lock(0,1); $comparisonLock.Unlock(0,1) } finally { $comparisonLock.Dispose() }
Write-ComparisonRecord 'independent_qa.json' ([ordered]@{
    status='PASS';utc=[DateTime]::UtcNow.ToString('o');method='Separate PowerShell SHA and state checks, same agent';
    protected_sources=$comparisonProtected.Count;delivery_files=$comparisonManifest.Count;
    source_records=3;dependency_gates=6;old_metric_qa_reused=$true;new_metrics_computed=$false;
    independent_source_peer_review=$false;mts_performance_validated=$false;seven_family_complete=$false
})
Write-ComparisonRecord 'final_verification.json' ([ordered]@{
    status='PASS';utc=[DateTime]::UtcNow.ToString('o');protected_sources=$comparisonProtected.Count;
    delivery_files=$comparisonManifest.Count;delivery_manifest_sha256=(Get-ComparisonSha (Join-Path $comparisonOut 'delivery_manifest.json'));
    independent_qa_sha256=(Get-ComparisonSha (Join-Path $comparisonOut 'independent_qa.json'));
    source_sha_unchanged=$true;os_lock='RELEASED';logical_write_scope='RELEASED';
    new_training=0;new_predictions=0;new_scores=0;weather_requests=0;external_messages=0;
    automation_modified=$false;mts_status='BLOCKED_NOT_COMPUTED';seven_family_comparison_complete=$false;
    scores_reference='Outputs/Experiments/Combined_Channel_Revision_012/REPORT.md'
})
Get-Content -LiteralPath (Join-Path $comparisonOut 'final_verification.json') -Raw
