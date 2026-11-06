param([ValidateSet('Test','Watch')][string]$Mode = 'Test')
$ErrorActionPreference = 'Stop'
$tvtRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$tvtOut = Join-Path $tvtRoot 'Outputs\Experiments\Combined_Channel_Revision_033'
$tvtPython = 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe'
$tvtOldScript = 'Code\Combined_Channel_Experiments\Revision_032\acquire_daily.py'
$tvtNewScript = 'Code\Combined_Channel_Experiments\Revision_033\trial.py'
$tvtPolicy = Get-Content -LiteralPath (Join-Path $tvtOut 'trial_policy.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$tvtCutoff = [DateTimeOffset]::Parse($tvtPolicy.cutoff_utc).UtcDateTime

function Write-NewJson([string]$Path, $Value) {
    $tvtStream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    $tvtWriter = [IO.StreamWriter]::new($tvtStream, [Text.UTF8Encoding]::new($false))
    try { $tvtWriter.WriteLine(($Value | ConvertTo-Json -Depth 8)); $tvtWriter.Flush(); $tvtStream.Flush($true) }
    finally { $tvtWriter.Dispose() }
}

function Get-ApprovedScript($ProcessInfo) {
    if ($ProcessInfo.Name -ine 'python.exe' -or $ProcessInfo.ExecutablePath -ine $tvtPython) { return $null }
    $tvtPrefix = '^"' + [regex]::Escape($tvtPython) + '"\s+-B\s+'
    $tvtOldPattern = $tvtPrefix + '"?' + [regex]::Escape($tvtOldScript) + '"?\s+run\s+--max-wait-seconds\s+(\d{1,4})$'
    if ($ProcessInfo.CommandLine -match $tvtOldPattern) {
        $tvtSeconds = [int]$Matches[1]
        if ($tvtSeconds -ge 60 -and $tvtSeconds -le 3600) { return $tvtOldScript }
    }
    $tvtNewPattern = $tvtPrefix + '"?' + [regex]::Escape($tvtNewScript) + '"?\s+resume$'
    if ($ProcessInfo.CommandLine -match $tvtNewPattern) { return $tvtNewScript }
    return $null
}

if ($Mode -eq 'Test') {
    $tvtCases = @(
        @{Name='old_allowed'; Exe=$tvtPython; Cmd=('"' + $tvtPython + '" -B ' + $tvtOldScript + ' run --max-wait-seconds 3600'); Expected=$tvtOldScript},
        @{Name='new_allowed'; Exe=$tvtPython; Cmd=('"' + $tvtPython + '" -B ' + $tvtNewScript + ' resume'); Expected=$tvtNewScript},
        @{Name='other_python'; Exe='C:\other\python.exe'; Cmd=('"' + $tvtPython + '" -B ' + $tvtOldScript + ' run --max-wait-seconds 3600'); Expected=$null},
        @{Name='other_script'; Exe=$tvtPython; Cmd=('"' + $tvtPython + '" -B other.py'); Expected=$null},
        @{Name='wrong_mode'; Exe=$tvtPython; Cmd=('"' + $tvtPython + '" -B ' + $tvtNewScript + ' test'); Expected=$null},
        @{Name='extra_args'; Exe=$tvtPython; Cmd=('"' + $tvtPython + '" -B ' + $tvtNewScript + ' resume --other'); Expected=$null},
        @{Name='excessive_wait'; Exe=$tvtPython; Cmd=('"' + $tvtPython + '" -B ' + $tvtOldScript + ' run --max-wait-seconds 7200'); Expected=$null}
    )
    foreach ($tvtCase in $tvtCases) {
        $tvtFake = [pscustomobject]@{Name='python.exe'; ExecutablePath=$tvtCase.Exe; CommandLine=$tvtCase.Cmd}
        if ((Get-ApprovedScript $tvtFake) -ne $tvtCase.Expected) { throw ('FAILED_' + $tvtCase.Name) }
    }
    Write-NewJson (Join-Path $tvtOut 'guard_tests.json') @{utc=[DateTime]::UtcNow.ToString('o');status='PASS';tests=$tvtCases.Count;processes_stopped=0;network_requests=0;source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash}
    Write-Output 'PASS_7_GUARD_SELECTION_TESTS_NO_PROCESS_STOPPED'
    exit 0
}

$tvtGuardLock = [IO.File]::Open((Join-Path $tvtOut 'deadline_guard.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $tvtCodeMap = Get-Content -LiteralPath (Join-Path $tvtOut 'code_before.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($tvtEntry in $tvtCodeMap.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $tvtEntry.Name -Algorithm SHA256).Hash -ine $tvtEntry.Value) { throw 'CONTROL_CODE_CHANGED' }
    }
    $tvtOldCodeMap = Get-Content -LiteralPath (Join-Path $tvtRoot 'Outputs\Experiments\Combined_Channel_Revision_032\code_before.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $tvtAllowedHashes = @{}
    foreach ($tvtEntry in $tvtCodeMap.PSObject.Properties) { $tvtAllowedHashes[$tvtEntry.Name] = $tvtEntry.Value }
    foreach ($tvtEntry in $tvtOldCodeMap.PSObject.Properties) { $tvtAllowedHashes[$tvtEntry.Name] = $tvtEntry.Value }
    $tvtInitialWorkers = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { Get-ApprovedScript $_ } | Select-Object ProcessId,CreationDate,CommandLine)
    Write-NewJson (Join-Path $tvtOut 'guard_started.json') @{utc=[DateTime]::UtcNow.ToString('o');pid=$PID;cutoff_utc=$tvtPolicy.cutoff_utc;source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash;initial_workers=$tvtInitialWorkers;network_requests=0}
    while ([DateTime]::UtcNow -lt $tvtCutoff) {
        $tvtMillis = [Math]::Max(1, [Math]::Min(5000, ($tvtCutoff - [DateTime]::UtcNow).TotalMilliseconds))
        Start-Sleep -Milliseconds ([int]$tvtMillis)
    }
    Write-NewJson (Join-Path $tvtOut 'trial_closed.json') @{utc=[DateTime]::UtcNow.ToString('o');cutoff_utc=$tvtPolicy.cutoff_utc;status='TRIAL_WINDOW_CLOSED_NO_MORE_ACQUISITION';automatic_extension_allowed=$false}
    $tvtStopped = @()
    $tvtCandidates = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { Get-ApprovedScript $_ })
    foreach ($tvtCandidate in $tvtCandidates) {
        $tvtScript = Get-ApprovedScript $tvtCandidate
        $tvtSource = Join-Path $tvtRoot $tvtScript
        if ((Get-FileHash -LiteralPath $tvtSource -Algorithm SHA256).Hash -ine $tvtAllowedHashes[$tvtSource]) { throw 'ACQUISITION_CODE_CHANGED_NO_BLIND_STOP' }
        $tvtAgain = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + [int]$tvtCandidate.ProcessId)
        if (-not $tvtAgain) { continue }
        if ($tvtAgain.CreationDate -ne $tvtCandidate.CreationDate -or $tvtAgain.CommandLine -cne $tvtCandidate.CommandLine -or -not (Get-ApprovedScript $tvtAgain)) { throw 'PROCESS_IDENTITY_CHANGED_NO_BLIND_STOP' }
        Write-NewJson (Join-Path $tvtOut ('stop_intent_' + $tvtCandidate.ProcessId + '.json')) @{utc=[DateTime]::UtcNow.ToString('o');pid=$tvtCandidate.ProcessId;creation_date=$tvtCandidate.CreationDate;command_line=$tvtCandidate.CommandLine;reason='USER_TWO_HOUR_TRIAL_CUTOFF'}
        Stop-Process -Id ([int]$tvtCandidate.ProcessId) -ErrorAction Stop
        $tvtStopped += [int]$tvtCandidate.ProcessId
    }
    Start-Sleep -Milliseconds 500
    $tvtRemaining = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { Get-ApprovedScript $_ } | Select-Object ProcessId,CreationDate,CommandLine)
    $tvtDone = @(Get-ChildItem -LiteralPath (Join-Path $tvtRoot 'Outputs\Experiments\Combined_Channel_Revision_032\batches') -Directory | ForEach-Object { $tvtFile = Join-Path $_.FullName 'complete.json'; if (Test-Path -LiteralPath $tvtFile) { Get-Content -LiteralPath $tvtFile -Raw -Encoding UTF8 | ConvertFrom-Json } })
    Write-NewJson (Join-Path $tvtOut 'cutoff_result.json') @{utc=[DateTime]::UtcNow.ToString('o');status=$(if ($tvtRemaining.Count -eq 0) {'CUTOFF_ENFORCED'} else {'CUTOFF_NEEDS_REVIEW'});stopped_pids=$tvtStopped;remaining_workers=$tvtRemaining;completed_subrequests=$tvtDone.Count;production_verified_values=($tvtDone | Measure-Object values -Sum).Sum;baseline_values=888;target_values=421104;files_deleted=0;remote_jobs_deleted=0;new_predictions=0}
} catch {
    Write-NewJson (Join-Path $tvtOut ('guard_failure_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss_fffffff') + '.json')) @{utc=[DateTime]::UtcNow.ToString('o');status='GUARD_FAILED_REVIEW_REQUIRED';error_class=$_.Exception.GetType().FullName;message=$_.Exception.Message}
    throw
} finally {
    $tvtGuardLock.Dispose()
}
