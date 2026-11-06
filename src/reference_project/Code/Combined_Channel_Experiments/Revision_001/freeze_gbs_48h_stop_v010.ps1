$ErrorActionPreference = 'Stop'
$root = 'E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction'
Set-Location -LiteralPath $root
$out = Join-Path $root 'Outputs\Experiments\Combined_Channel_Revision_001\external\validation_gbs_48h_v010'
$before = Join-Path $out 'before'
if (Test-Path -LiteralPath $before) { throw 'Snapshot directory already exists; do not rerun.' }
New-Item -ItemType Directory -Path $before | Out-Null
$sources = @{
  'T012.md' = Join-Path $root 'Tasks\T012_Combined_Channel_And_External_Validation.md'
  'PROGRESS.md' = Join-Path $root 'Code\Combined_Channel_Experiments\Revision_001\PROGRESS.md'
  'automation.toml' = 'C:\Users\SDR_HH\.codex\automations\tvt\automation.toml'
}
$snapshots = @()
foreach ($entry in $sources.GetEnumerator()) {
  $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Value).Hash.ToLower()
  $dest = Join-Path $before $entry.Key
  Copy-Item -LiteralPath $entry.Value -Destination $dest
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath $dest).Hash.ToLower() -ne $hash) { throw 'Snapshot hash mismatch' }
  $snapshots += [pscustomobject]@{source=$entry.Value; snapshot=$dest; sha256=$hash}
}
$actual = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'weather_download.*\.py' })
if ($actual.Count -ne 1) { throw ('Expected exactly one downloader; found ' + $actual.Count) }
$worker = $actual[0]
$expectedCommand = '"C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe" -B Code\Combined_Channel_Experiments\Revision_001\weather_download_top10_ncar_v009.py run'
if ($worker.ProcessId -ne 31720 -or $worker.CommandLine -cne $expectedCommand -or $worker.CreationDate.ToString('yyyy-MM-ddTHH:mm:ss.ffffff') -ne '2026-10-15T17:47:52.558971') { throw 'Worker identity changed; refuse stop' }
$cpu1 = (Get-Process -Id $worker.ProcessId).CPU
Start-Sleep -Seconds 2
$cpu2 = (Get-Process -Id $worker.ProcessId).CPU
$lockBefore = & 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe' -B -c "import sys;sys.path.insert(0,'Code/Combined_Channel_Experiments/Revision_001');import weather_download_top10_ncar_v009 as w;print(w.app.io.lock_held())"
if ($LASTEXITCODE -ne 0 -or $lockBefore -ne 'True') { throw 'Expected held OS lock before stop' }
$fresh = Get-CimInstance Win32_Process -Filter 'ProcessId=31720'
if ($fresh.CreationDate -ne $worker.CreationDate -or $fresh.CommandLine -cne $expectedCommand) { throw 'PID changed before stop' }
Stop-Process -Id 31720
Start-Sleep -Seconds 2
$remaining = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'weather_download.*\.py' })
$lockAfter = & 'C:\Users\SDR_HH\.conda\envs\pytorchmy\python.exe' -B -c "import sys;sys.path.insert(0,'Code/Combined_Channel_Experiments/Revision_001');import weather_download_top10_ncar_v009 as w;print(w.app.io.lock_held())"
if ($LASTEXITCODE -ne 0 -or $remaining.Count -ne 0 -or $lockAfter -ne 'False') { throw 'Stop verification failed' }
$oldOutput = Join-Path $root 'Outputs\Experiments\Combined_Channel_Revision_001\external\weather_top10_ncar_v009'
$stateCopy = Join-Path $out 'stopped_v009_snapshot'
Copy-Item -LiteralPath $oldOutput -Destination $stateCopy -Recurse
$copied = @()
foreach ($src in Get-ChildItem -LiteralPath $oldOutput -File -Recurse) {
  $relative = $src.FullName.Substring($oldOutput.Length + 1)
  $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $src.FullName).Hash.ToLower()
  if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $stateCopy $relative)).Hash.ToLower() -ne $hash) { throw ('State snapshot mismatch: ' + $relative) }
  $copied += [pscustomobject]@{relative_path=$relative; sha256=$hash; bytes=$src.Length}
}
$evidence = [ordered]@{status='STOPPED_BY_HUMAN_SCOPE_REDUCTION'; checked_local=(Get-Date).ToString('o'); pid=$worker.ProcessId; creation_local=$worker.CreationDate.ToString('o'); command=$worker.CommandLine; cpu_seconds=@($cpu1,$cpu2); os_lock_before=$lockBefore; os_lock_after=$lockAfter; actual_downloaders_remaining=$remaining.Count; original_state_files_unchanged=$true; before_snapshots=$snapshots; stopped_v009_files=$copied; remote_cancellation=$false; new_downloads=$false; new_fit_calls=0; predictions_executed=0}
$evidence | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $out 'stop_verification.json') -Encoding utf8
$evidence | Select-Object status,checked_local,pid,creation_local,cpu_seconds,os_lock_before,os_lock_after,actual_downloaders_remaining | ConvertTo-Json -Depth 4
