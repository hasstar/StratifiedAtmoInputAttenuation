$ErrorActionPreference='Stop'
$projectPath='E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction'
$deliveryPath=Join-Path $projectPath 'Outputs\Experiments\Combined_Channel_Revision_017'
$sources=Get-Content -Raw (Join-Path $deliveryPath 'sources_before.json')|ConvertFrom-Json
foreach($p in $sources.PSObject.Properties){if((Get-FileHash -LiteralPath $p.Name -Algorithm SHA256).Hash.ToLower() -ne $p.Value){throw "Protected SHA mismatch: $($p.Name)"}}
$manifest=Get-Content -Raw (Join-Path $deliveryPath 'delivery_manifest.json')|ConvertFrom-Json
foreach($p in $manifest.files.PSObject.Properties){if((Get-FileHash -LiteralPath (Join-Path $projectPath $p.Name) -Algorithm SHA256).Hash.ToLower() -ne $p.Value){throw "Delivery SHA mismatch: $($p.Name)"}}
$qa=Get-Content -Raw (Join-Path $deliveryPath 'independent_qa.json')|ConvertFrom-Json
if($qa.status -ne 'PASS' -or $qa.run_hours_replayed -ne 25888){throw 'QA incomplete'}
$samples=@()
foreach($i in 1..2){
 $v=@(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe' OR Name='GSDSU.x' OR Name='wrf.exe' OR Name='wsl.exe'" -ErrorAction Stop|ForEach-Object {$proc=Get-Process -Id $_.ProcessId -ErrorAction Stop;[pscustomobject]@{pid=$_.ProcessId;created=$_.CreationDate.ToUniversalTime().ToString('o');command=$_.CommandLine;cpu_s=$proc.CPU}})
 $samples += [pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o');processes=$v}
 if($v.Count -ne 0){throw 'Runtime present; inspect before final release'}
 if($i -eq 1){Start-Sleep -Seconds 2}
}
$taskLock=[IO.File]::Open((Join-Path $deliveryPath 'execution.lock'),[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::ReadWrite)
try{$taskLock.Lock(0,1);$taskLock.Unlock(0,1)}finally{$taskLock.Dispose()}
$logs=@{}
foreach($f in Get-ChildItem -LiteralPath $deliveryPath -Filter '*.log'){$logs[$f.Name]=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLower()}
$result=[ordered]@{status='PASS';utc=[DateTime]::UtcNow.ToString('o');protected_sources=@($sources.PSObject.Properties).Count;delivery_files=@($manifest.files.PSObject.Properties).Count;delivery_manifest_sha256=(Get-FileHash -LiteralPath (Join-Path $deliveryPath 'delivery_manifest.json') -Algorithm SHA256).Hash.ToLower();independent_qa_sha256=(Get-FileHash -LiteralPath (Join-Path $deliveryPath 'independent_qa.json') -Algorithm SHA256).Hash.ToLower();process_samples=$samples;log_sha256=$logs;source_sha_unchanged=$true;os_lock='RELEASED';logical_write_scope='RELEASED';new_training=0;new_hourly_predictions=0;weather_requests=0;new_cdf_or_reference_fits=0;automation_modified=$false;seven_family_comparison_complete=$false;new_delivery='Four-family common-frozen-fast marginal comparison on all completed rain/overcast dates'}
$json=$result|ConvertTo-Json -Depth 8
$stream=[IO.File]::Open((Join-Path $deliveryPath 'final_verification.json'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes($json);$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
$json
