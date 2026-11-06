# Download only official CEDA GBS files; read the bearer token from the clipboard in memory.
[CmdletBinding()]
param(
    [switch]$InventoryOnly,
    [switch]$VerifyOnly,
    [ValidateRange(0, 100000)][int]$Limit = 0,
    [ValidateRange(1, 16)][int]$ThrottleLimit = 8
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$outputRoot = Join-Path $projectRoot '外部样例数据CEDA\GBS_20p7GHz'
$manifestPath = Join-Path $outputRoot 'source_manifest.csv'
$outputPrefix = [IO.Path]::GetFullPath($outputRoot).TrimEnd('\') + '\'
if ($InventoryOnly -and $VerifyOnly) { throw 'Choose only one of InventoryOnly or VerifyOnly.' }
if (-not (Test-Path -LiteralPath $outputRoot)) { New-Item -ItemType Directory -Path $outputRoot | Out-Null }

if (-not (Test-Path -LiteralPath $manifestPath)) {
    $roots = @(
        [pscustomobject]@{Site='chilbolton';Path='/badc/chilbolton/data/gbs_chilbolton'},
        [pscustomobject]@{Site='sparsholt';Path='/badc/chilbolton/data/gbs_sparsholt'},
        [pscustomobject]@{Site='dundee';Path='/badc/chilbolton/data/gbs_dundee'}
    )
    $months = @()
    foreach ($root in $roots) {
        $rootListing = Invoke-RestMethod -Uri ('https://data.ceda.ac.uk' + $root.Path + '?json=') -TimeoutSec 45
        foreach ($year in @($rootListing.items | Where-Object { $_.type -eq 'dir' -and $_.name -match '^20\d{2}$' })) {
            $yearListing = Invoke-RestMethod -Uri ('https://data.ceda.ac.uk' + $year.path + '?json=') -TimeoutSec 45
            foreach ($month in @($yearListing.items | Where-Object { $_.type -eq 'dir' -and $_.name -match '^(0[1-9]|1[0-2])$' })) {
                $months += [pscustomobject]@{Site=$root.Site;Path=$month.path}
            }
        }
    }
    if ($months.Count -eq 0 -or @($months | Group-Object Site).Count -ne 3) {
        throw 'Monthly directories were not found for all three sites.'
    }
    $listings = @($months | ForEach-Object -Parallel {
        try {
            $listing = Invoke-RestMethod -Uri ('https://data.ceda.ac.uk' + $_.Path + '?json=') -TimeoutSec 45
            [pscustomobject]@{Site=$_.Site;Path=$_.Path;Items=@($listing.items | Where-Object { $_.type -eq 'file' -and $_.name -match '^cfarr-gbs_.*\.nc$' });Error=''}
        } catch {
            [pscustomobject]@{Site=$_.Site;Path=$_.Path;Items=@();Error=$_.Exception.Message}
        }
    } -ThrottleLimit $ThrottleLimit)
    $failedListings = @($listings | Where-Object { $_.Error -or $_.Items.Count -eq 0 })
    if ($failedListings.Count -or $listings.Count -ne $months.Count) {
        throw ('Monthly CEDA listing failed: ' + (($failedListings | ForEach-Object { $_.Path + ' ' + $_.Error }) -join '; '))
    }
    $manifest = @(foreach ($listing in $listings) {
        foreach ($file in $listing.Items) {
            $pattern = '^cfarr-gbs_' + [regex]::Escape($listing.Site) + '_(\d{8})\.nc$'
            if ($file.name -notmatch $pattern) { throw "Unexpected filename: $($file.name)" }
            $stamp = $Matches[1]
            $day = [datetime]::ParseExact($stamp, 'yyyyMMdd', [Globalization.CultureInfo]::InvariantCulture)
            $sourceUrl = [uri]$file.download
            if ($sourceUrl.Scheme -ne 'https' -or $sourceUrl.Host -ne 'dap.ceda.ac.uk' -or
                $sourceUrl.AbsolutePath -ne $file.path -or [long]$file.size -le 0 -or
                $file.md5 -notmatch '^[a-fA-F0-9]{32}$') {
                throw "Unexpected official metadata: $($file.name)"
            }
            [pscustomobject]@{
                site=$listing.Site; date_utc=$day.ToString('yyyy-MM-dd')
                relative_path=(Join-Path $listing.Site (Join-Path $day.ToString('yyyy') (Join-Path $day.ToString('MM') $file.name)))
                source_url=$file.download; bytes=[long]$file.size; md5=$file.md5.ToLowerInvariant()
            }
        }
    }) | Sort-Object site,date_utc
    if ($manifest.Count -eq 0 -or @($manifest | Group-Object relative_path | Where-Object Count -gt 1).Count) {
        throw 'File manifest is empty or has duplicate output paths.'
    }
    if (Test-Path -LiteralPath $manifestPath) { throw 'Manifest appeared during inventory; refusing to overwrite.' }
    $manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8
} else {
    $manifest = @(Import-Csv -LiteralPath $manifestPath)
}

if ($manifest.Count -eq 0) { throw 'The source manifest is empty.' }
foreach ($row in $manifest) {
    $destination = [IO.Path]::GetFullPath((Join-Path $outputRoot $row.relative_path))
    $sourceUrl = [uri]$row.source_url
    if (-not $destination.StartsWith($outputPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        $row.site -notin @('chilbolton','sparsholt','dundee') -or
        $sourceUrl.Scheme -ne 'https' -or $sourceUrl.Host -ne 'dap.ceda.ac.uk' -or
        $row.md5 -notmatch '^[a-fA-F0-9]{32}$' -or [long]$row.bytes -le 0) {
        throw "Invalid manifest row: $($row.relative_path)"
    }
}
Write-Output "Manifest: $($manifest.Count) files; $manifestPath"
if ($InventoryOnly) { return }
$selected = if ($Limit -gt 0) { @($manifest | Select-Object -First $Limit) } else { $manifest }
Write-Output "Selected: $($selected.Count) files"

$cedaToken = ''
if (-not $VerifyOnly) {
    $cedaToken = (Get-Clipboard -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($cedaToken)) { throw 'The clipboard does not contain a CEDA token.' }
    if ($cedaToken.Length -lt 40 -or $cedaToken -match '[^\x21-\x7E]') {
        throw 'Clipboard content is not an ASCII CEDA token; no request was sent.'
    }
    $check = Invoke-WebRequest -Uri 'https://dap.ceda.ac.uk/badc/ARCHIVE_INFO/ACCESS_TEST/RESTRICTED/TOKEN_CHECK' `
        -Headers @{Authorization=('Bearer ' + $cedaToken)} -MaximumRedirection 0 -TimeoutSec 30
    $message = if ($check.Content -is [byte[]]) { [Text.Encoding]::UTF8.GetString($check.Content) } else { [string]$check.Content }
    if ($check.StatusCode -ne 200 -or $message -notmatch 'Congratulations') {
        throw 'CEDA did not validate the clipboard token; no downloads started.'
    }
    Write-Output 'CEDA access token validated without saving or displaying its value.'
}

$verificationMode = [bool]$VerifyOnly
$results = @($selected | ForEach-Object -Parallel {
    $row = $_
    $root = $using:outputRoot
    $prefix = $using:outputPrefix
    $verifyOnly = $using:verificationMode
    $token = $using:cedaToken
    $destination = [IO.Path]::GetFullPath((Join-Path $root $row.relative_path))
    $status = 'failed'
    $detail = ''
    $actualBytes = 0L
    $actualMd5 = ''
    try {
        if (-not $destination.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output path escaped root.' }
        $expectedBytes = [long]$row.bytes
        $expectedMd5 = $row.md5.ToLowerInvariant()
        if (Test-Path -LiteralPath $destination) {
            $actualBytes = (Get-Item -LiteralPath $destination).Length
            if ($actualBytes -eq $expectedBytes) {
                $actualMd5 = (Get-FileHash -LiteralPath $destination -Algorithm MD5).Hash.ToLowerInvariant()
            }
            if ($actualBytes -eq $expectedBytes -and $actualMd5 -eq $expectedMd5) {
                $status = 'verified_existing'
            } else {
                $status = 'existing_mismatch'; $detail = 'Existing file differs from CEDA metadata; not overwritten.'
            }
        } elseif ($verifyOnly) {
            $status = 'missing'; $detail = 'File not present locally.'
        } else {
            $directory = Split-Path -Parent $destination
            if (-not (Test-Path -LiteralPath $directory)) { New-Item -ItemType Directory -Path $directory -Force | Out-Null }
            for ($attempt=1; $attempt -le 3; $attempt++) {
                $temporary = $destination + '.part.' + [guid]::NewGuid().ToString('N')
                try {
                    $response = Invoke-WebRequest -Uri $row.source_url -Headers @{Authorization=('Bearer ' + $token)} `
                        -MaximumRedirection 0 -TimeoutSec 120 -OutFile $temporary -PassThru
                    if ($response.StatusCode -ne 200) { throw "CEDA returned HTTP $($response.StatusCode)." }
                    $actualBytes = (Get-Item -LiteralPath $temporary).Length
                    if ($actualBytes -ne $expectedBytes) { throw "Byte count mismatch: $actualBytes vs $expectedBytes." }
                    $actualMd5 = (Get-FileHash -LiteralPath $temporary -Algorithm MD5).Hash.ToLowerInvariant()
                    if ($actualMd5 -ne $expectedMd5) { throw 'MD5 mismatch.' }
                    if (Test-Path -LiteralPath $destination) { throw 'Target appeared during download; not overwritten.' }
                    Move-Item -LiteralPath $temporary -Destination $destination -ErrorAction Stop
                    $status = 'downloaded_verified'; $detail = ''; break
                } catch {
                    $detail = $_.Exception.Message
                    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
                    if ($detail -match 'Target appeared') { break }
                    if ($attempt -lt 3) { Start-Sleep -Seconds (2 * $attempt) }
                }
            }
        }
    } catch { $detail = $_.Exception.Message }
    [pscustomobject]@{
        site=$row.site; date_utc=$row.date_utc; relative_path=$row.relative_path; status=$status
        expected_bytes=[long]$row.bytes; actual_bytes=$actualBytes
        expected_md5=$row.md5; actual_md5=$actualMd5; detail=$detail
    }
} -ThrottleLimit $ThrottleLimit)

$reportPath = Join-Path $outputRoot ('verification_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0,8) + '.csv')
$results | Sort-Object site,date_utc | Export-Csv -LiteralPath $reportPath -NoTypeInformation -Encoding utf8
$results | Group-Object status | Sort-Object Name | ForEach-Object { Write-Output ($_.Name + ': ' + $_.Count) }
Write-Output "Report: $reportPath"
if ($results.Count -ne $selected.Count -or @($results | Where-Object { $_.status -notin @('verified_existing','downloaded_verified') }).Count) {
    throw 'Some files are missing or mismatched; inspect the report and safely rerun.'
}
Write-Output 'All manifest files match CEDA size and MD5.'
