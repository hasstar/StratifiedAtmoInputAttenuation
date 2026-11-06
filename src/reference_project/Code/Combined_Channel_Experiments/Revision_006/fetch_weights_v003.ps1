$ErrorActionPreference = 'Stop'
$projectRoot = 'E:\0_Workflow_System\01_PhD\Papers\0_TVT_GEO_Prediction'
$weightRoot = Join-Path $projectRoot 'Code\Combined_Channel_Experiments\Revision_006\weights'
$referenceRoot = Join-Path $projectRoot 'Outputs\Experiments\Combined_Channel_Revision_006\references_003'
New-Item -ItemType Directory -Path $weightRoot -Force | Out-Null
New-Item -ItemType Directory -Path $referenceRoot -Force | Out-Null

function Get-PublicArtifact($url, $destination, $expectedHash, $expectedBytes) {
    if (Test-Path -LiteralPath $destination) { throw "Destination exists: $destination" }
    $partPath = $destination + '.partial'
    if (Test-Path -LiteralPath $partPath) { throw "Partial preserved: $partPath" }
    Invoke-WebRequest -Uri $url -OutFile $partPath -Headers @{ 'Accept-Encoding' = 'identity' } -TimeoutSec 180
    $actualBytes = (Get-Item -LiteralPath $partPath).Length
    if ($actualBytes -gt 268435456) { throw 'Size cap exceeded' }
    if ($expectedBytes -and $actualBytes -ne $expectedBytes) { throw 'Repository length mismatch' }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $partPath).Hash.ToLower()
    if ($expectedHash -and $actualHash -ne $expectedHash) { throw 'Repository SHA mismatch' }
    # Exact nonrecursive targets in the newly declared artifact directories.
    if (-not ([IO.Path]::GetFullPath($destination).StartsWith($weightRoot + '\') -or [IO.Path]::GetFullPath($destination).StartsWith($referenceRoot + '\'))) { throw 'Path boundary' }
    Move-Item -LiteralPath $partPath -Destination $destination
    [pscustomobject]@{Url=$url;Path=$destination;Bytes=$actualBytes;SHA256=$actualHash} | ConvertTo-Json -Compress
}

$pfnRev = '4972a65a1b30806315c6f92499959ffbfc69a673'
$iclRev = '4dcd344ece2c00be9e831fdd35bed57b5ad83e19'
Get-PublicArtifact "https://huggingface.co/api/models/Prior-Labs/TabPFN-v2-reg/revision/$pfnRev`?blobs=true" (Join-Path $referenceRoot 'tabpfn_hub.json') $null $null
Get-PublicArtifact "https://huggingface.co/api/models/jingang/TabICL/revision/$iclRev`?blobs=true" (Join-Path $referenceRoot 'tabicl_hub.json') $null $null
Get-PublicArtifact "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/$pfnRev/LICENSE.txt" (Join-Path $referenceRoot 'tabpfn_LICENSE.txt') $null 11483
Get-PublicArtifact "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/$pfnRev/README.md" (Join-Path $referenceRoot 'tabpfn_README.md') $null 2149
Get-PublicArtifact "https://huggingface.co/jingang/TabICL/resolve/$iclRev/README.md" (Join-Path $referenceRoot 'tabicl_README.md') $null $null
Get-PublicArtifact "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/$pfnRev/tabpfn-v2-regressor.ckpt" (Join-Path $weightRoot 'tabpfn-v2-regressor.ckpt') '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736' 44390977
Get-PublicArtifact "https://huggingface.co/jingang/TabICL/resolve/$iclRev/tabicl-regressor-v2-20260212.ckpt" (Join-Path $weightRoot 'tabicl-regressor-v2-20260212.ckpt') '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a' 114324594
