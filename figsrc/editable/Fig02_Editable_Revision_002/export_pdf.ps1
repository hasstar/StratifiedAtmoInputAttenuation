param(
    [string]$InputPptx = (Join-Path $PSScriptRoot 'Fig02_Stratified_Atmosphere_Editable.pptx'),
    [string]$OutputPdf = (Join-Path $PSScriptRoot 'Fig02_Stratified_Atmosphere_Editable.pdf')
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $InputPptx)) { throw 'Input PPTX missing.' }
if (Test-Path -LiteralPath $OutputPdf) { throw 'Output already exists; choose a new output filename to preserve prior exports.' }
if (Get-Process POWERPNT -ErrorAction SilentlyContinue) { throw 'Close PowerPoint first so this export cannot affect another open presentation.' }
$figApp = $null
$figDeck = $null
try {
    $figApp = New-Object -ComObject PowerPoint.Application
    $figDeck = $figApp.Presentations.Open($InputPptx, $true, $false, $false)
    $figDeck.SaveAs($OutputPdf, 32)
    [pscustomobject]@{Slides=$figDeck.Slides.Count; Shapes=$figDeck.Slides.Item(1).Shapes.Count; Pdf=$OutputPdf}
} finally {
    if ($null -ne $figDeck) { $figDeck.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($figDeck) }
    if ($null -ne $figApp) { $figApp.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($figApp) }
}
