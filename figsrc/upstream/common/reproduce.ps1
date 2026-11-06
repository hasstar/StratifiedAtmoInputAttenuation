param(
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [string]$SourcePackage = (Split-Path -Parent $PSScriptRoot),
  [string]$ScientificPython = 'D:\Anaconda202410\python.exe',
  [string]$DocumentPython = 'C:\Users\SDR_HH\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
  [string]$Node = 'C:\Users\SDR_HH\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe',
  [string]$PdfLatex = 'D:\latex\texlive\2024\bin\windows\pdflatex.exe'
)
# Re-render only. Never calls old workers, training, inference or network tools.
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $SourcePackage).Path
$destinationPath = [IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $destinationPath) { throw 'OutputDir must not exist; use a new directory.' }
if ($destinationPath.StartsWith($sourcePath + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Choose an output outside the frozen source package.' }
if (-not (Test-Path -LiteralPath (Join-Path $sourcePath '04_Source_Data\frozen_exports\fast_summary.csv'))) { throw 'SourcePackage is not the delivered plotting package.' }
foreach ($executable in @($ScientificPython,$DocumentPython,$Node,$PdfLatex)) {
  if (-not (Test-Path -LiteralPath $executable)) { throw "Existing executable unavailable: $executable" }
}
function Invoke-Checked([string]$Exe,[string[]]$Arguments) {
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Rendering step failed with exit $LASTEXITCODE : $Exe" }
}
New-Item -ItemType Directory -Path $destinationPath | Out-Null
Copy-Item -LiteralPath (Join-Path $sourcePath '04_Source_Data') -Destination $destinationPath -Recurse
Copy-Item -LiteralPath (Join-Path $sourcePath '06_Generation_Code') -Destination $destinationPath -Recurse
foreach ($name in @('01_Figures_PDF','02_Tables_LaTeX','03_Editable_PPTX','05_QA','05_QA\previews','05_QA\latex_build')) {
  New-Item -ItemType Directory -Path (Join-Path $destinationPath $name) -Force | Out-Null
}
foreach ($name in @('SOURCE_MANIFEST.json','UPSTREAM_CODE_MANIFEST.json','Fast003_hour_receipts.json','duration_distribution_qa.json','dense_acf_qa.json')) {
  Copy-Item -LiteralPath (Join-Path $sourcePath "05_QA\$name") -Destination (Join-Path $destinationPath '05_QA')
}
Copy-Item -LiteralPath (Join-Path $sourcePath '02_Tables_LaTeX\main.tex') -Destination (Join-Path $destinationPath '02_Tables_LaTeX')
$codeDir = Join-Path $destinationPath '06_Generation_Code'
Invoke-Checked $ScientificPython @('-X','utf8',(Join-Path $codeDir 'build_plots.py'),'--package',$destinationPath)
Invoke-Checked $Node @((Join-Path $codeDir 'build_diagrams.mjs'),'--package',$destinationPath,'--finalize')
Invoke-Checked $ScientificPython @('-X','utf8',(Join-Path $codeDir 'build_tables.py'),'--package',$destinationPath)
Invoke-Checked $DocumentPython @('-X','utf8',(Join-Path $codeDir 'assemble_pdfs.py'),'--package',$destinationPath)
$tableDir = Join-Path $destinationPath '02_Tables_LaTeX'
$texOutput = Join-Path $destinationPath '05_QA\latex_build'
Push-Location -LiteralPath $tableDir
try {
  for ($pass = 0; $pass -lt 2; $pass++) {
    Invoke-Checked $PdfLatex @('-interaction=nonstopmode','-halt-on-error',"-output-directory=$texOutput",'main.tex')
  }
} finally { Pop-Location }
Copy-Item -LiteralPath (Join-Path $texOutput 'main.pdf') -Destination (Join-Path $destinationPath '00_表格预览.pdf')
Invoke-Checked $ScientificPython @('-X','utf8',(Join-Path $codeDir 'verify_delivery.py'),'--package',$destinationPath,'--skip-originals','--render-replay')
@{
  source = $sourcePath
  output = $destinationPath
  time_utc = [DateTime]::UtcNow.ToString('o')
  scope = 'Render saved data only; no training, inference, generation or refitting'
  visual_review = 'Required after reproduction or any editing; original package review does not certify changed output'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destinationPath '05_QA\REPRODUCE_RECEIPT.json') -Encoding UTF8
Write-Output "Completed figures, native PPTX and LaTeX tables: $destinationPath"
