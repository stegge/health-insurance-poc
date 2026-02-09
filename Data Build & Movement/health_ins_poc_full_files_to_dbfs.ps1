<# Copy-To-Volume.ps1
   Copies a local folder into a date-stamped subfolder under a Databricks Volume.
   Requires: Databricks CLI authenticated (run once: `databricks auth login`).
#>

param(
  [string]$LocalPath = "C:\Users\STEGGE\Desktop\health_ins_poc_full\files",
  [string]$BaseDest  = "dbfs:/Volumes/workspace/tegge-insurance-data/health_ins_poc_raw",
  [switch]$Overwrite,
  [string]$DBXProfile   = "https://dbc-1522605f-073b.cloud.databricks.com"  # default to your saved profile name
)

# --- set the CLI profile for this process (inherited by child processes)
$env:DATABRICKS_CONFIG_PROFILE = $DBXProfile

# Build date-stamped destination (YYYY_MM_DD)
$stamp = Get-Date -Format 'yyyy_MM_dd'
$dest  = "$BaseDest/$stamp"

Write-Host "Source: $LocalPath"
Write-Host "Destination: $dest"

if (-not (Test-Path -LiteralPath $LocalPath)) {
  throw "Local path not found: $LocalPath"
}

# Ensure destination directory exists
databricks fs mkdirs $dest

# Copy recursively; include overwrite flag if requested
# Old (breaks on Windows PowerShell 5.1)
# $ow = $Overwrite.IsPresent ? "--overwrite" : ""

# New (works on 5.1 and 7+)
if ($Overwrite.IsPresent) {
    $ow = "--overwrite"
} else {
    $ow = ""
}

$cmd = "databricks fs cp $ow `"$LocalPath`" `"$dest`" -r"
Write-Host "Running: $cmd"
Invoke-Expression $cmd

if ($LASTEXITCODE -ne 0) {
  throw "Copy failed (exit code $LASTEXITCODE)."
}

Write-Host "✅ Copy complete. Listing destination:"
databricks fs ls $dest