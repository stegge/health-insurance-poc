<#
    pipeline_2.ps1 (minimal, synchronous)
    Runs Python generator, then DBFS loader.
    Exit codes: 0=OK, 10=Python not found, 11=Python failed, 12=Loader failed.
#>

[CmdletBinding()]
param(
    [string]$PythonScriptPath = "C:\Users\STEGGE\Desktop\script\full_load\health_ins_poc_file_gen.py",
    [string]$PythonExe = "python",
    [string]$DbfsLoaderScriptPath = "C:\Users\STEGGE\Desktop\script\full_load\health_ins_poc_full_files_to_dbfs.ps1",
    [string]$WorkingDir = "C:\Users\STEGGE\Desktop\health_ins_poc_full\files"
)

# --- Logging ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$LogFile = Join-Path $LogDir "pipeline_2_$stamp.log"
Start-Transcript -Path $LogFile -Append | Out-Null
Write-Host "== Run started: $(Get-Date -Format o) =="

try {
    # --- Check Python ---
    $pythonVersion = & $PythonExe --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python not found or not runnable via '$PythonExe'. Output: $pythonVersion"
        Stop-Transcript | Out-Null
        exit 10
    }
    Write-Host "Python version: $pythonVersion"

    # --- Run Python generator ---
    if (-not (Test-Path $PythonScriptPath)) {
        Write-Error "Python script not found: $PythonScriptPath"
        Stop-Transcript | Out-Null
        exit 11
    }

    Push-Location $WorkingDir
    Write-Host "Running Python generator..."
    & $PythonExe $PythonScriptPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python generator failed. ExitCode=$LASTEXITCODE"
        Pop-Location
        Stop-Transcript | Out-Null
        exit 11
    }
    Write-Host "Python generator completed successfully."

    # --- Run DBFS loader ---
    if (-not (Test-Path $DbfsLoaderScriptPath)) {
        Write-Error "DBFS loader script not found: $DbfsLoaderScriptPath"
        Pop-Location
        Stop-Transcript | Out-Null
        exit 12
    }

    Write-Host "Running DBFS loader..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $DbfsLoaderScriptPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "DBFS loader failed. ExitCode=$LASTEXITCODE"
        Pop-Location
        Stop-Transcript | Out-Null
        exit 12
    }
    Write-Host "DBFS loader completed successfully."
    Pop-Location

    Write-Host "== Run finished OK: $(Get-Date -Format o) =="
    Stop-Transcript | Out-Null
    exit 0
}
catch {
    Write-Error $_
    try { Pop-Location } catch {}
    Stop-Transcript | Out-Null
    exit 1
}