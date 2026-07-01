<#
.SYNOPSIS
    One-click launcher for an sEMG sleeve data-collection session.

.DESCRIPTION
    Opens a single Windows Terminal window with 4 tabs (one per script,
    each running with its package's own venv) and starts LabRecorder as
    a separate GUI window.

    Tabs:
      1. Open Ephys LSL streamer (python-open-ephys .venv)
      2. MoCap hand tracking GUI (Hand-Landmark-Tracker .venv)
      3. Cue player              (python-open-ephys .venv)
      4. BLE IMU GUI             (TinyNML .tinynml)

    The Open Ephys GUI itself is launched manually.

    To stop one script: focus its tab and press Ctrl+C.
    To stop everything: close the Windows Terminal window.

.NOTES
    Requires Windows Terminal (`wt.exe`) -- ships with Windows 11.
    Run via the sibling launch_semg_session.bat (double-click) or
    `powershell -ExecutionPolicy Bypass -File launch_semg_session.ps1`.
#>

$ErrorActionPreference = "Stop"

# ---- Paths (edit here if anything moves) ------------------------------------
$Root      = "C:\Users\NML\Documents\Github"

$EphysPy   = "$Root\python-open-ephys\.venv\Scripts\python.exe"
$HandPy    = "$Root\Hand-Landmark-Tracker\.venv\Scripts\python.exe"
$TinyPy    = "$Root\TinyNML\.tinynml\Scripts\python.exe"

$Streamer  = "$Root\python-open-ephys\examples\interface\lsl\open_ephys_lsl_streamer.py"
$Mocap     = "$Root\Hand-Landmark-Tracker\src\handtrack\applications\optitrack_gui.py"
$Cues      = "$Root\python-open-ephys\examples\applications\cue_player\cue_player.py"
$BleImu    = "$Root\TinyNML\mg24_imu\client\ble_imu_gui.py"
$LabRec    = "C:\Users\NML\Documents\LabRecorder\LabRecorder.exe"

# ---- Sanity check -----------------------------------------------------------
$required = @($EphysPy, $HandPy, $TinyPy, $Streamer, $Mocap, $Cues, $BleImu, $LabRec)
$missing  = $required | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host "Missing required paths:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

# ---- Build wt argument string -----------------------------------------------
# Each tab: cd into the script's folder, then run "<venv-python> <script>"
# under PowerShell -NoExit so the window stays open after the script ends.
function New-Tab([string]$Title, [string]$PyExe, [string]$Script) {
    $cwd = Split-Path -Parent $Script
    # cmd /k keeps the window open after the script exits (like PowerShell -NoExit).
    return "new-tab --title `"$Title`" -d `"$cwd`" cmd /k `"`"$PyExe`" `"$Script`"`""
}

$tabs = @(
    (New-Tab "Open Ephys LSL" $EphysPy $Streamer),
    (New-Tab "MoCap Hand"     $HandPy  $Mocap),
    (New-Tab "Cue Player"     $EphysPy $Cues),
    (New-Tab "BLE IMU"        $TinyPy  $BleImu)
) -join " ; "

# -w 0 reuses the currently-focused wt window if you re-run; omit for a new window.
$wtArgs = "-w 0 $tabs"

Write-Host "[launcher] Opening Windows Terminal with 4 tabs..." -ForegroundColor Cyan
Start-Process -FilePath "wt.exe" -ArgumentList $wtArgs

Write-Host "[launcher] Starting LabRecorder GUI..." -ForegroundColor Cyan
Start-Process -FilePath $LabRec

Write-Host ""
Write-Host "Session launched. Reminder:" -ForegroundColor Green
Write-Host "  * Open Ephys GUI must be started manually."
Write-Host "  * In LabRecorder, hit 'Update' so it discovers all LSL streams,"
Write-Host "    pick the save path, then 'Start' BEFORE pressing SPACE in the Cue Player."
Write-Host "  * Close the wt window to terminate all 4 scripts at once."
