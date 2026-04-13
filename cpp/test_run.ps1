$env:PATH = "$PSScriptRoot\build\vcpkg_installed\x64-windows\bin;$env:PATH"
& "$PSScriptRoot\build\Release\emg_3d_heatmap.exe" 2>&1
Write-Host "Exit code: $LASTEXITCODE"
Read-Host "Press Enter to close"
