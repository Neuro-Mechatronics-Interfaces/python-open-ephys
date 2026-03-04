@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PY_EXE=%SCRIPT_DIR%..\..\..\.venv\Scripts\python.exe"
set "GUI_SCRIPT=%SCRIPT_DIR%scripts\npz_angle_compare_gui.py"

if not exist "%PY_EXE%" (
  echo [ERROR] Expected Python interpreter not found:
  echo   %PY_EXE%
  echo.
  echo Activate the correct virtual environment first, or update this launcher path.
  pause
  exit /b 1
)

if not exist "%GUI_SCRIPT%" (
  echo [ERROR] GUI script not found:
  echo   %GUI_SCRIPT%
  pause
  exit /b 1
)

"%PY_EXE%" "%GUI_SCRIPT%"
