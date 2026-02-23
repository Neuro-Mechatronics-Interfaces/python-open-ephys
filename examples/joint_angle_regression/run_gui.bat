@echo off
REM Joint Angle Regression Data Collection GUI
REM Neuro-Mechatronics Lab (NML)
REM
REM Prerequisites:
REM   1. Open Ephys GUI running with ZMQ Interface plugin enabled
REM   2. Hand tracking system broadcasting joint angles via LSL

echo.
echo ========================================
echo Joint Angle Regression Data Collection
echo ========================================
echo.
echo This GUI collects synchronized EMG and joint angle data
echo for regression model training.
echo.
echo Before continuing, verify:
echo   [x] Open Ephys GUI is running
echo   [x] ZMQ Interface plugin is enabled
echo   [x] Hand tracking system is broadcasting via LSL
echo.
echo Default connection settings:
echo   ZMQ Host: 127.0.0.1
echo   ZMQ Port: 5556
echo   EMG Sampling Rate: 5000 Hz
echo   Number of Channels: 8
echo.
echo Press Ctrl+C to cancel, or
pause

python new_session_gui.py %*
