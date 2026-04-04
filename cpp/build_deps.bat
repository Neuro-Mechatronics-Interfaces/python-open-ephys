@echo off
echo Setting up Build Tools environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo ERROR: vcvarsall.bat failed
    exit /b 1
)

echo.
echo Compiler: %VCToolsInstallDir%
echo.

echo Installing vcpkg dependencies...
cd /d "%~dp0"
C:\vcpkg\vcpkg.exe install --triplet x64-windows
if errorlevel 1 (
    echo ERROR: vcpkg install failed
    exit /b 1
)

echo.
echo SUCCESS: All dependencies installed
