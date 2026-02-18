@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM Change to bat file directory to avoid incorrect working directory when double-clicking
cd /d "%~dp0"

echo ======================================
echo  Build Single-file EXE
echo ======================================
echo Current directory: %cd%

echo.
echo If window closes immediately, right-click this file -^> "Run as administrator" and try again.
echo Detailed log will be written to: build_exe.log
echo.

set "LOGFILE=build_exe.log"
echo [INFO] Build started at %date% %time% > "%LOGFILE%"

echo [1/7] Checking Python...
python --version >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] Python not detected.
  echo Please install: https://www.python.org/downloads/windows/
  echo Check "Add python.exe to PATH" during installation.
  echo [ERROR] Python not found. >> "%LOGFILE%"
  goto :fail
)

echo [2/7] Creating virtual environment .venv_build ...
if not exist ".venv_build\Scripts\python.exe" (
  python -m venv .venv_build >> "%LOGFILE%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    echo [ERROR] venv creation failed. >> "%LOGFILE%"
    goto :fail
  )
)

echo [3/7] Upgrading pip ...
call .venv_build\Scripts\python.exe -m pip install --upgrade pip >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.
  echo [ERROR] pip upgrade failed. >> "%LOGFILE%"
  goto :fail
)

echo [4/7] Installing project dependencies...
call .venv_build\Scripts\python.exe -m pip install -r requirements.txt >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] Failed to install project dependencies (may be network issue).
  echo [ERROR] requirements install failed. >> "%LOGFILE%"
  goto :fail
)

echo [5/7] Installing PyInstaller...
call .venv_build\Scripts\python.exe -m pip install pyinstaller >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] Failed to install PyInstaller.
  echo [ERROR] pyinstaller install failed. >> "%LOGFILE%"
  goto :fail
)

echo [6/7] Building EXE ...
call .venv_build\Scripts\pyinstaller.exe --noconfirm --clean --onefile --name waveform_monitor monitor_waveform.py >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] Build failed, please check %LOGFILE%.
  echo [ERROR] pyinstaller build failed. >> "%LOGFILE%"
  goto :fail
)

echo [7/7] Build successful!
echo EXE path: dist\waveform_monitor.exe
echo Log path: %LOGFILE%
echo You can share dist\waveform_monitor.exe + mp4 file together.
echo [INFO] Build succeeded at %date% %time% >> "%LOGFILE%"
goto :end

:fail
echo.
echo Build incomplete. Please send %LOGFILE% to me so I can help identify the issue.

:end
echo.
pause
endlocal
