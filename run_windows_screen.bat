@echo off
setlocal
chcp 65001 >nul

echo ======================================
echo  Waveform Monitor - Screen Real-time Mode
echo ======================================

echo [1/5] Checking Python...
python --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python not detected.
  echo Please install Python 3.10+:
  echo https://www.python.org/downloads/windows/
  echo Check "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)

echo [2/5] Creating virtual environment .venv (first time may be slow)...
if not exist .venv (
  python -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment. Please ensure Python is installed correctly.
    pause
    exit /b 1
  )
)

echo [3/5] Installing dependencies (first time may be slow)...
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install dependencies. Please check network and retry.
  pause
  exit /b 1
)

echo [4/5] Starting interactive ROI selection + real-time detection...
echo Drag the red box with mouse to move/resize, bottom-left X=cancel, bottom-right checkmark=confirm.
echo Press Enter to confirm, Esc to cancel
python monitor_waveform.py --screen --interactive-roi --preview

echo [5/5] Finished.
pause
endlocal
