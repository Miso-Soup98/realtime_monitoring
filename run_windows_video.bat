@echo off
setlocal
chcp 65001 >nul

echo ======================================
echo  Waveform Monitor - Video Mode
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

echo [4/5] Running detection script (video mode)...
call .venv\Scripts\python.exe monitor_waveform.py --source da3fd50e6979f4b2cc582dfe0695445b.mp4 --preview

echo [5/5] Finished.
pause
endlocal
