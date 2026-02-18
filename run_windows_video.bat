@echo off
setlocal
chcp 65001 >nul

echo ======================================
echo  波形监测 - 视频模式 一键启动

echo ======================================

echo [1/5] 检查 Python...
python --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo 未检测到 Python。
  echo 请先安装 Python 3.10+：
  echo https://www.python.org/downloads/windows/
  echo 安装时请勾选 "Add python.exe to PATH"。
  pause
  exit /b 1
)

echo [2/5] 创建虚拟环境 .venv（首次会稍慢）...
if not exist .venv (
  python -m venv .venv
  if errorlevel 1 (
    echo 创建虚拟环境失败，请确认 Python 安装完整。
    pause
    exit /b 1
  )
)

echo [3/5] 安装依赖（首次会稍慢）...
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo 依赖安装失败，请检查网络后重试。
  pause
  exit /b 1
)

echo [4/5] 运行检测脚本（视频模式）...
call .venv\Scripts\python.exe monitor_waveform.py --source da3fd50e6979f4b2cc582dfe0695445b.mp4 --preview

echo [5/5] 运行结束。
pause
endlocal
