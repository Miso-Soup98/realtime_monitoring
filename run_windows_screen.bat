@echo off
setlocal
chcp 65001 >nul

echo ======================================
echo  波形监测 - 屏幕实时模式 一键启动

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

echo [4/5] 运行检测脚本（屏幕模式）...
echo 你可以按实际窗口位置修改下面参数：
call .venv\Scripts\python.exe monitor_waveform.py --screen --screen-left 100 --screen-top 120 --screen-width 900 --screen-height 650 --roi 120,80,600,420 --preview

echo [5/5] 运行结束。
pause
endlocal
