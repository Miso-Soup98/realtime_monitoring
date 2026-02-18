@echo off
setlocal
chcp 65001 >nul

echo ======================================
echo  打包单文件 EXE（给不会电脑的人）

echo ======================================

echo [1/6] 检查 Python...
python --version >nul 2>nul
if errorlevel 1 (
  echo 未检测到 Python，请先安装：
  echo https://www.python.org/downloads/windows/
  echo 安装时请勾选 "Add python.exe to PATH"。
  pause
  exit /b 1
)

echo [2/6] 创建虚拟环境 .venv_build ...
if not exist .venv_build (
  python -m venv .venv_build
)

echo [3/6] 安装项目依赖...
call .venv_build\Scripts\python.exe -m pip install --upgrade pip
call .venv_build\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo 依赖安装失败。
  pause
  exit /b 1
)

echo [4/6] 安装 PyInstaller...
call .venv_build\Scripts\python.exe -m pip install pyinstaller
if errorlevel 1 (
  echo PyInstaller 安装失败。
  pause
  exit /b 1
)

echo [5/6] 生成 exe ...
call .venv_build\Scripts\pyinstaller.exe --onefile --name waveform_monitor monitor_waveform.py
if errorlevel 1 (
  echo 打包失败。
  pause
  exit /b 1
)

echo [6/6] 打包成功！
echo EXE 路径：dist\waveform_monitor.exe
echo 你可以把 dist\waveform_monitor.exe + mp4 文件一起发给朋友。
pause
endlocal
