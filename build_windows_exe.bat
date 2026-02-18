@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM 切换到 bat 所在目录，避免双击时工作目录不正确
cd /d "%~dp0"

echo ======================================
echo  打包单文件 EXE（给不会电脑的人）
echo ======================================
echo 当前目录：%cd%

echo.
echo 如果窗口一闪而过，请右键本文件 -^>“以管理员身份运行”再试。
echo 详细日志会写入：build_exe.log
echo.

set "LOGFILE=build_exe.log"
echo [INFO] Build started at %date% %time% > "%LOGFILE%"

echo [1/7] 检查 Python...
python --version >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] 未检测到 Python。
  echo 请先安装：https://www.python.org/downloads/windows/
  echo 安装时请勾选 "Add python.exe to PATH"。
  echo [ERROR] Python not found. >> "%LOGFILE%"
  goto :fail
)

echo [2/7] 创建虚拟环境 .venv_build ...
if not exist ".venv_build\Scripts\python.exe" (
  python -m venv .venv_build >> "%LOGFILE%" 2>&1
  if errorlevel 1 (
    echo [ERROR] 创建虚拟环境失败。
    echo [ERROR] venv creation failed. >> "%LOGFILE%"
    goto :fail
  )
)

echo [3/7] 升级 pip ...
call .venv_build\Scripts\python.exe -m pip install --upgrade pip >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] pip 升级失败。
  echo [ERROR] pip upgrade failed. >> "%LOGFILE%"
  goto :fail
)

echo [4/7] 安装项目依赖...
call .venv_build\Scripts\python.exe -m pip install -r requirements.txt >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] 项目依赖安装失败（可能是网络问题）。
  echo [ERROR] requirements install failed. >> "%LOGFILE%"
  goto :fail
)

echo [5/7] 安装 PyInstaller...
call .venv_build\Scripts\python.exe -m pip install pyinstaller >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] PyInstaller 安装失败。
  echo [ERROR] pyinstaller install failed. >> "%LOGFILE%"
  goto :fail
)

echo [6/7] 生成 EXE ...
call .venv_build\Scripts\pyinstaller.exe --noconfirm --clean --onefile --name waveform_monitor monitor_waveform.py >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [ERROR] 打包失败，请查看 %LOGFILE%。
  echo [ERROR] pyinstaller build failed. >> "%LOGFILE%"
  goto :fail
)

echo [7/7] 打包成功！
echo EXE 路径：dist\waveform_monitor.exe
echo 日志路径：%LOGFILE%
echo 你可以把 dist\waveform_monitor.exe + mp4 文件一起发给朋友。
echo [INFO] Build succeeded at %date% %time% >> "%LOGFILE%"
goto :end

:fail
echo.
echo 打包未完成。请把 %LOGFILE% 发给我，我可以帮你定位问题。

:end
echo.
pause
endlocal
