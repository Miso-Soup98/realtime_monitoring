#!/usr/bin/env bash
set -e

echo "======================================"
echo " 波形监测 一键启动（macOS/Linux）"
echo "======================================"

echo "[1/4] 检查 Python..."
python3 --version

echo "[2/4] 创建虚拟环境 .venv（首次会稍慢）..."
[ -d .venv ] || python3 -m venv .venv

echo "[3/4] 安装依赖（首次会稍慢）..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo "[4/4] 启动视频模式..."
./.venv/bin/python monitor_waveform.py --source da3fd50e6979f4b2cc582dfe0695445b.mp4 --preview
