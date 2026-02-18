# realtime_monitoring（小白友好版）

这个项目的目标：

- 出现**大面积、上下都宽**的波形块 => 输出 `1`
- 出现**细长条形**波形 => 输出 `0`

仓库已包含示例录屏：`da3fd50e6979f4b2cc582dfe0695445b.mp4`。

---

## 0. 你提到的“朦胧全屏 + 红框拖拽 + 叉/勾确认”已经支持

现在屏幕实时模式启动后，流程是：

1. 全屏画面会变成模糊（像磨砂玻璃）
2. 只有 ROI 检测框内部保持清晰，边框是红色
3. 你可以：
   - 拖动框内部：移动检测框
   - 拖动框边缘/角：调整大小
4. 检测框底部有两个按钮：
   - 左下角 **✕**：取消并退出
   - 右下角 **✓**：确认后开始检测
5. 确认后，屏幕恢复正常，程序进入实时输出

---

## 1. 完全不会计算机也能用（Windows 推荐）

> 不需要 conda，只需要装 Python，然后双击 `.bat` 文件。

### 第一步：安装 Python（只做一次）

1. 打开：<https://www.python.org/downloads/windows/>
2. 下载 Python 3.10+（新版本也可以）
3. 安装窗口中一定要勾选：`Add python.exe to PATH`
4. 一路下一步安装完成

### 第二步：拿到项目文件

解压后确保有这些文件：

- `monitor_waveform.py`
- `requirements.txt`
- `run_windows_video.bat`
- `run_windows_screen.bat`
- `da3fd50e6979f4b2cc582dfe0695445b.mp4`

### 第三步：双击运行

#### 方案 A：先验证视频（最稳妥）

双击：`run_windows_video.bat`

首次会自动完成：

- 创建独立环境 `.venv`
- 安装依赖
- 启动检测

#### 方案 B：实时监测屏幕（你最关心）

双击：`run_windows_screen.bat`

会自动进入“全屏模糊 + 红框交互选区”流程，确认后开始实时识别。

---

## 2. 依赖安装（详细解释）

依赖在 `requirements.txt`：

- `opencv-python`：图像处理和窗口显示
- `numpy`：矩阵计算
- `mss`：屏幕截图

脚本已经会自动安装依赖。若你想手动装：

```bash
pip install -r requirements.txt
```

如果报网络问题，通常是单位网络限制；可换家庭网络或手机热点重试。

---

## 3. 真·即插即用（朋友连 Python 都不用装）

你可以把项目打包成单文件 `.exe` 发给他：

1. 在你的电脑双击 `build_windows_exe.bat`
2. 等待完成
3. 会生成 `dist\waveform_monitor.exe`
4. 把这个 exe 和 mp4 一起发给你朋友

这样他只要双击 exe 就能运行。

> 注意：如果朋友电脑缺少某些 VC 运行库，Windows 可能会提示安装组件（常见于任何 Python 打包程序）。

---


### 如果双击 `build_windows_exe.bat` “没有任何反应”

请按下面顺序排查（非常常见）：

1. **先解压 zip 再运行**，不要在压缩包预览窗口里直接双击。
2. 右键 `build_windows_exe.bat` -> `以管理员身份运行`。
3. 等待脚本执行，失败时看同目录 `build_exe.log`。
4. 把 `build_exe.log` 发给我，我可以精确告诉你卡在哪一步。

新版脚本已经自动：

- 切换到脚本所在目录（避免路径错）
- 每一步写日志到 `build_exe.log`
- 出错会停在窗口，不会一闪而过

## 4. Mac / Linux 一键运行

```bash
chmod +x run_macos_linux.sh
./run_macos_linux.sh
```

---

## 5. 手动命令（给会一点命令行的人）

### 视频模式

```bash
python monitor_waveform.py --source da3fd50e6979f4b2cc582dfe0695445b.mp4 --preview
```

### 屏幕模式（交互选框）

```bash
python monitor_waveform.py --screen --interactive-roi --preview
```

---

## 6. 输出怎么看

终端会打印类似：

```text
ts=1710000000.123 output=1 active_col_ratio=0.301 vertical_fill=0.421
```

- `output=1`：宽厚波形块
- `output=0`：细长条形波形

---

## 7. 识别逻辑（简版）

每帧计算三个指标：

- `active_col_ratio`：有明显波形痕迹的列占比
- `vertical_fill_ratio`：波形像素占比
- `row_peaks`：横向高密度行占比

根据阈值判定 `1/0`，再用 `--hold-frames` 去抖动。
