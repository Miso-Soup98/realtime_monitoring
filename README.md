# realtime_monitoring（小白友好版）

这个项目的目标非常简单：

- 看到**大面积、上下都比较宽**的波形块 => 输出 `1`
- 看到**细长条形**波形 => 输出 `0`

仓库里已经放好了示例录屏：`da3fd50e6979f4b2cc582dfe0695445b.mp4`。

---

## 一、给完全不会编程的人：最省事用法（推荐）

> 你的朋友如果是 Windows，基本只要“安装 Python + 双击 `.bat` 文件”就能用，不需要 conda。

### 1) 只做一次：安装 Python

1. 打开官网：<https://www.python.org/downloads/windows/>
2. 点击下载最新版 Python（3.10+ 都行）
3. **安装时务必勾选**：`Add python.exe to PATH`
4. 一路 `Next` 安装完成

### 2) 下载并解压本项目

- 方式 A：你把整个项目文件夹打包发给他（zip）
- 方式 B：他自己下载 zip

解压后，进入项目目录，应该能看到这些文件：

- `monitor_waveform.py`
- `requirements.txt`
- `run_windows_video.bat`
- `run_windows_screen.bat`
- `da3fd50e6979f4b2cc582dfe0695445b.mp4`

### 3) 运行视频识别（最简单）

- 直接双击：`run_windows_video.bat`
- 首次运行会自动：
  - 创建独立环境 `.venv`
  - 自动安装依赖
  - 启动检测

运行时终端会打印类似：

```text
frame=000123 output=1 active_col_ratio=0.301 vertical_fill=0.421
```

其中 `output=1/0` 就是结果。

### 4) 运行实时屏幕监测

- 双击：`run_windows_screen.bat`
- 会调用实时屏幕模式（默认参数可改）

如果检测框位置不对，编辑 `run_windows_screen.bat` 最后一行这些参数：

- `--screen-left/--screen-top/--screen-width/--screen-height`：屏幕大框
- `--roi x,y,w,h`：在大框里再截出波形窗口（推荐）

---

## 二、为什么不需要 conda？

因为这个项目已经使用了 **Python 自带虚拟环境 venv**：

- 自动创建：`.venv`
- 依赖自动安装到这个目录
- 不污染系统 Python
- 不需要学习 conda 指令

---

## 三、Mac / Linux 用法

在项目目录执行：

```bash
chmod +x run_macos_linux.sh
./run_macos_linux.sh
```

脚本会自动创建 `.venv`、安装依赖、启动视频模式。

---

## 四、手动命令（给会一点命令行的人）

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：

- `opencv-python`
- `numpy`
- `mss`

### 2) 用录屏视频验证

```bash
python monitor_waveform.py --source da3fd50e6979f4b2cc582dfe0695445b.mp4
```

加预览窗口：

```bash
python monitor_waveform.py --source da3fd50e6979f4b2cc582dfe0695445b.mp4 --preview
```

### 3) 实时屏幕模式

```bash
python monitor_waveform.py \
  --screen \
  --screen-left 100 \
  --screen-top 120 \
  --screen-width 900 \
  --screen-height 650 \
  --roi 120,80,600,420 \
  --preview
```

---

## 五、即插即用还能再进一步吗？（可选）

可以，有两种升级路线：

1. **打包成 Windows 单文件 exe（推荐）**
   - 用户无需安装 Python，双击 exe 即可
   - 适合“真正零配置”交付
2. **做一个带按钮的小 GUI**
   - 比命令行更直观，适合完全不懂技术的人

如果你愿意，我可以在下一版直接帮你加上“`build_exe.bat` 一键打包脚本”，你在自己电脑执行一次后，把 exe 发给你朋友即可。

---

## 六、识别逻辑（简化说明）

脚本每帧会计算：

- `active_col_ratio`：有明显“墨迹”（波形）的列占比
- `vertical_fill_ratio`：墨迹像素占比
- `row_peaks`：横向上波形密集行占比

然后按阈值判断 `1/0`，再用 `--hold-frames` 做去抖动，减少跳变。

如果实际实验画面和录屏差异较大，可在 `DetectorConfig` 内微调阈值。
