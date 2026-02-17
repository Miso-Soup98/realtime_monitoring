# realtime_monitoring

这是一个用于“波形窗口状态识别”的最小项目：

- 当画面中出现**大面积、上下都比较宽的波形块**时，输出 `1`
- 当画面中只出现**细长条形波形**时，输出 `0`

仓库中默认视频：`da3fd50e6979f4b2cc582dfe0695445b.mp4`。

## 1. 安装依赖

```bash
pip install opencv-python numpy mss
```

> 说明：在当前执行环境里网络受限，无法直接安装这些包；你在自己的电脑上执行即可。

## 2. 用录屏视频验证

```bash
python monitor_waveform.py --source da3fd50e6979f4b2cc582dfe0695445b.mp4
```

可选加预览窗口：

```bash
python monitor_waveform.py --source da3fd50e6979f4b2cc582dfe0695445b.mp4 --preview
```

## 3. 实时监测电脑屏幕

先把要监测的软件窗口放在固定区域，然后指定采样区域：

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

- `--screen-*`：截取整个屏幕中的一个矩形区域
- `--roi`：在这个截屏区域里进一步只看波形窗口（推荐使用）

## 4. 识别逻辑（简化版）

脚本每帧会计算三个指标：

- `active_col_ratio`：有明显墨迹（波形）的列占比
- `vertical_fill_ratio`：墨迹像素占比
- `row_peaks`：横向上“波形密集”的行占比

然后根据阈值判断输出 `1` 或 `0`，再用 `--hold-frames` 做轻量去抖。

如果你朋友真实实验画面和录屏差异较大，可微调 `DetectorConfig` 里各阈值。
