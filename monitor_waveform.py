#!/usr/bin/env python3
"""实时识别波形窗口：宽厚波形输出 1，细长条形输出 0。"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - import guard for environments without deps
    cv2 = None
    np = None


@dataclass
class DetectorConfig:
    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    ink_threshold: int = 135
    blur_kernel: int = 3
    active_col_ratio_for_one: float = 0.22
    active_col_ratio_for_zero: float = 0.07
    vertical_fill_for_one: float = 0.35
    vertical_fill_for_zero: float = 0.18
    min_hold_frames: int = 3


class WaveformDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._history: Deque[int] = deque(maxlen=config.min_hold_frames)
        self._last_output: Optional[int] = None

    def process_frame(self, frame: "np.ndarray") -> Tuple[int, dict]:
        cfg = self.config
        if cfg.roi is not None:
            x, y, w, h = cfg.roi
            frame = frame[y : y + h, x : x + w]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cfg.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0)

        # 波形一般为深色线条，阈值后深色像素变为 1
        ink_mask = gray < cfg.ink_threshold
        col_ink = ink_mask.sum(axis=0)
        row_ink = ink_mask.sum(axis=1)

        h, w = ink_mask.shape
        active_cols = (col_ink > (0.08 * h)).sum()
        active_col_ratio = float(active_cols) / float(w)
        vertical_fill_ratio = float(ink_mask.mean())
        row_peaks = (row_ink > (0.20 * w)).sum() / float(h)

        is_one = (
            active_col_ratio >= cfg.active_col_ratio_for_one
            and vertical_fill_ratio >= cfg.vertical_fill_for_one
            and row_peaks >= 0.08
        )
        is_zero = (
            active_col_ratio <= cfg.active_col_ratio_for_zero
            and vertical_fill_ratio <= cfg.vertical_fill_for_zero
        )

        # 若处于模糊地带，则沿用上一状态，避免抖动
        if is_one:
            current = 1
        elif is_zero:
            current = 0
        else:
            current = self._last_output if self._last_output is not None else 0

        self._history.append(current)
        if len(self._history) == self._history.maxlen and len(set(self._history)) == 1:
            output = self._history[-1]
        else:
            output = self._last_output if self._last_output is not None else current

        self._last_output = output
        metrics = {
            "active_col_ratio": active_col_ratio,
            "vertical_fill_ratio": vertical_fill_ratio,
            "row_peaks": row_peaks,
            "current": current,
            "output": output,
        }
        return output, metrics


def parse_roi(roi: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi:
        return None
    parts = [int(x.strip()) for x in roi.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be in format x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def open_capture(source: str):
    if source == "screen":
        raise ValueError("screen mode requires --screen flag")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def run_video_mode(args: argparse.Namespace) -> int:
    cap = open_capture(args.source)
    detector = WaveformDetector(
        DetectorConfig(roi=parse_roi(args.roi), min_hold_frames=args.hold_frames)
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        output, metrics = detector.process_frame(frame)

        print(
            f"frame={frame_idx:06d} output={output} "
            f"active_col_ratio={metrics['active_col_ratio']:.3f} "
            f"vertical_fill={metrics['vertical_fill_ratio']:.3f}"
        )
        frame_idx += 1

        if args.preview:
            display = frame.copy()
            if detector.config.roi:
                x, y, w, h = detector.config.roi
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"output={output}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("waveform-monitor", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def run_screen_mode(args: argparse.Namespace) -> int:
    import mss  # type: ignore

    detector = WaveformDetector(
        DetectorConfig(roi=parse_roi(args.roi), min_hold_frames=args.hold_frames)
    )
    monitor = {
        "top": args.screen_top,
        "left": args.screen_left,
        "width": args.screen_width,
        "height": args.screen_height,
    }

    with mss.mss() as sct:
        while True:
            shot = sct.grab(monitor)
            frame = np.array(shot)[:, :, :3]
            output, metrics = detector.process_frame(frame)
            print(
                f"ts={time.time():.3f} output={output} "
                f"active_col_ratio={metrics['active_col_ratio']:.3f} "
                f"vertical_fill={metrics['vertical_fill_ratio']:.3f}"
            )
            if args.preview:
                vis = frame.copy()
                cv2.putText(
                    vis,
                    f"output={output}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("waveform-monitor-screen", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(max(args.interval_ms, 1) / 1000.0)

    cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时监测波形窗口并输出 1/0")
    parser.add_argument(
        "--source",
        default="da3fd50e6979f4b2cc582dfe0695445b.mp4",
        help="视频文件路径（默认使用仓库中的录屏文件）",
    )
    parser.add_argument("--roi", default=None, help="感兴趣区域，格式 x,y,w,h")
    parser.add_argument("--hold-frames", type=int, default=3, help="去抖动帧数")
    parser.add_argument("--preview", action="store_true", help="显示调试画面，按 q 退出")

    parser.add_argument("--screen", action="store_true", help="启用屏幕实时监测模式")
    parser.add_argument("--screen-left", type=int, default=0)
    parser.add_argument("--screen-top", type=int, default=0)
    parser.add_argument("--screen-width", type=int, default=800)
    parser.add_argument("--screen-height", type=int, default=600)
    parser.add_argument("--interval-ms", type=int, default=60, help="screen 模式采样间隔")
    return parser


def main() -> int:
    if cv2 is None or np is None:
        print(
            "Missing dependencies. Please install: pip install opencv-python numpy mss",
            file=sys.stderr,
        )
        return 2

    args = build_parser().parse_args()
    if args.screen:
        return run_screen_mode(args)
    return run_video_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
