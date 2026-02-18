#!/usr/bin/env python3
"""实时识别波形窗口：宽厚波形输出 1，细长条形输出 0。"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2  # type: ignore
import mss  # type: ignore
import numpy as np  # type: ignore


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


class InteractiveROISelector:
    HANDLE_SIZE = 10
    BUTTON_SIZE = 30
    MIN_W = 80
    MIN_H = 80

    def __init__(self, monitor_w: int, monitor_h: int, initial_roi: Optional[Tuple[int, int, int, int]]) -> None:
        self.monitor_w = monitor_w
        self.monitor_h = monitor_h
        if initial_roi:
            self.roi = list(initial_roi)
        else:
            self.roi = [monitor_w // 4, monitor_h // 4, monitor_w // 2, monitor_h // 2]

        self.drag_mode: Optional[str] = None
        self.drag_start = (0, 0)
        self.roi_start = self.roi.copy()
        self.done = False
        self.confirmed = False

    def _clamp_roi(self) -> None:
        x, y, w, h = self.roi
        w = max(self.MIN_W, min(w, self.monitor_w))
        h = max(self.MIN_H, min(h, self.monitor_h))
        x = max(0, min(x, self.monitor_w - w))
        y = max(0, min(y, self.monitor_h - h))
        self.roi = [x, y, w, h]

    def _button_rects(self) -> Dict[str, Tuple[int, int, int, int]]:
        x, y, w, h = self.roi
        y0 = min(self.monitor_h - self.BUTTON_SIZE - 2, y + h + 8)
        cancel = (x, y0, self.BUTTON_SIZE, self.BUTTON_SIZE)
        confirm = (x + w - self.BUTTON_SIZE, y0, self.BUTTON_SIZE, self.BUTTON_SIZE)
        return {"cancel": cancel, "confirm": confirm}

    @staticmethod
    def _in_rect(px: int, py: int, rect: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = rect
        return x <= px <= x + w and y <= py <= y + h

    def _hit_test(self, px: int, py: int) -> Optional[str]:
        x, y, w, h = self.roi
        hs = self.HANDLE_SIZE
        left = abs(px - x) <= hs
        right = abs(px - (x + w)) <= hs
        top = abs(py - y) <= hs
        bottom = abs(py - (y + h)) <= hs
        inside = x <= px <= x + w and y <= py <= y + h

        if left and top:
            return "resize_tl"
        if right and top:
            return "resize_tr"
        if left and bottom:
            return "resize_bl"
        if right and bottom:
            return "resize_br"
        if left and inside:
            return "resize_l"
        if right and inside:
            return "resize_r"
        if top and inside:
            return "resize_t"
        if bottom and inside:
            return "resize_b"
        if inside:
            return "move"
        return None

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        buttons = self._button_rects()

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._in_rect(x, y, buttons["cancel"]):
                self.confirmed = False
                self.done = True
                return
            if self._in_rect(x, y, buttons["confirm"]):
                self.confirmed = True
                self.done = True
                return

            hit = self._hit_test(x, y)
            if hit is not None:
                self.drag_mode = hit
                self.drag_start = (x, y)
                self.roi_start = self.roi.copy()

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_mode is not None:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            sx, sy, sw, sh = self.roi_start

            if self.drag_mode == "move":
                self.roi = [sx + dx, sy + dy, sw, sh]
            elif self.drag_mode == "resize_l":
                new_x = sx + dx
                new_w = sw - dx
                self.roi = [new_x, sy, new_w, sh]
            elif self.drag_mode == "resize_r":
                self.roi = [sx, sy, sw + dx, sh]
            elif self.drag_mode == "resize_t":
                new_y = sy + dy
                new_h = sh - dy
                self.roi = [sx, new_y, sw, new_h]
            elif self.drag_mode == "resize_b":
                self.roi = [sx, sy, sw, sh + dy]
            elif self.drag_mode == "resize_tl":
                self.roi = [sx + dx, sy + dy, sw - dx, sh - dy]
            elif self.drag_mode == "resize_tr":
                self.roi = [sx, sy + dy, sw + dx, sh - dy]
            elif self.drag_mode == "resize_bl":
                self.roi = [sx + dx, sy, sw - dx, sh + dy]
            elif self.drag_mode == "resize_br":
                self.roi = [sx, sy, sw + dx, sh + dy]

            self._clamp_roi()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_mode = None

    def draw(self, frame: "np.ndarray") -> "np.ndarray":
        blurred = cv2.GaussianBlur(frame, (51, 51), 0)
        display = blurred.copy()

        x, y, w, h = self.roi
        display[y : y + h, x : x + w] = frame[y : y + h, x : x + w]

        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            display,
            "Drag to move/resize ROI, Enter=confirm, Esc=cancel",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for rect_name, rect in self._button_rects().items():
            bx, by, bw, bh = rect
            color = (0, 180, 0) if rect_name == "confirm" else (0, 0, 200)
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, -1)
            symbol = "✓" if rect_name == "confirm" else "✕"
            cv2.putText(
                display,
                symbol,
                (bx + 6, by + bh - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return display

    def get_roi(self) -> Tuple[int, int, int, int]:
        return tuple(self.roi)  # type: ignore[return-value]


def select_roi_interactively(
    sct: mss.mss, monitor: Dict[str, int], initial_roi: Optional[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, int, int, int]]:
    selector = InteractiveROISelector(monitor["width"], monitor["height"], initial_roi)
    win_name = "Select ROI"

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win_name, selector.on_mouse)

    while not selector.done:
        shot = sct.grab(monitor)
        frame = np.array(shot)[:, :, :3]
        display = selector.draw(frame)
        cv2.imshow(win_name, display)

        key = cv2.waitKey(16) & 0xFF
        if key in (13, 10):
            selector.confirmed = True
            selector.done = True
        elif key == 27:
            selector.confirmed = False
            selector.done = True

    cv2.destroyWindow(win_name)
    if selector.confirmed:
        return selector.get_roi()
    return None


def parse_roi(roi: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi:
        return None
    parts = [int(x.strip()) for x in roi.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be in format x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def open_capture(source: str):
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
    initial_roi = parse_roi(args.roi)

    with mss.mss() as sct:
        if args.interactive_roi:
            primary = sct.monitors[1]
            monitor = {
                "top": primary["top"],
                "left": primary["left"],
                "width": primary["width"],
                "height": primary["height"],
            }
            selected = select_roi_interactively(sct, monitor, initial_roi)
            if selected is None:
                print("ROI selection canceled, exit.")
                return 0
            roi = selected
            print(f"ROI selected: {roi[0]},{roi[1]},{roi[2]},{roi[3]}")
        else:
            monitor = {
                "top": args.screen_top,
                "left": args.screen_left,
                "width": args.screen_width,
                "height": args.screen_height,
            }
            roi = initial_roi

        detector = WaveformDetector(DetectorConfig(roi=roi, min_hold_frames=args.hold_frames))

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
                if detector.config.roi:
                    x, y, w, h = detector.config.roi
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
    parser.add_argument("--interactive-roi", action="store_true", help="全屏交互式选取 ROI")
    parser.add_argument("--screen-left", type=int, default=0)
    parser.add_argument("--screen-top", type=int, default=0)
    parser.add_argument("--screen-width", type=int, default=800)
    parser.add_argument("--screen-height", type=int, default=600)
    parser.add_argument("--interval-ms", type=int, default=60, help="screen 模式采样间隔")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.screen:
        return run_screen_mode(args)
    return run_video_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
