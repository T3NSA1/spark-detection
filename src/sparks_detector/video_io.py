from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def find_video_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not path.is_dir():
        return []
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ensure_output_dirs(output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    dirs = {
        "root": root,
        "videos": root / "videos",
        "predictions": root / "predictions",
        "eval_frames": root / "eval_frames",
        "metrics": root / "metrics",
        "debug_masks": root / "debug_masks",
        "screenshots": root / "screenshots",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def resize_frame(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if max_width <= 0 or width <= max_width:
        return frame_bgr.copy()
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(round(height * scale))))
    return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)


def get_video_fps(capture: cv2.VideoCapture, default_fps: float = 30.0) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        return default_fps
    return fps


def get_frame_count(capture: cv2.VideoCapture) -> int | None:
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count if frame_count > 0 else None


def open_video_writer(output_path: str | Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    return writer
