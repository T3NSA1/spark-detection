from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparks_detector.evaluation import GROUND_TRUTH_COLUMNS
from sparks_detector.video_io import SUPPORTED_EXTENSIONS, find_video_files, get_frame_count, get_video_fps


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_video_files(args.input)
    rows: list[dict[str, object]] = []

    if not videos:
        print(
            "No supported videos found in "
            f"{args.input}. Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        write_template(output_dir, rows)
        return 0

    for video_path in videos:
        rows.extend(extract_frames(video_path, output_dir, args.every_n_frames, args.max_frames_per_video))

    template_path = write_template(output_dir, rows)
    print(f"Saved manual label template: {template_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sampled frames for manual spark labels.")
    parser.add_argument("--input", default="data/raw/videos", help="Video file or directory of videos.")
    parser.add_argument("--output", default="outputs/eval_frames", help="Output directory for JPEGs and CSV.")
    parser.add_argument("--every-n-frames", type=int, default=10, help="Sampling interval in frames.")
    parser.add_argument("--max-frames-per-video", type=int, default=150, help="Maximum sampled frames per video.")
    args = parser.parse_args()
    if args.every_n_frames <= 0:
        parser.error("--every-n-frames must be positive")
    if args.max_frames_per_video <= 0:
        parser.error("--max-frames-per-video must be positive")
    return args


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int,
    max_frames_per_video: int,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open video, skipping: {video_path}")
        return []

    fps = get_video_fps(capture)
    frame_count = get_frame_count(capture)
    rows: list[dict[str, object]] = []
    saved_count = 0
    frame_idx = 0
    pbar = tqdm(total=frame_count, desc=video_path.name, unit="frame")

    try:
        while saved_count < max_frames_per_video:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_idx % every_n_frames == 0:
                image_name = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
                image_path = output_dir / image_name
                cv2.imwrite(str(image_path), frame)
                rows.append(
                    {
                        "video_name": video_path.name,
                        "frame_idx": frame_idx,
                        "time_sec": round(frame_idx / fps, 6),
                        "image_path": str(image_path),
                        "has_sparks_gt": "",
                        "source_gt": "",
                        "comment": "",
                    }
                )
                saved_count += 1

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        capture.release()

    print(f"Extracted {saved_count} frames from {video_path.name}")
    return rows


def write_template(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    template_path = output_dir / "manual_labels_template.csv"
    pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS).to_csv(template_path, index=False)
    return template_path


if __name__ == "__main__":
    raise SystemExit(main())
