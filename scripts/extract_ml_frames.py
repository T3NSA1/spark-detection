from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparks_detector.video_io import SUPPORTED_EXTENSIONS, find_video_files, get_frame_count, get_video_fps


MANIFEST_COLUMNS = [
    "video_name",
    "frame_idx",
    "time_sec",
    "image_path",
    "suggested_split",
    "label_notes",
]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    images_dir = output_dir / "images_to_label"
    images_dir.mkdir(parents=True, exist_ok=True)

    videos = find_video_files(args.input)
    if not videos:
        print(
            "No supported videos found in "
            f"{args.input}. Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        write_manifest(output_dir, [])
        return 0

    rows: list[dict[str, object]] = []
    for video_path in videos:
        rows.extend(
            extract_frames(
                video_path=video_path,
                images_dir=images_dir,
                every_n_frames=args.every_n_frames,
                max_frames_per_video=args.max_frames_per_video,
                val_every=args.val_every,
            )
        )

    manifest_path = write_manifest(output_dir, rows)
    print(f"Saved ML frame manifest: {manifest_path}")
    print(f"Images for annotation: {images_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video frames for YOLO spark/fire annotation.")
    parser.add_argument("--input", default="data/raw/videos", help="Video file or directory of videos.")
    parser.add_argument("--output", default="data/ml", help="YOLO dataset root.")
    parser.add_argument("--every-n-frames", type=int, default=15, help="Sampling interval in frames.")
    parser.add_argument("--max-frames-per-video", type=int, default=250, help="Maximum sampled frames per video.")
    parser.add_argument(
        "--val-every",
        type=int,
        default=5,
        help="Every Nth extracted frame is suggested for validation.",
    )
    args = parser.parse_args()
    if args.every_n_frames <= 0:
        parser.error("--every-n-frames must be positive")
    if args.max_frames_per_video <= 0:
        parser.error("--max-frames-per-video must be positive")
    if args.val_every <= 1:
        parser.error("--val-every must be greater than 1")
    return args


def extract_frames(
    video_path: Path,
    images_dir: Path,
    every_n_frames: int,
    max_frames_per_video: int,
    val_every: int,
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
                image_path = images_dir / image_name
                cv2.imwrite(str(image_path), frame)
                suggested_split = "val" if saved_count % val_every == 0 else "train"
                rows.append(
                    {
                        "video_name": video_path.name,
                        "frame_idx": frame_idx,
                        "time_sec": round(frame_idx / fps, 6),
                        "image_path": str(image_path),
                        "suggested_split": suggested_split,
                        "label_notes": "",
                    }
                )
                saved_count += 1

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        capture.release()

    print(f"Extracted {saved_count} ML frames from {video_path.name}")
    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    manifest_path = output_dir / "ml_frames_manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    return manifest_path


if __name__ == "__main__":
    raise SystemExit(main())
