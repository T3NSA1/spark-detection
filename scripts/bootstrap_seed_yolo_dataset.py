from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparks_detector.video_io import get_video_fps, resize_frame


CLASS_TO_ID = {
    "welding_arc": 0,
    "spark_plume": 1,
    "fire": 2,
}


def main() -> int:
    args = parse_args()
    output_root = Path(args.output)
    ensure_dataset_dirs(output_root)
    if args.clear_existing_seed:
        clear_existing_seed_files(output_root)

    rows: list[dict[str, object]] = []
    rows.extend(label_easy_from_predictions(args, output_root))
    rows.extend(label_medium_static_arcs(args, output_root))
    rows.extend(label_fire_and_empty_negatives(args, output_root))

    manifest_path = output_root / "seed_labels_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    write_summary(rows)
    print(f"Saved seed manifest: {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a small YOLO seed dataset from the demo videos.")
    parser.add_argument("--videos", default="data/raw/videos", help="Directory with source videos.")
    parser.add_argument("--predictions", default="outputs/predictions", help="OpenCV prediction CSV directory.")
    parser.add_argument("--output", default="data/ml", help="YOLO dataset root.")
    parser.add_argument("--max-width", type=int, default=960, help="Saved image max width.")
    parser.add_argument("--easy-max", type=int, default=120, help="Max easy positive frames.")
    parser.add_argument("--medium-max", type=int, default=180, help="Max medium welding-arc frames.")
    parser.add_argument("--fire-max", type=int, default=160, help="Max fire frames from nonpositive video.")
    parser.add_argument("--empty-negative-max", type=int, default=60, help="Max empty negative frames.")
    parser.add_argument("--sample-step", type=int, default=8, help="Frame sampling step for heuristic videos.")
    parser.add_argument("--fire-sample-step", type=int, default=2, help="Frame sampling step for fire video.")
    parser.add_argument("--val-every", type=int, default=5, help="Every Nth sample goes to val.")
    parser.add_argument(
        "--clear-existing-seed",
        action="store_true",
        help="Delete existing seed_*.jpg/txt files in the YOLO dataset before writing new seed labels.",
    )
    args = parser.parse_args()
    if args.max_width <= 0:
        parser.error("--max-width must be positive")
    if args.sample_step <= 0:
        parser.error("--sample-step must be positive")
    if args.fire_sample_step <= 0:
        parser.error("--fire-sample-step must be positive")
    if args.val_every <= 1:
        parser.error("--val-every must be greater than 1")
    return args


def ensure_dataset_dirs(output_root: Path) -> None:
    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "contact_sheets",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def clear_existing_seed_files(output_root: Path) -> None:
    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        directory = output_root / relative
        if not directory.exists():
            continue
        for path in directory.glob("seed_*"):
            if path.is_file():
                path.unlink()


def label_easy_from_predictions(args: argparse.Namespace, output_root: Path) -> list[dict[str, object]]:
    video_path = Path(args.videos) / "test-sparks-easy.mp4"
    prediction_path = Path(args.predictions) / "test-sparks-easy_predictions.csv"
    if not video_path.exists() or not prediction_path.exists():
        print("Skipping easy labels: video or prediction CSV missing.")
        return []

    predictions = pd.read_csv(prediction_path)
    positives = []
    for _, row in predictions.iterrows():
        if str(row.get("raw_has_sparks", "")).lower() != "true":
            continue
        boxes = json.loads(row.get("boxes_json", "[]"))
        if not boxes:
            continue
        positives.append((int(row["frame_idx"]), boxes))

    selected = evenly_sample(positives, args.easy_max)
    capture = cv2.VideoCapture(str(video_path))
    fps = get_video_fps(capture)
    rows: list[dict[str, object]] = []
    for sample_idx, (frame_idx, boxes) in enumerate(selected):
        frame = read_frame(capture, frame_idx)
        if frame is None:
            continue
        image = resize_frame(frame, args.max_width)
        image_h, image_w = image.shape[:2]
        labels = []
        for box in boxes[:1]:
            xywh = clip_xywh(
                int(box["x"]),
                int(box["y"]),
                int(box["w"]),
                int(box["h"]),
                image_w,
                image_h,
                pad=6,
            )
            if xywh is not None:
                labels.append(("spark_plume", xywh))

        if not labels:
            continue
        split = split_for_sample(len(rows), args.val_every)
        image_path, label_path = write_sample(
            output_root=output_root,
            split=split,
            source_name="easy",
            frame_idx=frame_idx,
            image=image,
            labels=labels,
        )
        rows.append(
            manifest_row(
                video_path=video_path,
                frame_idx=frame_idx,
                fps=fps,
                split=split,
                image_path=image_path,
                label_path=label_path,
                labels=labels,
                method="opencv_main_plume",
            )
        )
    capture.release()
    return rows


def label_medium_static_arcs(args: argparse.Namespace, output_root: Path) -> list[dict[str, object]]:
    video_path = Path(args.videos) / "test-sparks-medium.mp4"
    if not video_path.exists():
        print("Skipping medium labels: video missing.")
        return []

    capture = cv2.VideoCapture(str(video_path))
    fps = get_video_fps(capture)
    rows: list[dict[str, object]] = []
    frame_idx = 0
    while len(rows) < args.medium_max:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % args.sample_step != 0:
            frame_idx += 1
            continue

        image = resize_frame(frame, args.max_width)
        box = find_static_welding_arc_box(image)
        if box is not None:
            split = split_for_sample(len(rows), args.val_every)
            image_path, label_path = write_sample(
                output_root=output_root,
                split=split,
                source_name="medium",
                frame_idx=frame_idx,
                image=image,
                labels=[("welding_arc", box)],
            )
            rows.append(
                manifest_row(
                    video_path=video_path,
                    frame_idx=frame_idx,
                    fps=fps,
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    labels=[("welding_arc", box)],
                    method="static_welding_core",
                )
            )
        frame_idx += 1

    capture.release()
    return rows


def label_fire_and_empty_negatives(args: argparse.Namespace, output_root: Path) -> list[dict[str, object]]:
    video_path = Path(args.videos) / "test-sparks-nopositives.mp4"
    if not video_path.exists():
        print("Skipping nonpositive labels: video missing.")
        return []

    capture = cv2.VideoCapture(str(video_path))
    fps = get_video_fps(capture)
    rows: list[dict[str, object]] = []
    fire_count = 0
    empty_count = 0
    frame_idx = 0
    while fire_count < args.fire_max or empty_count < args.empty_negative_max:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % args.fire_sample_step != 0:
            frame_idx += 1
            continue

        image = resize_frame(frame, args.max_width)
        fire_box, warm_fraction = find_fire_box(image)
        labels: list[tuple[str, tuple[int, int, int, int]]] = []
        method = ""
        source_name = ""
        if fire_box is not None and fire_count < args.fire_max:
            labels = [("fire", fire_box)]
            method = "large_warm_fire_mask"
            source_name = "nopositive_fire"
            fire_count += 1
        elif fire_box is None and warm_fraction < 0.015 and empty_count < args.empty_negative_max:
            labels = []
            method = "empty_negative"
            source_name = "nopositive_empty"
            empty_count += 1
        else:
            frame_idx += 1
            continue

        split = split_for_sample(len(rows), args.val_every)
        image_path, label_path = write_sample(
            output_root=output_root,
            split=split,
            source_name=source_name,
            frame_idx=frame_idx,
            image=image,
            labels=labels,
        )
        rows.append(
            manifest_row(
                video_path=video_path,
                frame_idx=frame_idx,
                fps=fps,
                split=split,
                image_path=image_path,
                label_path=label_path,
                labels=labels,
                method=method,
            )
        )
        frame_idx += 1

    capture.release()
    return rows


def find_static_welding_arc_box(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    image_h, image_w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    _, g, r = cv2.split(image_bgr)
    yy, xx = np.indices(v.shape)
    operation_roi = (
        (xx >= int(image_w * 0.38))
        & (xx <= int(image_w * 0.50))
        & (yy >= int(image_h * 0.47))
        & (yy <= int(image_h * 0.64))
    )
    core_mask = (
        operation_roi
        & (((h <= 35) | (h >= 170)))
        & (s >= 40)
        & (v >= 70)
        & (r.astype(np.int16) >= g.astype(np.int16) - 5)
    ).astype(np.uint8)
    core_mask = cv2.morphologyEx(core_mask * 255, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    core_mask = cv2.dilate(core_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(core_mask, connectivity=8)
    candidates = []
    for label_idx in range(1, n_labels):
        x, y, w, h_box, area = [int(value) for value in stats[label_idx]]
        if area < 8 or area > 900:
            continue
        if w > image_w * 0.09 or h_box > image_h * 0.08:
            continue
        cx, cy = centroids[label_idx]
        score = area + 1.5 * float(np.max(v[y : y + h_box, x : x + w])) + 0.12 * float(cy)
        candidates.append((score, x, y, w, h_box))

    if not candidates:
        return None

    _, anchor_x, anchor_y, anchor_w, anchor_h = max(candidates, key=lambda item: item[0])
    anchor_cx = anchor_x + anchor_w / 2.0
    anchor_cy = anchor_y + anchor_h / 2.0
    close_boxes = []
    for _, x, y, w, h_box in candidates:
        cx = x + w / 2.0
        cy = y + h_box / 2.0
        if np.hypot(cx - anchor_cx, cy - anchor_cy) <= max(36.0, image_w * 0.04):
            close_boxes.append((x, y, w, h_box))

    x1 = min(x for x, _, _, _ in close_boxes)
    y1 = min(y for _, y, _, _ in close_boxes)
    x2 = max(x + w for x, _, w, _ in close_boxes)
    y2 = max(y + h_box for _, y, _, h_box in close_boxes)
    return clip_xyxy(x1, y1, x2, y2, image_w, image_h, pad=14)


def find_fire_box(image_bgr: np.ndarray) -> tuple[tuple[int, int, int, int] | None, float]:
    image_h, image_w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(image_bgr)
    red_dominant = r.astype(np.int16) > g.astype(np.int16) + 12
    bright_yellow_orange = (v >= 215) & (s >= 80) & (r.astype(np.int16) >= g.astype(np.int16) - 5)
    flame = (
        (
            (((h <= 38) | (h >= 170)) & (s >= 85) & (v >= 115))
            | ((h >= 5) & (h <= 35) & (s >= 55) & (v >= 160))
        )
        & (red_dominant | bright_yellow_orange)
    ).astype(np.uint8)
    flame = cv2.morphologyEx(flame * 255, cv2.MORPH_CLOSE, np.ones((13, 13), dtype=np.uint8))
    flame_pixels = int(np.count_nonzero(flame))
    flame_fraction = flame_pixels / max(1.0, float(image_w * image_h))
    if flame_fraction < 0.015:
        return None, flame_fraction

    mean_saturation = float(np.mean(s[flame > 0])) if flame_pixels else 0.0
    if mean_saturation < 90.0:
        return None, flame_fraction

    contours, _ = cv2.findContours(flame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, flame_fraction
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < image_w * image_h * 0.008:
        return None, flame_fraction

    x, y, w, h_box = cv2.boundingRect(largest)
    return clip_xywh(x, y, w, h_box, image_w, image_h, pad=8), flame_fraction


def write_sample(
    output_root: Path,
    split: str,
    source_name: str,
    frame_idx: int,
    image: np.ndarray,
    labels: list[tuple[str, tuple[int, int, int, int]]],
) -> tuple[Path, Path]:
    image_name = f"seed_{source_name}_frame_{frame_idx:06d}.jpg"
    label_name = f"seed_{source_name}_frame_{frame_idx:06d}.txt"
    image_path = output_root / "images" / split / image_name
    label_path = output_root / "labels" / split / label_name
    cv2.imwrite(str(image_path), image)

    lines = []
    image_h, image_w = image.shape[:2]
    for class_name, box in labels:
        class_id = CLASS_TO_ID[class_name]
        lines.append(yolo_line(class_id, box, image_w, image_h))
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return image_path, label_path


def yolo_line(class_id: int, box: tuple[int, int, int, int], image_w: int, image_h: int) -> str:
    x, y, w, h = box
    cx = (x + w / 2.0) / image_w
    cy = (y + h / 2.0) / image_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w / image_w:.6f} {h / image_h:.6f}"


def manifest_row(
    video_path: Path,
    frame_idx: int,
    fps: float,
    split: str,
    image_path: Path,
    label_path: Path,
    labels: list[tuple[str, tuple[int, int, int, int]]],
    method: str,
) -> dict[str, object]:
    return {
        "video_name": video_path.name,
        "frame_idx": frame_idx,
        "time_sec": round(frame_idx / fps, 6),
        "split": split,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "labels_json": json.dumps(
            [{"class": class_name, "box": list(box)} for class_name, box in labels]
        ),
        "method": method,
    }


def split_for_sample(sample_idx: int, val_every: int) -> str:
    return "val" if sample_idx % val_every == 0 else "train"


def read_frame(capture: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    return frame if ok else None


def evenly_sample(items: list[tuple[int, list[dict[str, object]]]], max_count: int):
    if len(items) <= max_count:
        return items
    indices = np.linspace(0, len(items) - 1, max_count).round().astype(int)
    return [items[int(index)] for index in indices]


def clip_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_w: int,
    image_h: int,
    pad: int = 0,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image_w, x2 + pad)
    y2 = min(image_h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def clip_xywh(
    x: int,
    y: int,
    w: int,
    h: int,
    image_w: int,
    image_h: int,
    pad: int = 0,
) -> tuple[int, int, int, int] | None:
    return clip_xyxy(x, y, x + w, y + h, image_w, image_h, pad=pad)


def write_summary(rows: list[dict[str, object]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        labels = json.loads(str(row["labels_json"]))
        class_name = labels[0]["class"] if labels else "empty"
        key = (str(row["split"]), class_name)
        counts[key] = counts.get(key, 0) + 1

    print("Seed dataset summary:")
    for (split, class_name), count in sorted(counts.items()):
        print(f"  {split:5s} {class_name:12s} {count}")


if __name__ == "__main__":
    raise SystemExit(main())
