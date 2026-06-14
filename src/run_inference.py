from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sparks_detector.detector import SparksDetector, SparksDetectorConfig
from sparks_detector.evaluation import PREDICTION_COLUMNS
from sparks_detector.ml_detector import YoloDetectorConfig, YoloFrameResult, YoloSparksDetector
from sparks_detector.video_io import (
    SUPPORTED_EXTENSIONS,
    ensure_output_dirs,
    find_video_files,
    get_frame_count,
    get_video_fps,
    open_video_writer,
)
from sparks_detector.visualization import draw_prediction


def main() -> int:
    args = parse_args()
    output_dirs = ensure_output_dirs(args.output)
    videos = find_video_files(args.input)

    if not videos:
        print(
            "No supported videos found in "
            f"{args.input}. Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        return 0

    try:
        for video_path in videos:
            process_video(video_path, output_dirs, args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect sparks in videos with OpenCV, YOLO, or hybrid logic.")
    parser.add_argument("--input", default="data/raw/videos", help="Video file or directory of videos.")
    parser.add_argument("--output", default="outputs", help="Output root directory.")
    parser.add_argument(
        "--backend",
        choices=("opencv", "yolo", "hybrid"),
        default="opencv",
        help="Detection backend. Hybrid uses YOLO first, then OpenCV fallback unless fire is detected.",
    )
    parser.add_argument(
        "--model",
        default="runs/sparks_yolo/train/weights/best.pt",
        help="YOLO model path for --backend yolo or hybrid.",
    )
    parser.add_argument("--max-width", type=int, default=960, help="Resize frames to this max width.")
    parser.add_argument(
        "--temporal-smoothing",
        type=parse_bool,
        default=True,
        help="Enable temporal smoothing: true or false.",
    )
    parser.add_argument(
        "--save-debug-masks",
        type=parse_bool,
        default=False,
        help="Save per-frame binary masks: true or false.",
    )
    parser.add_argument(
        "--main-plume-only",
        type=parse_bool,
        default=True,
        help="Keep only the dominant spark/plasma plume and ignore side outliers: true or false.",
    )
    parser.add_argument(
        "--core-anchored-plume-filter",
        type=parse_bool,
        default=True,
        help="For welding-like hot cores, crop the plume around the core to suppress flash reflections.",
    )
    parser.add_argument(
        "--main-plume-anchor-radius",
        type=float,
        default=0.10,
        help="Core-anchored plume radius as a fraction of frame max dimension.",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.50, help="YOLO NMS IoU threshold.")
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, for example 0, cpu, or cuda:0. Leave empty for Ultralytics default.",
    )
    parser.add_argument(
        "--fire-suppression-conf",
        type=float,
        default=0.35,
        help="Minimum fire confidence to suppress spark detections in YOLO/hybrid mode.",
    )
    parser.add_argument(
        "--fire-suppression-area",
        type=float,
        default=0.08,
        help="Fire box area fraction that can suppress weaker event boxes.",
    )
    parser.add_argument(
        "--hybrid-core-gating",
        type=parse_bool,
        default=True,
        help="In hybrid mode, require OpenCV fallback plumes to be near a compact hot core.",
    )
    parser.add_argument(
        "--hybrid-core-radius",
        type=float,
        default=0.16,
        help="Hybrid fallback core-to-plume radius as a fraction of frame max dimension.",
    )
    parser.add_argument(
        "--hybrid-suppress-large-warm-scenes",
        type=parse_bool,
        default=True,
        help="In hybrid mode, reject OpenCV fallback on frames dominated by fire-like warm color.",
    )
    parser.add_argument("--temporal-window", type=int, default=5, help="Temporal smoothing window.")
    parser.add_argument(
        "--temporal-min-positive",
        type=int,
        default=2,
        help="Minimum positives in temporal window.",
    )
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def process_video(video_path: Path, output_dirs: dict[str, Path], args: argparse.Namespace) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open video, skipping: {video_path}")
        return

    fps = get_video_fps(capture)
    frame_count = get_frame_count(capture)
    opencv_detector = (
        build_opencv_detector(
            args,
            enable_temporal_smoothing=args.temporal_smoothing and args.backend == "opencv",
        )
        if args.backend in {"opencv", "hybrid"}
        else None
    )
    yolo_detector = (
        build_yolo_detector(
            args,
            enable_temporal_smoothing=args.temporal_smoothing and args.backend == "yolo",
        )
        if args.backend in {"yolo", "hybrid"}
        else None
    )
    hybrid_history: deque[bool] | None = (
        deque(maxlen=max(1, int(args.temporal_window))) if args.backend == "hybrid" else None
    )

    output_suffix = "sparks_detected" if args.backend == "opencv" else f"{args.backend}_sparks_detected"
    prediction_suffix = "predictions" if args.backend == "opencv" else f"{args.backend}_predictions"
    annotated_path = output_dirs["videos"] / f"{video_path.stem}_{output_suffix}.mp4"
    prediction_path = output_dirs["predictions"] / f"{video_path.stem}_{prediction_suffix}.csv"
    debug_dir = output_dirs["debug_masks"] / video_path.stem
    if args.save_debug_masks:
        debug_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    writer = None
    pbar = tqdm(total=frame_count, desc=video_path.name, unit="frame")
    frame_idx = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            time_sec = frame_idx / fps
            resized_frame, prediction = process_frame_with_backend(
                frame=frame,
                frame_idx=frame_idx,
                time_sec=time_sec,
                backend=args.backend,
                opencv_detector=opencv_detector,
                yolo_detector=yolo_detector,
                hybrid_history=hybrid_history,
                args=args,
            )
            annotated = draw_prediction(resized_frame, prediction, prediction.mask)

            if writer is None:
                height, width = annotated.shape[:2]
                writer = open_video_writer(annotated_path, fps, (width, height))

            writer.write(annotated)

            if args.save_debug_masks and prediction.mask is not None:
                mask_path = debug_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(mask_path), prediction.mask)

            rows.append(
                {
                    "video_name": video_path.name,
                    "frame_idx": prediction.frame_idx,
                    "time_sec": round(prediction.time_sec, 6),
                    "has_sparks": prediction.has_sparks,
                    "raw_has_sparks": prediction.raw_has_sparks,
                    "pred_label": prediction.pred_label,
                    "n_components": prediction.n_components,
                    "mask_pixels": prediction.mask_pixels,
                    "boxes_json": json.dumps([box.to_dict() for box in prediction.boxes]),
                }
            )

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        capture.release()
        if writer is not None:
            writer.release()

    if not rows:
        print(f"No readable frames found in {video_path}")
        return

    pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(prediction_path, index=False)
    print(f"Saved annotated video: {annotated_path}")
    print(f"Saved predictions: {prediction_path}")


def build_opencv_detector(
    args: argparse.Namespace,
    enable_temporal_smoothing: bool | None = None,
) -> SparksDetector:
    return SparksDetector(
        SparksDetectorConfig(
            max_width=args.max_width,
            enable_temporal_smoothing=(
                args.temporal_smoothing
                if enable_temporal_smoothing is None
                else enable_temporal_smoothing
            ),
            temporal_window=args.temporal_window,
            temporal_min_positive=args.temporal_min_positive,
            keep_only_main_plume=args.main_plume_only,
            enable_core_anchored_plume_filter=args.core_anchored_plume_filter,
            main_plume_anchor_radius_fraction=args.main_plume_anchor_radius,
        )
    )


def build_yolo_detector(
    args: argparse.Namespace,
    enable_temporal_smoothing: bool | None = None,
) -> YoloSparksDetector:
    model_path = Path(args.model)
    model_name = model_path.name.lower()
    is_ultralytics_builtin = model_name.startswith("yolo") and model_name.endswith(".pt")
    if not model_path.exists() and not is_ultralytics_builtin:
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    return YoloSparksDetector(
        YoloDetectorConfig(
            model_path=args.model,
            max_width=args.max_width,
            image_size=args.yolo_imgsz,
            conf_threshold=args.yolo_conf,
            iou_threshold=args.yolo_iou,
            device=args.device,
            fire_suppression_confidence=args.fire_suppression_conf,
            fire_suppression_area_fraction=args.fire_suppression_area,
            temporal_window=args.temporal_window,
            temporal_min_positive=args.temporal_min_positive,
            enable_temporal_smoothing=(
                args.temporal_smoothing
                if enable_temporal_smoothing is None
                else enable_temporal_smoothing
            ),
        )
    )


def process_frame_with_backend(
    frame,
    frame_idx: int,
    time_sec: float,
    backend: str,
    opencv_detector: SparksDetector | None,
    yolo_detector: YoloSparksDetector | None,
    hybrid_history: deque[bool] | None,
    args: argparse.Namespace,
):
    if backend == "opencv":
        if opencv_detector is None:
            raise RuntimeError("OpenCV detector is not initialized.")
        return opencv_detector.process_frame(frame, frame_idx, time_sec)

    if backend == "yolo":
        if yolo_detector is None:
            raise RuntimeError("YOLO detector is not initialized.")
        return yolo_detector.process_frame(frame, frame_idx, time_sec)

    if backend != "hybrid":
        raise ValueError(f"Unsupported backend: {backend}")
    if opencv_detector is None or yolo_detector is None:
        raise RuntimeError("Hybrid backend requires both OpenCV and YOLO detectors.")

    resized_frame, opencv_prediction = opencv_detector.process_frame(frame, frame_idx, time_sec)
    yolo_result: YoloFrameResult = yolo_detector.process_resized_frame(
        resized_frame,
        frame_idx,
        time_sec,
    )
    yolo_prediction = yolo_result.prediction
    if yolo_result.suppress_boxes:
        return resized_frame, _apply_hybrid_temporal(yolo_prediction, hybrid_history, args)
    if yolo_result.event_boxes:
        return resized_frame, _apply_hybrid_temporal(yolo_prediction, hybrid_history, args)

    fallback_prediction = _filter_hybrid_opencv_fallback(
        resized_frame,
        opencv_prediction,
        args,
    )
    return resized_frame, _apply_hybrid_temporal(fallback_prediction, hybrid_history, args)


def _filter_hybrid_opencv_fallback(
    frame_bgr: np.ndarray,
    prediction: FramePrediction,
    args: argparse.Namespace,
) -> FramePrediction:
    if not prediction.raw_has_sparks or not prediction.boxes or prediction.mask is None:
        return _empty_fallback_prediction(prediction)

    if args.hybrid_suppress_large_warm_scenes and _has_large_warm_scene(frame_bgr):
        return _empty_fallback_prediction(prediction)

    if not args.hybrid_core_gating:
        return prediction

    anchors = _find_compact_hot_anchors(frame_bgr)
    if not anchors:
        return _empty_fallback_prediction(prediction)

    frame_h, frame_w = frame_bgr.shape[:2]
    radius = max(48.0, max(frame_w, frame_h) * float(args.hybrid_core_radius))
    accepted_boxes = []
    for box in prediction.boxes:
        if not _is_core_supported_plume_box(box, anchors, radius):
            continue
        accepted_boxes.append(box)

    if not accepted_boxes:
        return _empty_fallback_prediction(prediction)

    accepted_mask = _mask_for_boxes(prediction.mask, accepted_boxes)
    mask_pixels = int(np.count_nonzero(accepted_mask))
    if mask_pixels < 20:
        return _empty_fallback_prediction(prediction)

    prediction.boxes = accepted_boxes
    prediction.mask = accepted_mask
    prediction.mask_pixels = mask_pixels
    prediction.n_components = max(1, sum(max(1, box.n_children) for box in accepted_boxes))
    prediction.raw_has_sparks = True
    prediction.has_sparks = True
    prediction.pred_label = _event_label_from_boxes(accepted_boxes, prediction.pred_label)
    return prediction


def _empty_fallback_prediction(prediction: FramePrediction) -> FramePrediction:
    prediction.has_sparks = False
    prediction.raw_has_sparks = False
    prediction.pred_label = "no sparks"
    prediction.n_components = 0
    prediction.mask_pixels = 0
    prediction.boxes = []
    if prediction.mask is not None:
        prediction.mask = np.zeros_like(prediction.mask, dtype=np.uint8)
    return prediction


def _apply_hybrid_temporal(
    prediction: FramePrediction,
    history: deque[bool] | None,
    args: argparse.Namespace,
) -> FramePrediction:
    if history is None:
        return prediction

    history.append(bool(prediction.raw_has_sparks))
    if args.temporal_smoothing:
        prediction.has_sparks = sum(history) >= max(1, int(args.temporal_min_positive))
    else:
        prediction.has_sparks = bool(prediction.raw_has_sparks)

    if prediction.has_sparks:
        prediction.pred_label = _event_label_from_boxes(prediction.boxes, prediction.pred_label)
    elif prediction.raw_has_sparks:
        prediction.pred_label = "no sparks"
    elif suppress_label := _suppress_label_from_boxes(prediction.boxes):
        prediction.pred_label = f"no sparks ({suppress_label})"
    else:
        prediction.pred_label = "no sparks"
    return prediction


def _event_label_from_boxes(boxes, fallback_label: str) -> str:
    if boxes:
        best_box = max(boxes, key=lambda box: box.confidence or box.area)
        if best_box.label and "no sparks" not in best_box.label:
            return best_box.label
    if fallback_label and "no sparks" not in fallback_label:
        return fallback_label
    return "sparks"


def _suppress_label_from_boxes(boxes) -> str | None:
    for box in sorted(boxes, key=lambda item: item.confidence or item.area, reverse=True):
        label = (box.label or "").strip()
        normalized = label.lower().replace(" ", "_").replace("-", "_")
        if normalized in {"fire", "grinder", "grinder_spark"}:
            return label
    return None


def _find_compact_hot_anchors(frame_bgr: np.ndarray) -> list[tuple[float, float]]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white_core = ((v >= 245) & (s <= 125)).astype(np.uint8) * 255
    blue_core = (((h >= 85) & (h <= 145) & (s >= 25) & (v >= 170)).astype(np.uint8) * 255)
    core_mask = cv2.bitwise_or(white_core, blue_core)
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    frame_h, frame_w = core_mask.shape[:2]
    frame_area = float(frame_h * frame_w)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(core_mask, connectivity=8)

    anchors: list[tuple[float, float]] = []
    border_margin = 4
    for label_idx in range(1, n_labels):
        x, y, w, h, area = stats[label_idx]
        bbox_area = int(w * h)
        if area < 4 or area > 1600:
            continue
        if bbox_area > frame_area * 0.018:
            continue
        if x <= border_margin or y <= border_margin:
            continue
        if x + w >= frame_w - border_margin or y + h >= frame_h - border_margin:
            continue
        anchors.append((float(centroids[label_idx][0]), float(centroids[label_idx][1])))
    return anchors


def _is_core_supported_plume_box(
    box,
    anchors: list[tuple[float, float]],
    radius: float,
) -> bool:
    bbox_area = max(1.0, float(box.w * box.h))
    density = float(box.area) / bbox_area
    aspect = max(box.w / max(1.0, float(box.h)), box.h / max(1.0, float(box.w)))
    has_stream_shape = (
        box.warm_pixels >= 20
        and (aspect >= 1.45 or box.n_children >= 3 or box.n_streaks >= 1)
        and density <= 0.38
    )
    has_hot_core = box.strong_pixels >= 6
    if not has_stream_shape and not has_hot_core:
        return False

    return any(_distance_point_to_box(anchor_x, anchor_y, box) <= radius for anchor_x, anchor_y in anchors)


def _distance_point_to_box(px: float, py: float, box) -> float:
    x1 = float(box.x)
    y1 = float(box.y)
    x2 = float(box.x + box.w)
    y2 = float(box.y + box.h)
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def _mask_for_boxes(mask: np.ndarray, boxes) -> np.ndarray:
    selected = np.zeros_like(mask, dtype=np.uint8)
    frame_h, frame_w = mask.shape[:2]
    for box in boxes:
        x1 = max(0, int(box.x))
        y1 = max(0, int(box.y))
        x2 = min(frame_w, int(box.x + box.w))
        y2 = min(frame_h, int(box.y + box.h))
        if x2 <= x1 or y2 <= y1:
            continue
        selected[y1:y2, x1:x2] = cv2.bitwise_or(selected[y1:y2, x1:x2], mask[y1:y2, x1:x2])
    return selected


def _has_large_warm_scene(frame_bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    warm = (
        (((h <= 42) | (h >= 170)) & (s >= 45) & (v >= 115)).astype(np.uint8) * 255
    )
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    warm_fraction = np.count_nonzero(warm) / float(max(1, warm.size))
    if warm_fraction >= 0.14:
        return True

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(warm, connectivity=8)
    frame_area = float(warm.size)
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area / frame_area >= 0.06:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
