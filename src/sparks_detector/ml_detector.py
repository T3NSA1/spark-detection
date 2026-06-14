from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .detector import DetectedRegion, FramePrediction


@dataclass
class YoloDetectorConfig:
    model_path: str | Path
    max_width: int = 960
    image_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.50
    device: str | None = None
    event_labels: tuple[str, ...] = ("welding_arc", "spark_plume")
    suppress_labels: tuple[str, ...] = ("fire", "grinder_spark", "grinder")
    fire_suppression_confidence: float = 0.35
    fire_suppression_area_fraction: float = 0.08
    temporal_window: int = 5
    temporal_min_positive: int = 2
    enable_temporal_smoothing: bool = True


@dataclass
class YoloFrameResult:
    prediction: FramePrediction
    event_boxes: list[DetectedRegion] = field(default_factory=list)
    suppress_boxes: list[DetectedRegion] = field(default_factory=list)


class YoloSparksDetector:
    """YOLO-based detector for welding arcs, main spark plumes, and fire suppression."""

    def __init__(self, config: YoloDetectorConfig) -> None:
        self.config = config
        self._history: deque[bool] = deque(maxlen=max(1, self.config.temporal_window))
        self._model = self._load_model(config.model_path)

    def reset(self) -> None:
        self._history.clear()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        time_sec: float,
    ) -> tuple[np.ndarray, FramePrediction]:
        resized_frame = self._resize_frame(frame_bgr)
        result = self.process_resized_frame(resized_frame, frame_idx, time_sec)
        return resized_frame, result.prediction

    def process_resized_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        time_sec: float,
    ) -> YoloFrameResult:
        detections = self._predict(frame_bgr)
        event_boxes = [box for box in detections if self._matches_label(box.label, self.config.event_labels)]
        suppress_boxes = [
            box for box in detections if self._matches_label(box.label, self.config.suppress_labels)
        ]

        raw_has_sparks = bool(event_boxes)
        suppressed_by_fire = self._should_suppress_for_fire(
            event_boxes=event_boxes,
            suppress_boxes=suppress_boxes,
            frame_shape=frame_bgr.shape[:2],
        )
        if suppressed_by_fire:
            raw_has_sparks = False

        self._history.append(raw_has_sparks)
        has_sparks = self._apply_temporal_smoothing(raw_has_sparks)
        pred_label = self._prediction_label(
            has_sparks=has_sparks,
            raw_has_sparks=raw_has_sparks,
            event_boxes=event_boxes,
            suppress_boxes=suppress_boxes,
            suppressed_by_fire=suppressed_by_fire,
        )

        boxes = event_boxes if raw_has_sparks else suppress_boxes
        prediction = FramePrediction(
            frame_idx=frame_idx,
            time_sec=time_sec,
            has_sparks=has_sparks,
            raw_has_sparks=raw_has_sparks,
            pred_label=pred_label,
            n_components=len(detections),
            mask_pixels=0,
            boxes=boxes,
            mask=None,
        )
        return YoloFrameResult(
            prediction=prediction,
            event_boxes=event_boxes,
            suppress_boxes=suppress_boxes,
        )

    def _predict(self, frame_bgr: np.ndarray) -> list[DetectedRegion]:
        predict_kwargs = {
            "source": frame_bgr,
            "imgsz": self.config.image_size,
            "conf": self.config.conf_threshold,
            "iou": self.config.iou_threshold,
            "verbose": False,
        }
        if self.config.device:
            predict_kwargs["device"] = self.config.device

        results = self._model.predict(**predict_kwargs)
        if not results:
            return []

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return []

        detections: list[DetectedRegion] = []
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            cls_idx = int(box.cls[0].detach().cpu().item())
            conf = float(box.conf[0].detach().cpu().item())
            label = str(names.get(cls_idx, cls_idx))
            detections.append(
                DetectedRegion(
                    x=max(0, x1),
                    y=max(0, y1),
                    w=w,
                    h=h,
                    area=float(w * h),
                    label=label,
                    confidence=conf,
                )
            )

        detections.sort(key=lambda detection: detection.confidence or 0.0, reverse=True)
        return detections

    def _should_suppress_for_fire(
        self,
        event_boxes: list[DetectedRegion],
        suppress_boxes: list[DetectedRegion],
        frame_shape: tuple[int, int],
    ) -> bool:
        if not suppress_boxes:
            return False
        strongest_fire = max((box.confidence or 0.0 for box in suppress_boxes), default=0.0)
        if strongest_fire < self.config.fire_suppression_confidence:
            return False

        frame_h, frame_w = frame_shape
        frame_area = max(1.0, float(frame_h * frame_w))
        fire_area = sum(box.area for box in suppress_boxes)
        fire_area_fraction = fire_area / frame_area
        if not event_boxes:
            return True
        if fire_area_fraction >= self.config.fire_suppression_area_fraction:
            strongest_event = max((box.confidence or 0.0 for box in event_boxes), default=0.0)
            return strongest_fire >= strongest_event - 0.05
        return False

    def _prediction_label(
        self,
        has_sparks: bool,
        raw_has_sparks: bool,
        event_boxes: list[DetectedRegion],
        suppress_boxes: list[DetectedRegion],
        suppressed_by_fire: bool,
    ) -> str:
        if has_sparks or raw_has_sparks:
            if not event_boxes:
                return "sparks"
            best = max(event_boxes, key=lambda box: box.confidence or 0.0)
            return best.label
        if suppress_boxes or suppressed_by_fire:
            best = max(suppress_boxes, key=lambda box: box.confidence or box.area, default=None)
            if best is not None:
                return f"no sparks ({best.label})"
            return "no sparks (suppress)"
        return "no sparks"

    def _apply_temporal_smoothing(self, raw_has_sparks: bool) -> bool:
        if not self.config.enable_temporal_smoothing:
            return raw_has_sparks
        return sum(self._history) >= max(1, self.config.temporal_min_positive)

    def _resize_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        if self.config.max_width <= 0 or width <= self.config.max_width:
            return frame_bgr.copy()
        scale = self.config.max_width / float(width)
        new_size = (self.config.max_width, max(1, int(round(height * scale))))
        return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

    def _matches_label(self, label: str, candidates: Sequence[str]) -> bool:
        normalized = self._normalize_label(label)
        return normalized in {self._normalize_label(candidate) for candidate in candidates}

    def _normalize_label(self, label: str) -> str:
        return label.strip().lower().replace(" ", "_").replace("-", "_")

    def _load_model(self, model_path: str | Path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The YOLO backend requires ultralytics. Install it with: "
                "python -m pip install ultralytics"
            ) from exc
        return YOLO(str(model_path))
