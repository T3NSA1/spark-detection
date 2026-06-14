from __future__ import annotations

import cv2
import numpy as np

from .detector import FramePrediction


def draw_prediction(
    frame_bgr: np.ndarray,
    prediction: FramePrediction,
    mask: np.ndarray | None,
    alpha: float = 0.35,
    max_boxes: int = 8,
    max_labels: int = 3,
) -> np.ndarray:
    output = frame_bgr.copy()

    if mask is not None and np.any(mask):
        overlay = output.copy()
        overlay[mask > 0] = (0, 140, 255)
        output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

    boxes_to_draw = sorted(prediction.boxes, key=lambda box: box.area, reverse=True)[:max_boxes]
    for box_idx, box in enumerate(boxes_to_draw):
        color = _label_color(box.label)
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.h
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        if box_idx < max_labels:
            _draw_text(output, box.label, (x1, max(16, y1 - 6)), color)

    if prediction.has_sparks:
        status = "SPARKS"
        if prediction.pred_label and prediction.pred_label not in {"sparks", "no sparks"}:
            status = f"{status} ({prediction.pred_label})"
    else:
        status = prediction.pred_label or "no sparks"
    status_color = (0, 220, 255) if prediction.has_sparks else (220, 220, 220)
    boxes_note = f" | boxes shown {len(boxes_to_draw)}/{len(prediction.boxes)}"
    status_text = (
        f"{status} | frame {prediction.frame_idx} | "
        f"components {prediction.n_components} | mask pixels {prediction.mask_pixels}{boxes_note}"
    )
    _draw_text(output, status_text, (10, 24), status_color)
    return output


def _label_color(label: str) -> tuple[int, int, int]:
    if "fire" in label:
        return (40, 40, 255)
    if "welding" in label:
        return (255, 180, 40)
    if "grinder" in label:
        return (0, 180, 255)
    return (0, 255, 180)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(text_h + baseline + 2, y)
    cv2.rectangle(
        image,
        (x - 3, y - text_h - baseline - 3),
        (x + text_w + 3, y + baseline + 3),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
