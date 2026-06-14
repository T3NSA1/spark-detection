"""Spark and plasma plume detection with OpenCV and optional YOLO backends."""

from .detector import DetectedRegion, FramePrediction, SparksDetector, SparksDetectorConfig
from .ml_detector import YoloDetectorConfig, YoloSparksDetector

__all__ = [
    "DetectedRegion",
    "FramePrediction",
    "SparksDetector",
    "SparksDetectorConfig",
    "YoloDetectorConfig",
    "YoloSparksDetector",
]
