from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SparksDetectorConfig:
    max_width: int = 960
    motion_threshold: int = 22
    brightening_threshold: int = 18
    adaptive_brightness_percentile: float = 99.6
    adaptive_brightness_min_value: int = 170
    local_contrast_threshold: int = 24
    min_component_area: int = 3
    max_component_area: int = 700
    min_event_mask_pixels: int = 35
    min_components_for_event: int = 6
    min_warm_pixels_for_event: int = 30
    min_warm_particle_components_for_event: int = 4
    min_warm_plume_aspect: float = 1.4
    max_warm_plume_density: float = 0.18
    min_fan_components_for_event: int = 18
    min_fan_warm_pixels_for_event: int = 500
    max_fan_density: float = 0.10
    min_warm_group_ratio: float = 0.18
    min_fan_warm_group_ratio: float = 0.12
    min_stream_components_for_event: int = 7
    min_stream_pixels_for_event: int = 60
    min_stream_brightening_pixels: int = 35
    min_stream_flow_pixels: int = 20
    min_stream_aspect: float = 1.8
    max_stream_density: float = 0.14
    max_stream_bbox_frame_fraction: float = 0.45
    enable_optical_flow: bool = True
    optical_flow_max_width: int = 360
    optical_flow_min_magnitude: float = 1.1
    min_motion_overlap_ratio: float = 0.30
    max_solid_region_density: float = 0.45
    ignore_top_border_margin: int = 4
    group_dilation_kernel: int = 13
    group_dilation_iterations: int = 2
    keep_only_main_plume: bool = True
    enable_core_anchored_plume_filter: bool = True
    min_core_pixels_for_anchor_filter: int = 35
    main_plume_anchor_radius_fraction: float = 0.10
    temporal_window: int = 5
    temporal_min_positive: int = 2
    enable_temporal_smoothing: bool = True
    enable_background_subtractor: bool = False
    enable_welding_arc_mask: bool = True
    enable_source_heuristic: bool = True


@dataclass
class DetectedRegion:
    x: int
    y: int
    w: int
    h: int
    area: float
    label: str = "sparks"
    confidence: float | None = None
    n_children: int = 1
    n_particles: int = 0
    n_streaks: int = 0
    warm_pixels: int = 0
    strong_pixels: int = 0
    local_pixels: int = 0
    dynamic_pixels: int = 0
    brightening_pixels: int = 0
    flow_pixels: int = 0
    pixel_count: int = 0
    is_particle: bool = False
    is_streak: bool = False

    def to_dict(self) -> dict[str, int | float | str]:
        data: dict[str, int | float | str] = {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "area": float(self.area),
            "label": self.label,
        }
        if self.confidence is not None:
            data["confidence"] = float(self.confidence)
        return data


@dataclass
class FramePrediction:
    frame_idx: int
    time_sec: float
    has_sparks: bool
    raw_has_sparks: bool
    pred_label: str
    n_components: int
    mask_pixels: int
    boxes: list[DetectedRegion] = field(default_factory=list)
    mask: np.ndarray | None = None


class SparksDetector:
    """Deterministic OpenCV detector for bright dynamic spark-like events."""

    def __init__(self, config: SparksDetectorConfig | None = None) -> None:
        self.config = config or SparksDetectorConfig()
        self._previous_gray: np.ndarray | None = None
        self._raw_history: deque[bool] = deque(maxlen=max(1, self.config.temporal_window))
        self._frames_seen = 0
        self._background_subtractor = None
        if self.config.enable_background_subtractor:
            self._background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=120,
                varThreshold=32,
                detectShadows=False,
            )

    def reset(self) -> None:
        self._previous_gray = None
        self._raw_history.clear()
        self._frames_seen = 0
        if self.config.enable_background_subtractor:
            self._background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=120,
                varThreshold=32,
                detectShadows=False,
            )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        time_sec: float,
    ) -> tuple[np.ndarray, FramePrediction]:
        resized_frame = self._resize_frame(frame_bgr)
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        masks = self._build_masks(resized_frame, hsv, gray)
        final_mask = self._combine_masks(masks)
        cleaned_mask = self._clean_mask(final_mask)

        components, component_mask = self._extract_components(
            cleaned_mask=cleaned_mask,
            dynamic_mask=masks["dynamic"],
            local_contrast_mask=masks["local_contrast"],
            brightening_mask=masks["brightening"],
            flow_mask=masks["flow"],
            orange_mask=masks["orange"],
            white_hot_mask=masks["white_hot"],
            blue_arc_mask=masks["blue_arc"],
        )
        grouped_boxes = self._group_components(
            component_mask=component_mask,
            components=components,
            orange_mask=masks["orange"],
            white_hot_mask=masks["white_hot"],
            blue_arc_mask=masks["blue_arc"],
        )
        components, component_mask, grouped_boxes = self._keep_main_plume_only(
            component_mask=component_mask,
            components=components,
            grouped_boxes=grouped_boxes,
        )

        mask_pixels = int(np.count_nonzero(component_mask))
        strong_core = cv2.bitwise_and(
            component_mask,
            cv2.bitwise_or(masks["white_hot"], masks["blue_arc"]),
        )
        strong_core_pixels = int(np.count_nonzero(strong_core))
        warm_pixels = int(np.count_nonzero(cv2.bitwise_and(component_mask, masks["orange"])))
        blue_pixels = int(np.count_nonzero(cv2.bitwise_and(component_mask, masks["blue_arc"])))
        raw_has_sparks = self._is_spark_event(
            components=components,
            grouped_boxes=grouped_boxes,
            mask_pixels=mask_pixels,
            strong_core_pixels=strong_core_pixels,
            warm_pixels=warm_pixels,
            blue_pixels=blue_pixels,
            frame_shape=component_mask.shape,
        )

        self._raw_history.append(raw_has_sparks)
        has_sparks = self._apply_temporal_smoothing(raw_has_sparks)

        raw_label = self._classify_source(
            boxes=grouped_boxes,
            n_components=len(components),
            final_mask=component_mask,
            orange_mask=masks["orange"],
            white_hot_mask=masks["white_hot"],
            blue_arc_mask=masks["blue_arc"],
        )
        pred_label = raw_label if has_sparks else "no sparks"
        for box in grouped_boxes:
            box.label = raw_label if raw_has_sparks else pred_label

        self._previous_gray = gray
        self._frames_seen += 1

        prediction = FramePrediction(
            frame_idx=frame_idx,
            time_sec=time_sec,
            has_sparks=has_sparks,
            raw_has_sparks=raw_has_sparks,
            pred_label=pred_label,
            n_components=len(components),
            mask_pixels=mask_pixels,
            boxes=grouped_boxes if raw_has_sparks else [],
            mask=component_mask,
        )
        return resized_frame, prediction

    def _resize_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        if self.config.max_width <= 0 or width <= self.config.max_width:
            return frame_bgr.copy()

        scale = self.config.max_width / float(width)
        new_size = (self.config.max_width, max(1, int(round(height * scale))))
        return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

    def _build_masks(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        gray: np.ndarray,
    ) -> dict[str, np.ndarray]:
        h, s, v = cv2.split(hsv)

        tophat_kernel = np.ones((9, 9), dtype=np.uint8)
        local_contrast = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, tophat_kernel)
        local_contrast_mask = (
            (local_contrast >= self.config.local_contrast_threshold).astype(np.uint8) * 255
        )

        orange_mask = (((h >= 7) & (h <= 38) & (s >= 45) & (v >= 125)).astype(np.uint8) * 255)
        white_hot_mask = (((v >= 235) & (s <= 110)).astype(np.uint8) * 255)

        if self.config.enable_welding_arc_mask:
            blue_arc_mask = (
                ((h >= 85) & (h <= 140) & (s >= 20) & (v >= 170)).astype(np.uint8)
                * 255
            )
        else:
            blue_arc_mask = np.zeros_like(v, dtype=np.uint8)

        threshold_value = max(
            float(self.config.adaptive_brightness_min_value),
            float(np.percentile(v, self.config.adaptive_brightness_percentile)),
        )
        adaptive_bright_mask = ((v >= threshold_value).astype(np.uint8) * 255)

        if self._previous_gray is None:
            motion_mask = np.zeros_like(gray, dtype=np.uint8)
            brightening_mask = np.zeros_like(gray, dtype=np.uint8)
            flow_mask = np.zeros_like(gray, dtype=np.uint8)
        else:
            diff = cv2.absdiff(gray, self._previous_gray)
            _, motion_mask = cv2.threshold(
                diff,
                self.config.motion_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            positive_delta = cv2.subtract(gray, self._previous_gray)
            _, brightening_mask = cv2.threshold(
                positive_delta,
                self.config.brightening_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            flow_mask = self._build_flow_mask(self._previous_gray, gray)

        foreground_mask = np.zeros_like(gray, dtype=np.uint8)
        if self._background_subtractor is not None:
            raw_foreground = self._background_subtractor.apply(frame_bgr)
            if self._frames_seen > 0:
                _, foreground_mask = cv2.threshold(raw_foreground, 200, 255, cv2.THRESH_BINARY)
                foreground_mask = cv2.medianBlur(foreground_mask, 3)

        detailed_motion = cv2.bitwise_and(motion_mask, local_contrast_mask)
        flow_details = cv2.bitwise_and(flow_mask, local_contrast_mask)
        foreground_details = cv2.bitwise_and(foreground_mask, local_contrast_mask)
        dynamic_mask = cv2.bitwise_or(brightening_mask, detailed_motion)
        dynamic_mask = cv2.bitwise_or(dynamic_mask, flow_details)
        dynamic_mask = cv2.bitwise_or(dynamic_mask, foreground_details)

        return {
            "orange": orange_mask,
            "white_hot": white_hot_mask,
            "blue_arc": blue_arc_mask,
            "adaptive_bright": adaptive_bright_mask,
            "local_contrast": local_contrast_mask,
            "motion": motion_mask,
            "brightening": brightening_mask,
            "flow": flow_mask,
            "foreground": foreground_mask,
            "dynamic": dynamic_mask,
        }

    def _build_flow_mask(self, previous_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
        if not self.config.enable_optical_flow:
            return np.zeros_like(gray, dtype=np.uint8)

        frame_h, frame_w = gray.shape[:2]
        max_width = max(1, int(self.config.optical_flow_max_width))
        if frame_w > max_width:
            scale = max_width / float(frame_w)
            flow_size = (max_width, max(1, int(round(frame_h * scale))))
            previous_small = cv2.resize(previous_gray, flow_size, interpolation=cv2.INTER_AREA)
            gray_small = cv2.resize(gray, flow_size, interpolation=cv2.INTER_AREA)
        else:
            flow_size = (frame_w, frame_h)
            previous_small = previous_gray
            gray_small = gray

        flow = cv2.calcOpticalFlowFarneback(
            previous_small,
            gray_small,
            None,
            pyr_scale=0.5,
            levels=1,
            winsize=11,
            iterations=1,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        flow_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mask_small = (
            (flow_magnitude >= self.config.optical_flow_min_magnitude).astype(np.uint8) * 255
        )
        if flow_size == (frame_w, frame_h):
            return flow_mask_small
        return cv2.resize(flow_mask_small, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

    def _combine_masks(self, masks: dict[str, np.ndarray]) -> np.ndarray:
        color_mask = cv2.bitwise_or(masks["orange"], masks["white_hot"])
        color_mask = cv2.bitwise_or(color_mask, masks["blue_arc"])

        bright_detail = cv2.bitwise_or(masks["adaptive_bright"], masks["local_contrast"])
        dynamic_color = cv2.bitwise_and(color_mask, masks["dynamic"])
        dynamic_color = cv2.bitwise_and(dynamic_color, bright_detail)

        moving_bright_detail = cv2.bitwise_and(masks["adaptive_bright"], masks["dynamic"])
        moving_bright_detail = cv2.bitwise_and(moving_bright_detail, masks["local_contrast"])

        moving_hot_core = cv2.bitwise_and(masks["white_hot"], masks["adaptive_bright"])
        moving_hot_core = cv2.bitwise_and(moving_hot_core, masks["dynamic"])

        combined = cv2.bitwise_or(dynamic_color, moving_bright_detail)
        combined = cv2.bitwise_or(combined, moving_hot_core)

        stream_detail = cv2.bitwise_and(masks["local_contrast"], masks["dynamic"])
        stream_detail = cv2.bitwise_and(
            stream_detail,
            cv2.bitwise_or(masks["brightening"], masks["flow"]),
        )
        return cv2.bitwise_or(combined, stream_detail)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), dtype=np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.dilate(opened, kernel, iterations=1)

    def _extract_components(
        self,
        cleaned_mask: np.ndarray,
        dynamic_mask: np.ndarray,
        local_contrast_mask: np.ndarray,
        brightening_mask: np.ndarray,
        flow_mask: np.ndarray,
        orange_mask: np.ndarray,
        white_hot_mask: np.ndarray,
        blue_arc_mask: np.ndarray,
    ) -> tuple[list[DetectedRegion], np.ndarray]:
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_h, frame_w = cleaned_mask.shape[:2]
        frame_area = float(frame_h * frame_w)

        valid_contours = []
        components: list[DetectedRegion] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_component_area:
                continue
            if area > self.config.max_component_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = float(w * h)
            if bbox_area <= 0:
                continue
            if bbox_area > frame_area * 0.45:
                continue
            if w > frame_w * 0.90 or h > frame_h * 0.90:
                continue

            single_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
            cv2.drawContours(single_mask, [contour], -1, 255, thickness=-1)
            component_pixels = max(1, int(np.count_nonzero(single_mask)))
            component_crop = single_mask[y : y + h, x : x + w]
            dynamic_crop = dynamic_mask[y : y + h, x : x + w]
            local_crop = local_contrast_mask[y : y + h, x : x + w]
            brightening_crop = brightening_mask[y : y + h, x : x + w]
            flow_crop = flow_mask[y : y + h, x : x + w]
            orange_crop = orange_mask[y : y + h, x : x + w]
            hot_crop = white_hot_mask[y : y + h, x : x + w]
            blue_crop = blue_arc_mask[y : y + h, x : x + w]

            dynamic_pixels = int(np.count_nonzero(cv2.bitwise_and(dynamic_crop, component_crop)))
            local_pixels = int(np.count_nonzero(cv2.bitwise_and(local_crop, component_crop)))
            brightening_pixels = int(
                np.count_nonzero(cv2.bitwise_and(brightening_crop, component_crop))
            )
            flow_pixels = int(np.count_nonzero(cv2.bitwise_and(flow_crop, component_crop)))
            orange_pixels = int(np.count_nonzero(cv2.bitwise_and(orange_crop, component_crop)))
            hot_pixels = int(np.count_nonzero(cv2.bitwise_and(hot_crop, component_crop)))
            blue_pixels = int(np.count_nonzero(cv2.bitwise_and(blue_crop, component_crop)))

            motion_overlap = dynamic_pixels / component_pixels
            if motion_overlap < self.config.min_motion_overlap_ratio:
                continue

            local_ratio = local_pixels / component_pixels
            hot_or_blue_ratio = (hot_pixels + blue_pixels) / component_pixels
            orange_ratio = orange_pixels / component_pixels
            density = component_pixels / bbox_area

            if local_ratio < 0.18 and hot_or_blue_ratio < 0.25:
                continue
            if density > self.config.max_solid_region_density and area > 120 and hot_or_blue_ratio < 0.30:
                continue
            if bbox_area > frame_area * 0.02 and orange_ratio > 0.50 and local_ratio < 0.35:
                continue
            if hot_or_blue_ratio >= 0.50 and orange_ratio < 0.10 and area > 250:
                continue

            valid_contours.append(contour)
            aspect = max(w / max(1.0, float(h)), h / max(1.0, float(w)))
            is_particle = area <= 160 and max(w, h) <= 45
            is_streak = aspect >= 2.0 and density < 0.45 and bbox_area <= 2000
            components.append(
                DetectedRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area=area,
                    n_particles=int(is_particle),
                    n_streaks=int(is_streak),
                    warm_pixels=orange_pixels,
                    strong_pixels=hot_pixels + blue_pixels,
                    local_pixels=local_pixels,
                    dynamic_pixels=dynamic_pixels,
                    brightening_pixels=brightening_pixels,
                    flow_pixels=flow_pixels,
                    pixel_count=component_pixels,
                    is_particle=is_particle,
                    is_streak=is_streak,
                )
            )

        component_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
        if valid_contours:
            cv2.drawContours(component_mask, valid_contours, -1, 255, thickness=-1)

        return components, component_mask

    def _group_components(
        self,
        component_mask: np.ndarray,
        components: list[DetectedRegion],
        orange_mask: np.ndarray,
        white_hot_mask: np.ndarray,
        blue_arc_mask: np.ndarray,
    ) -> list[DetectedRegion]:
        if not np.any(component_mask):
            return []

        kernel_size = max(1, int(self.config.group_dilation_kernel))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        grouped_mask = cv2.dilate(
            component_mask,
            kernel,
            iterations=max(1, int(self.config.group_dilation_iterations)),
        )
        contours, _ = cv2.findContours(grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_h, frame_w = component_mask.shape[:2]
        frame_area = float(frame_h * frame_w)
        boxes: list[DetectedRegion] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = float(w * h)
            if bbox_area <= 0:
                continue
            if bbox_area > frame_area * 0.50:
                continue
            area = float(np.count_nonzero(component_mask[y : y + h, x : x + w]))
            if area < self.config.min_component_area:
                continue

            child_components = [
                component
                for component in components
                if x <= component.x + component.w / 2.0 <= x + w
                and y <= component.y + component.h / 2.0 <= y + h
            ]
            group_component_mask = component_mask[y : y + h, x : x + w]
            group_warm_pixels = int(
                np.count_nonzero(cv2.bitwise_and(group_component_mask, orange_mask[y : y + h, x : x + w]))
            )
            strong_mask = cv2.bitwise_or(
                white_hot_mask[y : y + h, x : x + w],
                blue_arc_mask[y : y + h, x : x + w],
            )
            group_strong_pixels = int(np.count_nonzero(cv2.bitwise_and(group_component_mask, strong_mask)))
            boxes.append(
                DetectedRegion(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area=area,
                    n_children=len(child_components),
                    n_particles=sum(component.is_particle for component in child_components),
                    n_streaks=sum(component.is_streak for component in child_components),
                    warm_pixels=group_warm_pixels,
                    strong_pixels=group_strong_pixels,
                    local_pixels=sum(component.local_pixels for component in child_components),
                    dynamic_pixels=sum(component.dynamic_pixels for component in child_components),
                    brightening_pixels=sum(
                        component.brightening_pixels for component in child_components
                    ),
                    flow_pixels=sum(component.flow_pixels for component in child_components),
                    pixel_count=sum(component.pixel_count for component in child_components),
                )
            )

        boxes.sort(key=lambda box: box.area, reverse=True)
        return boxes

    def _keep_main_plume_only(
        self,
        component_mask: np.ndarray,
        components: list[DetectedRegion],
        grouped_boxes: list[DetectedRegion],
    ) -> tuple[list[DetectedRegion], np.ndarray, list[DetectedRegion]]:
        if not self.config.keep_only_main_plume:
            return components, component_mask, grouped_boxes
        if not np.any(component_mask):
            return components, component_mask, grouped_boxes
        if not grouped_boxes:
            return [], np.zeros_like(component_mask, dtype=np.uint8), []

        main_box = max(
            grouped_boxes,
            key=lambda box: self._main_plume_score(box, component_mask.shape),
        )
        focused_components = self._components_inside_box(components, main_box)
        focused_components = self._filter_components_near_hot_core(
            components=focused_components,
            frame_shape=component_mask.shape,
        )
        if not focused_components:
            return [], np.zeros_like(component_mask, dtype=np.uint8), []

        focused_mask = self._component_subset_mask(component_mask, focused_components)
        refined_box = self._box_from_components(
            components=focused_components,
            component_mask=focused_mask,
        )
        if refined_box is None:
            return [], np.zeros_like(component_mask, dtype=np.uint8), []
        return focused_components, focused_mask, [refined_box]

    def _components_inside_box(
        self,
        components: list[DetectedRegion],
        box: DetectedRegion,
    ) -> list[DetectedRegion]:
        x1 = box.x
        y1 = box.y
        x2 = box.x + box.w
        y2 = box.y + box.h
        return [
            component
            for component in components
            if x1 <= component.x + component.w / 2.0 <= x2
            and y1 <= component.y + component.h / 2.0 <= y2
        ]

    def _filter_components_near_hot_core(
        self,
        components: list[DetectedRegion],
        frame_shape: tuple[int, int],
    ) -> list[DetectedRegion]:
        if not self.config.enable_core_anchored_plume_filter or not components:
            return components

        total_strong = sum(component.strong_pixels for component in components)
        if total_strong < self.config.min_core_pixels_for_anchor_filter:
            return components

        anchor = max(
            components,
            key=lambda component: (
                3.0 * component.strong_pixels
                + 1.4 * component.warm_pixels
                + 0.7 * component.local_pixels
                + 0.8 * component.flow_pixels
                + 0.15 * component.area
            ),
        )
        if anchor.strong_pixels < 6 and total_strong < self.config.min_core_pixels_for_anchor_filter * 2:
            return components

        frame_h, frame_w = frame_shape
        radius = max(
            24.0,
            max(frame_w, frame_h) * float(self.config.main_plume_anchor_radius_fraction),
        )
        anchor_cx = anchor.x + anchor.w / 2.0
        anchor_cy = anchor.y + anchor.h / 2.0

        selected: list[DetectedRegion] = []
        for component in components:
            component_cx = component.x + component.w / 2.0
            component_cy = component.y + component.h / 2.0
            distance = float(np.hypot(component_cx - anchor_cx, component_cy - anchor_cy))
            component_support = max(1, component.pixel_count)
            local_ratio = component.local_pixels / component_support
            hot_pixels = component.strong_pixels + component.warm_pixels

            is_anchor = component is anchor
            is_near_core = component.strong_pixels >= 5 and distance <= radius * 1.15
            is_near_warm_detail = (
                component.warm_pixels >= 8
                and local_ratio >= 0.20
                and distance <= radius * 1.15
            )
            is_near_streak = (
                component.is_streak
                and hot_pixels >= 15
                and distance <= radius * 1.25
            )
            if is_anchor or is_near_core or is_near_warm_detail or is_near_streak:
                selected.append(component)

        selected_pixels = sum(component.pixel_count for component in selected)
        if selected_pixels < self.config.min_event_mask_pixels:
            return components
        return selected

    def _component_subset_mask(
        self,
        component_mask: np.ndarray,
        components: list[DetectedRegion],
    ) -> np.ndarray:
        subset_mask = np.zeros_like(component_mask, dtype=np.uint8)
        for component in components:
            x1 = max(0, component.x)
            y1 = max(0, component.y)
            x2 = min(component_mask.shape[1], component.x + component.w)
            y2 = min(component_mask.shape[0], component.y + component.h)
            if x2 <= x1 or y2 <= y1:
                continue
            subset_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                subset_mask[y1:y2, x1:x2],
                component_mask[y1:y2, x1:x2],
            )
        return subset_mask

    def _box_from_components(
        self,
        components: list[DetectedRegion],
        component_mask: np.ndarray,
    ) -> DetectedRegion | None:
        if not components or not np.any(component_mask):
            return None

        x1 = min(component.x for component in components)
        y1 = min(component.y for component in components)
        x2 = max(component.x + component.w for component in components)
        y2 = max(component.y + component.h for component in components)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(component_mask.shape[1], x2)
        y2 = min(component_mask.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None

        area = float(np.count_nonzero(component_mask[y1:y2, x1:x2]))
        if area < self.config.min_component_area:
            return None

        return DetectedRegion(
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
            area=area,
            n_children=len(components),
            n_particles=sum(component.is_particle for component in components),
            n_streaks=sum(component.is_streak for component in components),
            warm_pixels=sum(component.warm_pixels for component in components),
            strong_pixels=sum(component.strong_pixels for component in components),
            local_pixels=sum(component.local_pixels for component in components),
            dynamic_pixels=sum(component.dynamic_pixels for component in components),
            brightening_pixels=sum(component.brightening_pixels for component in components),
            flow_pixels=sum(component.flow_pixels for component in components),
            pixel_count=sum(component.pixel_count for component in components),
        )

    def _main_plume_score(self, box: DetectedRegion, frame_shape: tuple[int, int]) -> float:
        bbox_area = max(1.0, float(box.w * box.h))
        density = box.area / bbox_area
        aspect = max(
            box.w / max(1.0, float(box.h)),
            box.h / max(1.0, float(box.w)),
        )
        support = max(1.0, float(box.pixel_count))
        local_ratio = box.local_pixels / support
        warm_ratio = box.warm_pixels / max(1.0, float(box.area))
        strong_ratio = box.strong_pixels / max(1.0, float(box.area))
        brightening_ratio = box.brightening_pixels / support
        flow_ratio = box.flow_pixels / support
        frame_h, frame_w = frame_shape
        frame_fraction = bbox_area / max(1.0, float(frame_w * frame_h))

        score = 0.35 * float(box.area)
        score += 1.7 * float(box.warm_pixels)
        score += 1.2 * float(box.strong_pixels)
        score += 0.85 * float(box.local_pixels)
        score += 1.0 * float(box.flow_pixels)
        score += 6.0 * float(box.n_children)
        score += 10.0 * float(box.n_particles)
        score += 18.0 * float(box.n_streaks)

        if aspect >= 1.6 and density <= 0.22:
            score *= 1.25
        if box.strong_pixels >= self.config.min_event_mask_pixels and density <= 0.45:
            score *= 1.20
        if frame_fraction >= 0.10 and aspect < 1.6:
            score *= 0.55
        if strong_ratio >= 0.45 and warm_ratio < 0.12 and aspect < 1.7:
            score *= 0.45
        if brightening_ratio >= 0.85 and flow_ratio < 0.08 and local_ratio < 0.45:
            score *= 0.55
        if box.n_children <= 1 and box.strong_pixels < self.config.min_event_mask_pixels:
            score *= 0.50
        return score

    def _is_spark_event(
        self,
        components: list[DetectedRegion],
        grouped_boxes: list[DetectedRegion],
        mask_pixels: int,
        strong_core_pixels: int,
        warm_pixels: int,
        blue_pixels: int,
        frame_shape: tuple[int, int],
    ) -> bool:
        if mask_pixels < self.config.min_event_mask_pixels:
            return False
        frame_h, _ = frame_shape
        top_margin = max(0, self.config.ignore_top_border_margin)

        if self._has_dynamic_stream_plume(components, top_margin, frame_shape):
            return True

        for box in grouped_boxes:
            if box.y <= top_margin:
                continue
            aspect = max(
                box.w / max(1.0, float(box.h)),
                box.h / max(1.0, float(box.w)),
            )
            group_density = box.area / max(1.0, float(box.w * box.h))
            warm_ratio = box.warm_pixels / max(1.0, float(box.area))
            core_ratio = box.strong_pixels / max(1.0, float(box.area))
            group_has_warm_flow = (
                box.warm_pixels >= self.config.min_warm_pixels_for_event
                and warm_ratio >= self.config.min_fan_warm_group_ratio
            )
            group_has_core = box.strong_pixels >= self.config.min_event_mask_pixels
            if box.y + box.h >= frame_h and not group_has_warm_flow:
                continue
            if group_density > 0.35 and not group_has_core:
                continue
            if (
                group_has_warm_flow
                and box.n_children >= self.config.min_fan_components_for_event
                and box.warm_pixels >= self.config.min_fan_warm_pixels_for_event
                and warm_ratio >= self.config.min_fan_warm_group_ratio
                and group_density <= self.config.max_fan_density
            ):
                return True
            if (
                group_has_warm_flow
                and box.n_children >= self.config.min_components_for_event
                and box.n_particles >= max(4, self.config.min_warm_particle_components_for_event)
                and warm_ratio >= self.config.min_warm_group_ratio
                and group_density <= self.config.max_warm_plume_density
            ):
                return True
            if (
                group_has_warm_flow
                and box.n_children >= self.config.min_components_for_event
                and warm_ratio >= self.config.min_fan_warm_group_ratio
                and (
                    box.n_streaks >= 2
                    or (
                        box.n_particles >= 4
                        and aspect >= 2.0
                        and group_density <= self.config.max_warm_plume_density
                    )
                )
            ):
                return True
            if (
                group_has_warm_flow
                and box.n_children >= 3
                and box.n_streaks >= 1
                and aspect >= 2.2
                and warm_ratio >= self.config.min_warm_group_ratio
            ):
                return True
            if (
                group_has_core
                and core_ratio >= 0.28
                and box.n_children <= 6
                and group_density <= 0.35
                and box.local_pixels >= int(box.area * 0.35)
                and (box.flow_pixels >= int(box.area * 0.10) or warm_ratio >= 0.10)
            ):
                return True
            if group_has_core and core_ratio >= 0.55 and box.n_children <= 3 and group_density <= 0.40:
                return True
            if (
                group_has_core
                and group_has_warm_flow
                and box.n_children >= 3
                and warm_ratio >= self.config.min_warm_group_ratio
                and (box.n_streaks >= 1 or aspect >= 2.0)
            ):
                return True
        return False

    def _has_warm_particle_plume(
        self,
        components: list[DetectedRegion],
        top_margin: int,
    ) -> bool:
        warm_particles = [
            component
            for component in components
            if component.is_particle
            and component.warm_pixels >= 3
            and component.y > top_margin
        ]
        if len(warm_particles) < self.config.min_warm_particle_components_for_event:
            return False

        x1 = min(component.x for component in warm_particles)
        y1 = min(component.y for component in warm_particles)
        x2 = max(component.x + component.w for component in warm_particles)
        y2 = max(component.y + component.h for component in warm_particles)
        plume_w = max(1, x2 - x1)
        plume_h = max(1, y2 - y1)
        aspect = max(plume_w / float(plume_h), plume_h / float(plume_w))
        density = sum(component.area for component in warm_particles) / float(plume_w * plume_h)
        total_warm = sum(component.warm_pixels for component in warm_particles)

        return bool(
            total_warm >= self.config.min_warm_pixels_for_event
            and aspect >= self.config.min_warm_plume_aspect
            and density <= self.config.max_warm_plume_density
        )

    def _has_dynamic_stream_plume(
        self,
        components: list[DetectedRegion],
        top_margin: int,
        frame_shape: tuple[int, int],
    ) -> bool:
        frame_h, frame_w = frame_shape
        stream_components = [
            component
            for component in components
            if (component.is_particle or component.is_streak)
            and component.y > top_margin
            and component.local_pixels >= max(2, int(component.area * 0.20))
            and (
                component.brightening_pixels >= max(2, int(component.area * 0.08))
                or component.flow_pixels >= max(2, int(component.area * 0.08))
            )
        ]
        if len(stream_components) < self.config.min_stream_components_for_event:
            return False

        x1 = min(component.x for component in stream_components)
        y1 = min(component.y for component in stream_components)
        x2 = max(component.x + component.w for component in stream_components)
        y2 = max(component.y + component.h for component in stream_components)
        plume_w = max(1, x2 - x1)
        plume_h = max(1, y2 - y1)
        if plume_w > frame_w * self.config.max_stream_bbox_frame_fraction:
            return False
        if plume_h > frame_h * self.config.max_stream_bbox_frame_fraction:
            return False

        total_area = sum(component.area for component in stream_components)
        total_brightening = sum(component.brightening_pixels for component in stream_components)
        total_flow = sum(component.flow_pixels for component in stream_components)
        aspect = max(plume_w / float(plume_h), plume_h / float(plume_w))
        density = total_area / float(plume_w * plume_h)

        return bool(
            total_area >= self.config.min_stream_pixels_for_event
            and total_brightening >= self.config.min_stream_brightening_pixels
            and total_flow >= self.config.min_stream_flow_pixels
            and aspect >= self.config.min_stream_aspect
            and density <= self.config.max_stream_density
        )

    def _apply_temporal_smoothing(self, raw_has_sparks: bool) -> bool:
        if not self.config.enable_temporal_smoothing:
            return raw_has_sparks
        return sum(self._raw_history) >= max(1, self.config.temporal_min_positive)

    def _classify_source(
        self,
        boxes: list[DetectedRegion],
        n_components: int,
        final_mask: np.ndarray,
        orange_mask: np.ndarray,
        white_hot_mask: np.ndarray,
        blue_arc_mask: np.ndarray,
    ) -> str:
        if not self.config.enable_source_heuristic:
            return "sparks"
        if not boxes or not np.any(final_mask):
            return "sparks"

        mask_pixels = max(1, int(np.count_nonzero(final_mask)))
        orange_pixels = int(np.count_nonzero(cv2.bitwise_and(final_mask, orange_mask)))
        white_pixels = int(np.count_nonzero(cv2.bitwise_and(final_mask, white_hot_mask)))
        blue_pixels = int(np.count_nonzero(cv2.bitwise_and(final_mask, blue_arc_mask)))

        largest = max(boxes, key=lambda box: box.w * box.h)
        aspect = max(
            largest.w / max(1.0, float(largest.h)),
            largest.h / max(1.0, float(largest.w)),
        )
        bbox_density = mask_pixels / max(1.0, float(sum(box.w * box.h for box in boxes)))
        orange_ratio = orange_pixels / mask_pixels
        hot_or_blue_ratio = (white_pixels + blue_pixels) / mask_pixels

        if orange_ratio >= 0.45 and (aspect >= 2.0 or n_components >= 5):
            return "sparks (grinder)"
        if (blue_pixels >= 10 or hot_or_blue_ratio >= 0.45) and aspect < 2.0 and n_components <= 3:
            return "sparks (welding)"
        return "sparks"
