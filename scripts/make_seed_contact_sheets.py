from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


COLORS = {
    "welding_arc": (255, 180, 40),
    "spark_plume": (0, 220, 255),
    "fire": (40, 80, 255),
    "empty": (180, 180, 180),
}


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)

    groups = {
        "spark_plume": manifest[manifest["labels_json"].str.contains("spark_plume", na=False)],
        "welding_arc": manifest[manifest["labels_json"].str.contains("welding_arc", na=False)],
        "fire": manifest[manifest["labels_json"].str.contains("fire", na=False)],
        "empty": manifest[manifest["labels_json"].eq("[]")],
    }

    for group_name, group in groups.items():
        if group.empty:
            continue
        sampled = evenly_sample_rows(group, args.max_samples)
        sheet = build_sheet(sampled, args.tile_width, args.tile_height, args.cols)
        out_path = output_dir / f"seed_{group_name}_contact_sheet.jpg"
        cv2.imwrite(str(out_path), sheet)
        print(f"Saved {out_path} ({len(sampled)} samples)")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render contact sheets for bootstrapped YOLO seed labels.")
    parser.add_argument("--manifest", default="data/ml/seed_labels_manifest.csv")
    parser.add_argument("--output", default="outputs/screenshots/seed_labels")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=220)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def build_sheet(rows: pd.DataFrame, tile_width: int, tile_height: int, cols: int) -> np.ndarray:
    tiles = [render_tile(row, tile_width, tile_height) for _, row in rows.iterrows()]
    rows_count = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows_count * tile_height, cols * tile_width, 3), 24, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        y = (idx // cols) * tile_height
        x = (idx % cols) * tile_width
        sheet[y : y + tile_height, x : x + tile_width] = tile
    return sheet


def evenly_sample_rows(rows: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    if len(rows) <= max_samples:
        return rows
    indices = np.linspace(0, len(rows) - 1, max_samples).round().astype(int)
    return rows.iloc[indices]


def render_tile(row: pd.Series, tile_width: int, tile_height: int) -> np.ndarray:
    image = cv2.imread(str(row["image_path"]))
    if image is None:
        return missing_tile(str(row["image_path"]), tile_width, tile_height)

    original_h, original_w = image.shape[:2]
    scale = min(tile_width / original_w, tile_height / original_h)
    resized_w = max(1, int(original_w * scale))
    resized_h = max(1, int(original_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    tile = np.full((tile_height, tile_width, 3), 12, dtype=np.uint8)
    offset_x = (tile_width - resized_w) // 2
    offset_y = (tile_height - resized_h) // 2
    tile[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized

    labels = json.loads(row["labels_json"])
    if labels:
        for label in labels:
            class_name = label["class"]
            color = COLORS.get(class_name, (255, 255, 255))
            x, y, w, h = label["box"]
            pt1 = (offset_x + int(x * scale), offset_y + int(y * scale))
            pt2 = (offset_x + int((x + w) * scale), offset_y + int((y + h) * scale))
            cv2.rectangle(tile, pt1, pt2, color, 2)
            cv2.putText(tile, class_name, (pt1[0], max(14, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    else:
        cv2.rectangle(tile, (4, 4), (tile_width - 5, tile_height - 5), COLORS["empty"], 1)
        cv2.putText(tile, "empty negative", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["empty"], 1)

    caption = f'{row["video_name"]} f{int(row["frame_idx"])}'
    cv2.rectangle(tile, (0, tile_height - 22), (tile_width, tile_height), (0, 0, 0), -1)
    cv2.putText(tile, caption, (8, tile_height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return tile


def missing_tile(path: str, tile_width: int, tile_height: int) -> np.ndarray:
    tile = np.full((tile_height, tile_width, 3), 32, dtype=np.uint8)
    cv2.putText(tile, "missing image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(tile, path[:48], (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return tile


if __name__ == "__main__":
    raise SystemExit(main())
