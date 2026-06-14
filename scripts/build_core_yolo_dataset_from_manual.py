from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


BASE_CLASS_NAMES = ("welding_arc", "fire")
GRINDER_CLASS_NAME = "grinder_spark"
GRINDER_MANUAL_DIR = "grinder"
GRINDER_SOURCE_TASK = "04_grinder"
GRINDER_COCO_FILE = "grinder_coco.json"


@dataclass(frozen=True)
class ManualTask:
    name: str
    source_task: str
    class_name: str | None
    include_missing_as_empty: bool = False
    include_only_seed_empty_when_missing: bool = False


TASKS = (
    ManualTask(
        name="easy_spark_plume",
        source_task="01_easy_spark_plume",
        class_name="welding_arc",
    ),
    ManualTask(
        name="medium_welding_arc",
        source_task="02_medium_welding_arc",
        class_name="welding_arc",
    ),
    ManualTask(
        name="nopositive_fire_empty",
        source_task="03_nopositive_fire_empty",
        class_name="fire",
        include_only_seed_empty_when_missing=True,
    ),
)


def main() -> int:
    args = parse_args()
    manual_root = Path(args.manual_root)
    source_root = Path(args.source_root)
    output_root = Path(args.output)
    manifest_path = source_root / "manifest.csv"
    class_names = list(BASE_CLASS_NAMES)
    if args.include_grinder:
        class_names.append(GRINDER_CLASS_NAME)

    if not manifest_path.exists():
        print(f"Source manifest not found: {manifest_path}")
        return 1
    if not manual_root.exists():
        print(f"Manual annotation directory not found: {manual_root}")
        return 1

    if args.clear and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dataset_dirs(output_root)

    rows = load_manifest(manifest_path)
    rows_by_task = group_by_task(rows)

    output_rows: list[dict[str, str]] = []
    for task in TASKS:
        task_rows = rows_by_task.get(task.source_task, [])
        output_rows.extend(convert_task(task, task_rows, manual_root, source_root, output_root, class_names))

    if args.include_grinder:
        output_rows.extend(
            convert_grinder_coco(
                manual_root=manual_root,
                source_root=source_root,
                output_root=output_root,
                class_names=class_names,
                val_every=args.grinder_val_every,
            )
        )

    write_dataset_yaml(output_root, args.yaml_name, class_names)
    write_manifest(output_root / "manual_core_manifest.csv", output_rows)
    print_summary(output_rows)
    print(f"Saved dataset: {output_root}")
    print(f"Dataset YAML: {output_root / args.yaml_name}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset from manual MakeSense annotations."
    )
    parser.add_argument("--manual-root", default="outputs/manual_annotation")
    parser.add_argument("--source-root", default="outputs/cvat_manual_annotation")
    parser.add_argument("--output", default="data/ml_core")
    parser.add_argument("--yaml-name", default="sparks_core.yaml")
    parser.add_argument(
        "--include-grinder",
        action="store_true",
        help="Add grinder COCO annotations as a suppress class.",
    )
    parser.add_argument(
        "--grinder-val-every",
        type=int,
        default=5,
        help="Every Nth grinder frame goes to validation.",
    )
    parser.add_argument("--clear", action="store_true", help="Delete output before rebuilding.")
    args = parser.parse_args()
    if args.grinder_val_every <= 1:
        parser.error("--grinder-val-every must be greater than 1")
    return args


def ensure_dataset_dirs(output_root: Path) -> None:
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def group_by_task(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)
    return grouped


def convert_task(
    task: ManualTask,
    rows: list[dict[str, str]],
    manual_root: Path,
    source_root: Path,
    output_root: Path,
    class_names: list[str],
) -> list[dict[str, str]]:
    manual_label_dir = manual_root / task.name
    source_image_dir = source_root / "tasks" / task.source_task / "images"
    converted_rows: list[dict[str, str]] = []

    for row in rows:
        image_name = row["image_name"]
        source_image = source_image_dir / image_name
        if not source_image.exists():
            continue

        manual_label = manual_label_dir / Path(image_name).with_suffix(".txt").name
        has_manual_label = manual_label.exists() and manual_label.stat().st_size > 0
        should_include_empty = (
            task.include_missing_as_empty
            or (
                task.include_only_seed_empty_when_missing
                and not has_manual_label
                and row.get("current_seed_classes") == "empty"
            )
        )

        if not has_manual_label and not should_include_empty:
            continue

        split = row.get("seed_split") or "train"
        if split not in {"train", "val"}:
            split = "train"

        output_image = output_root / "images" / split / image_name
        output_label = output_root / "labels" / split / Path(image_name).with_suffix(".txt").name
        shutil.copy2(source_image, output_image)

        label_lines: list[str] = []
        if has_manual_label and task.class_name is not None:
            label_lines = remap_label_file(manual_label, task.class_name, class_names)
        output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        converted_rows.append(
            {
                "image_name": image_name,
                "split": split,
                "source_task": task.source_task,
                "manual_task": task.name,
                "video_name": row.get("video_name", ""),
                "frame_idx": row.get("frame_idx", ""),
                "time_sec": row.get("time_sec", ""),
                "class_name": task.class_name if label_lines else "empty",
                "n_boxes": str(len(label_lines)),
                "image_path": str(output_image),
                "label_path": str(output_label),
            }
        )

    return converted_rows


def convert_grinder_coco(
    manual_root: Path,
    source_root: Path,
    output_root: Path,
    class_names: list[str],
    val_every: int,
) -> list[dict[str, str]]:
    coco_path = manual_root / GRINDER_MANUAL_DIR / GRINDER_COCO_FILE
    source_image_dir = source_root / "tasks" / GRINDER_SOURCE_TASK / "images"
    manifest_path = source_root / "tasks" / GRINDER_SOURCE_TASK / "manifest.csv"

    if not coco_path.exists():
        print(f"Skipping grinder labels: COCO file not found: {coco_path}")
        return []
    if not source_image_dir.exists() or not manifest_path.exists():
        print("Skipping grinder labels: source images or manifest missing.")
        return []

    source_rows = load_manifest(manifest_path)
    annotations_by_image = load_coco_boxes(coco_path)
    class_id = class_names.index(GRINDER_CLASS_NAME)
    converted_rows: list[dict[str, str]] = []

    for sample_idx, row in enumerate(source_rows):
        image_name = row["image_name"]
        source_image = source_image_dir / image_name
        if not source_image.exists():
            continue

        split = "val" if sample_idx % val_every == 0 else "train"
        output_image = output_root / "images" / split / image_name
        output_label = output_root / "labels" / split / Path(image_name).with_suffix(".txt").name
        shutil.copy2(source_image, output_image)

        label_lines = [
            format_yolo_line(class_id, *box)
            for box in annotations_by_image.get(image_name, [])
        ]
        output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        converted_rows.append(
            {
                "image_name": image_name,
                "split": split,
                "source_task": GRINDER_SOURCE_TASK,
                "manual_task": GRINDER_MANUAL_DIR,
                "video_name": row.get("video_name", ""),
                "frame_idx": row.get("frame_idx", ""),
                "time_sec": row.get("time_sec", ""),
                "class_name": GRINDER_CLASS_NAME if label_lines else "empty",
                "n_boxes": str(len(label_lines)),
                "image_path": str(output_image),
                "label_path": str(output_label),
            }
        )

    return converted_rows


def load_coco_boxes(coco_path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images_by_id = {
        int(image["id"]): image
        for image in data.get("images", [])
        if "id" in image and "file_name" in image
    }
    boxes_by_image: dict[str, list[tuple[float, float, float, float]]] = {}

    for annotation in data.get("annotations", []):
        image = images_by_id.get(int(annotation.get("image_id", -1)))
        if image is None:
            continue
        width = float(image.get("width", 0))
        height = float(image.get("height", 0))
        if width <= 0 or height <= 0:
            continue

        bbox = annotation.get("bbox")
        if bbox is None:
            bbox = bbox_from_segmentation(annotation.get("segmentation"))
        if bbox is None or len(bbox) < 4:
            continue

        yolo_box = coco_bbox_to_yolo(bbox, width, height)
        if yolo_box is None:
            continue
        boxes_by_image.setdefault(str(image["file_name"]), []).append(yolo_box)

    return boxes_by_image


def bbox_from_segmentation(segmentation: object) -> list[float] | None:
    if not isinstance(segmentation, list):
        return None
    xs: list[float] = []
    ys: list[float] = []
    for polygon in segmentation:
        if not isinstance(polygon, list):
            continue
        coords = [float(value) for value in polygon]
        xs.extend(coords[0::2])
        ys.extend(coords[1::2])
    if not xs or not ys:
        return None
    x_min = min(xs)
    y_min = min(ys)
    return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]


def coco_bbox_to_yolo(
    bbox: object,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    x, y, width, height = [float(value) for value in list(bbox)[:4]]
    x1 = max(0.0, min(image_width, x))
    y1 = max(0.0, min(image_height, y))
    x2 = max(0.0, min(image_width, x + width))
    y2 = max(0.0, min(image_height, y + height))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 1.0 or box_height < 1.0:
        return None
    return (
        (x1 + box_width / 2.0) / image_width,
        (y1 + box_height / 2.0) / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def remap_label_file(label_path: Path, class_name: str, class_names: list[str]) -> list[str]:
    class_id = class_names.index(class_name)
    lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        coords = parts[1:5]
        lines.append(" ".join([str(class_id), *coords]))
    return lines


def format_yolo_line(class_id: int, x_center: float, y_center: float, width: float, height: float) -> str:
    return f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"


def write_dataset_yaml(output_root: Path, yaml_name: str, class_names: list[str]) -> None:
    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend(f"  {class_id}: {class_name}" for class_id, class_name in enumerate(class_names))
    lines.append("")
    (output_root / yaml_name).write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    boxes: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["class_name"])
        counts[key] = counts.get(key, 0) + 1
        boxes[key] = boxes.get(key, 0) + int(row["n_boxes"])

    print("Manual core dataset summary:")
    for key in sorted(counts):
        print(f"  {key[0]:5s} {key[1]:12s} images={counts[key]:4d} boxes={boxes[key]:4d}")
    print(f"  total images={len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
