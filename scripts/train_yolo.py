from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    args = parse_args()
    ok = validate_dataset(args.data)
    if not ok:
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Missing dependency: ultralytics")
        print("Install it with: python -m pip install ultralytics")
        return 1

    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "exist_ok": args.exist_ok,
        "resume": args.resume,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    print(f"Training finished. Check: {Path(args.project) / args.name}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small YOLO model for welding arcs, spark plumes, and fire.")
    parser.add_argument("--data", default="data/ml/sparks.yaml", help="YOLO dataset YAML.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model, e.g. yolo11n.pt or yolo11s.pt.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size. Lower this if GPU memory is low.")
    parser.add_argument("--device", default=None, help="Training device, e.g. 0, cuda:0, or cpu.")
    parser.add_argument("--project", default="runs/sparks_yolo", help="Ultralytics project directory.")
    parser.add_argument("--name", default="train", help="Run name.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run directory.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted Ultralytics run.")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.batch == 0:
        parser.error("--batch must be non-zero")
    return args


def validate_dataset(data_yaml: str) -> bool:
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"Dataset YAML not found: {data_path}")
        return False

    dataset_root = data_path.parent
    train_images = dataset_root / "images" / "train"
    val_images = dataset_root / "images" / "val"
    train_count = count_images(train_images)
    val_count = count_images(val_images)

    if train_count == 0:
        print(f"No training images found in {train_images}")
        return False
    if val_count == 0:
        print(f"No validation images found in {val_images}")
        return False

    train_labels = count_label_files(dataset_root / "labels" / "train")
    val_labels = count_label_files(dataset_root / "labels" / "val")
    if train_labels == 0:
        print(f"No training label files found in {dataset_root / 'labels' / 'train'}")
        return False
    if val_labels == 0:
        print(f"No validation label files found in {dataset_root / 'labels' / 'val'}")
        return False

    print(
        "Dataset check: "
        f"train_images={train_count}, val_images={val_count}, "
        f"train_labels={train_labels}, val_labels={val_labels}"
    )
    return True


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def count_label_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*.txt") if path.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
