from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparks_detector.evaluation import evaluate_frame_level


def main() -> int:
    args = parse_args()
    ok, message = evaluate_frame_level(args.predictions, args.ground_truth, args.output)
    print(message)
    return 0 if ok else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frame-level spark/no-spark predictions.")
    parser.add_argument("--predictions", default="outputs/predictions", help="Prediction CSV file or directory.")
    parser.add_argument(
        "--ground-truth",
        default="outputs/eval_frames/manual_labels_filled.csv",
        help="Filled manual-label CSV.",
    )
    parser.add_argument(
        "--output",
        default="outputs/metrics/frame_level_metrics.txt",
        help="Metrics text output path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
