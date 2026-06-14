from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


PREDICTION_COLUMNS = [
    "video_name",
    "frame_idx",
    "time_sec",
    "has_sparks",
    "raw_has_sparks",
    "pred_label",
    "n_components",
    "mask_pixels",
    "boxes_json",
]

GROUND_TRUTH_COLUMNS = [
    "video_name",
    "frame_idx",
    "time_sec",
    "image_path",
    "has_sparks_gt",
    "source_gt",
    "comment",
]


def load_predictions(predictions_path: str | Path) -> pd.DataFrame:
    path = Path(predictions_path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*_predictions.csv"))
    else:
        files = []

    if not files:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    frames = [pd.read_csv(file_path) for file_path in files]
    predictions = pd.concat(frames, ignore_index=True)
    if "frame_idx" in predictions.columns:
        predictions["frame_idx"] = predictions["frame_idx"].astype(int)
    if "has_sparks" in predictions.columns:
        predictions["has_sparks_eval"] = _to_binary(predictions["has_sparks"])
    return predictions


def load_ground_truth(ground_truth_path: str | Path) -> pd.DataFrame:
    path = Path(ground_truth_path)
    if not path.exists() or not path.is_file():
        return pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)

    ground_truth = pd.read_csv(path)
    if ground_truth.empty or "has_sparks_gt" not in ground_truth.columns:
        return pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)

    ground_truth = ground_truth.dropna(subset=["has_sparks_gt"]).copy()
    ground_truth = ground_truth[ground_truth["has_sparks_gt"].astype(str).str.strip() != ""]
    if ground_truth.empty:
        return pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)

    ground_truth["frame_idx"] = ground_truth["frame_idx"].astype(int)
    ground_truth["has_sparks_gt_eval"] = _to_binary(ground_truth["has_sparks_gt"])
    return ground_truth


def evaluate_frame_level(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    output_metrics_path: str | Path,
) -> tuple[bool, str]:
    output_path = Path(output_metrics_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(ground_truth_path)
    if ground_truth.empty:
        return False, f"Ground truth is missing or has no filled labels: {ground_truth_path}"

    predictions = load_predictions(predictions_path)
    if predictions.empty:
        message = f"No prediction CSV files found in {predictions_path}"
        output_path.write_text(message + "\n", encoding="utf-8")
        return False, message

    joined = ground_truth.merge(
        predictions,
        on=["video_name", "frame_idx"],
        how="inner",
        suffixes=("_gt", "_pred"),
    )
    joined_path = output_dir / "frame_level_joined.csv"
    joined.to_csv(joined_path, index=False)

    if joined.empty:
        message = "No overlapping rows found between ground truth and predictions."
        output_path.write_text(message + "\n", encoding="utf-8")
        (output_dir / "false_positives.csv").write_text("", encoding="utf-8")
        (output_dir / "false_negatives.csv").write_text("", encoding="utf-8")
        return False, message

    y_true = joined["has_sparks_gt_eval"].astype(int)
    y_pred = joined["has_sparks_eval"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    false_positives = joined[(y_true == 0) & (y_pred == 1)]
    false_negatives = joined[(y_true == 1) & (y_pred == 0)]
    false_positives.to_csv(output_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)

    metrics_text = "\n".join(
        [
            "Frame-level spark detection metrics",
            f"evaluated_frames: {len(joined)}",
            f"true_positives: {tp}",
            f"false_positives: {fp}",
            f"true_negatives: {tn}",
            f"false_negatives: {fn}",
            f"precision: {precision:.6f}",
            f"recall: {recall:.6f}",
            f"f1: {f1:.6f}",
            "",
            "Confusion matrix labels: rows=true [0,1], columns=predicted [0,1]",
            f"[[{tn}, {fp}],",
            f" [{fn}, {tp}]]",
            "",
        ]
    )
    output_path.write_text(metrics_text, encoding="utf-8")
    return True, f"Saved metrics to {output_path}"


def _to_binary(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(
        {
            "1": 1,
            "1.0": 1,
            "true": 1,
            "yes": 1,
            "y": 1,
            "sparks": 1,
            "0": 0,
            "0.0": 0,
            "false": 0,
            "no": 0,
            "n": 0,
            "no sparks": 0,
        }
    ).fillna(0).astype(int)
