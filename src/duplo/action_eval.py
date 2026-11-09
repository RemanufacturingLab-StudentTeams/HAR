"""Evaluatie van actieclassificatie en temporele localisatie.

Classificatiemodellen voorspellen één label per clip. VLM's zoals Qwen3-VL
en LLaVAction kunnen daarnaast begin- en eindtijden geven. Deze module meet
die twee dingen bewust apart:

* classificatie: welk label hoort bij de clip (vergeleken met de ground
  truth die de clip het meest overlapt);
* temporal: IoU en boundary-fout van voorspelde start/end-tijden, met
  R@IoU-drempels zoals in temporal-action-localization benchmarks.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IOU_THRESHOLDS = (0.3, 0.5, 0.7)


@dataclass(frozen=True)
class ActionInterval:
    action: str
    start_s: float | None
    end_s: float | None
    video: str | None = None


def _as_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().replace(",", ".")
    if not text or text.lower() in ("none", "null", "nan", "n.v.t."):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first(record: dict, *names):
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def _normalise_record(record: dict) -> ActionInterval:
    action = _first(record, "action", "klasse", "label", "class", "class_name")
    if action is None or not str(action).strip():
        raise SystemExit(
            "Actie-ground-truth mist een actie/label-kolom "
            "(action, klasse of label).")
    start = _as_float(_first(record, "start_s", "start", "start_time",
                             "start_timestamp"))
    end = _as_float(_first(record, "end_s", "end", "end_time",
                           "end_timestamp"))
    video = _first(record, "video", "video_path")
    return ActionInterval(
        action=str(action).strip(),
        start_s=start,
        end_s=end,
        video=str(video).strip() if video else None,
    )


def load_action_ground_truth(path: str | Path,
                             video: str | None = None
                             ) -> list[ActionInterval]:
    """Lees ground truth uit CSV (`action,start_s,end_s[,video]`) of JSON."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"Actie-ground-truth bestaat niet: {path}. Zet "
            "'action_ground_truth' in de config of laat de sleutel leeg.")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, dict):
            raw = raw.get("actions") or raw.get("segments") or []
        records = list(raw)
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    intervals = [_normalise_record(record) for record in records]
    if video:
        wanted = Path(str(video)).name
        selected = [item for item in intervals
                    if not item.video or Path(item.video).name == wanted]
        if any(item.video for item in intervals):
            intervals = selected
    return intervals


def interval_iou(a_start: float, a_end: float,
                 b_start: float, b_end: float) -> float:
    """Tijdelijke IoU van twee intervallen (0 bij geen overlap)."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


def _overlap(a_start: float, a_end: float,
             b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _clip_interval(row: dict, fps: float | None) -> tuple[float, float] | None:
    start_s = _as_float(_first(row, "clip_start_s", "timestamp"))
    end_s = _as_float(_first(row, "clip_end_s"))
    if start_s is not None and end_s is not None and end_s > start_s:
        return start_s, end_s
    if fps and fps > 0:
        start_frame = _as_float(_first(row, "clip_start", "frame"))
        end_frame = _as_float(_first(row, "clip_end"))
        if start_frame is not None and end_frame is not None:
            return start_frame / fps, end_frame / fps
    if start_s is not None:
        return start_s, start_s
    return None


def _prediction_interval(row: dict) -> tuple[float, float] | None:
    start = _as_float(_first(row, "start_s", "pred_start_s"))
    end = _as_float(_first(row, "end_s", "pred_end_s"))
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    if end <= start:
        return None
    return start, end


def _label(row: dict) -> str:
    value = _first(row, "klasse", "action", "label", "class_name")
    return str(value).strip() if value is not None else ""


def _same_label(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _relevant_ground_truth(ground_truth: list[ActionInterval],
                           row: dict) -> list[ActionInterval]:
    video = _first(row, "video", "video_path")
    if not video:
        return ground_truth
    wanted = Path(str(video)).name
    selected = [item for item in ground_truth
                if not item.video or Path(item.video).name == wanted]
    if any(item.video for item in ground_truth):
        return selected
    return ground_truth


def evaluate_action_rows(rows: list[dict],
                         ground_truth: list[ActionInterval],
                         fps: float | None = None,
                         iou_thresholds: tuple[float, ...] =
                         DEFAULT_IOU_THRESHOLDS) -> dict:
    """Bereken classificatie- en temporal-metrics voor actierijen."""
    rows = list(rows)
    if not fps:
        # Oude summaries hebben soms geen fps; leid die af uit timestamp.
        for row in rows:
            start_frame = _as_float(_first(row, "clip_start", "frame"))
            timestamp = _as_float(_first(row, "timestamp"))
            if start_frame and timestamp:
                fps = start_frame / timestamp
                break
    gt_intervals = [item for item in ground_truth
                    if item.start_s is not None and item.end_s is not None
                    and item.end_s > item.start_s]
    classification_total = 0
    classification_correct = 0
    classification_matched = 0
    for row in rows:
        label = _label(row)
        clip = _clip_interval(row, fps)
        if not label or clip is None:
            continue
        relevant = [item for item in _relevant_ground_truth(ground_truth, row)
                    if item.start_s is not None and item.end_s is not None]
        if not relevant:
            continue
        best = max(
            relevant,
            key=lambda item: _overlap(clip[0], clip[1],
                                      item.start_s, item.end_s))
        if _overlap(clip[0], clip[1], best.start_s, best.end_s) <= 0:
            classification_total += 1
            continue
        classification_total += 1
        classification_matched += 1
        if _same_label(label, best.action):
            classification_correct += 1

    predictions: list[tuple[int, float, float]] = []
    for index, row in enumerate(rows):
        interval = _prediction_interval(row)
        if interval is not None:
            predictions.append((index, interval[0], interval[1]))

    pairs: list[tuple[float, int, int]] = []
    for pred_index, (_, pred_start, pred_end) in enumerate(predictions):
        for gt_index, item in enumerate(gt_intervals):
            iou = interval_iou(pred_start, pred_end,
                               item.start_s, item.end_s)
            if iou > 0:
                pairs.append((iou, pred_index, gt_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for iou, pred_index, gt_index in pairs:
        if pred_index in used_predictions or gt_index in used_gt:
            continue
        used_predictions.add(pred_index)
        used_gt.add(gt_index)
        matches.append((iou, pred_index, gt_index))

    def recall(threshold: float) -> float:
        if not gt_intervals:
            return 0.0
        hits = sum(1 for iou, _, _ in matches if iou >= threshold)
        return round(hits / len(gt_intervals), 3)

    def precision(threshold: float) -> float:
        if not predictions:
            return 0.0
        hits = sum(1 for iou, _, _ in matches if iou >= threshold)
        return round(hits / len(predictions), 3)

    label_matches = 0
    start_errors = []
    end_errors = []
    for _, pred_index, gt_index in matches:
        pred_row = rows[predictions[pred_index][0]]
        gt = gt_intervals[gt_index]
        if _same_label(_label(pred_row), gt.action):
            label_matches += 1
        pred_start, pred_end = predictions[pred_index][1:]
        start_errors.append(abs(pred_start - gt.start_s))
        end_errors.append(abs(pred_end - gt.end_s))

    accuracy = (round(classification_correct / classification_total, 3)
                if classification_total else 0.0)
    temporal_iou = (round(sum(iou for iou, _, _ in matches) / len(matches), 3)
                    if matches else 0.0)
    return {
        "predictions_total": len(rows),
        "ground_truth_intervals": len(gt_intervals),
        "classification_total": classification_total,
        "classification_matched": classification_matched,
        "classification_correct": classification_correct,
        "classification_accuracy": accuracy,
        "temporal_predictions": len(predictions),
        "temporal_coverage": (round(len(predictions) / len(rows), 3)
                              if rows else 0.0),
        "temporal_matches": len(matches),
        "temporal_iou_mean": temporal_iou,
        "temporal_recall_0.3": recall(0.3),
        "temporal_recall_0.5": recall(0.5),
        "temporal_recall_0.7": recall(0.7),
        "temporal_precision_0.5": precision(0.5),
        "temporal_start_mae_s": (round(sum(start_errors) / len(start_errors), 3)
                                 if start_errors else None),
        "temporal_end_mae_s": (round(sum(end_errors) / len(end_errors), 3)
                               if end_errors else None),
        "temporal_label_accuracy": (
            round(label_matches / len(matches), 3) if matches else 0.0),
        "iou_thresholds": list(iou_thresholds),
    }


def evaluate_action_csv(csv_path: str | Path,
                        ground_truth: list[ActionInterval],
                        fps: float | None = None) -> dict:
    """Lees een ``*_actions.csv`` en evalueer die tegen ground truth."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return evaluate_action_rows(rows, ground_truth, fps=fps)
