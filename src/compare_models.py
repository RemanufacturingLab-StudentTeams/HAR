"""Benchmarkrapport over alle modelcategorieën van het project.

Het project heeft geen enkel soort model: objectdetectie/segmentatie,
handdetectie, 3D-handvoorspelling (VITRA) en actieclassificatie. Dit rapport
groepeert de runs per taak en gebruikt per taak de passende metric:

* objectdetectie: tooldekking, tool-frames, confidence, doorvoer en de
  bestaande weegbare score;
* handdetectie: frames met handen, landmarkrijen en doorvoer;
* 3D-handvoorspelling: actierijen, handen, chunkgrootte en doorvoer;
* actieclassificatie: clip-labels, confidence, classificatie-accuracy en
  temporele IoU/R@IoU wanneer ``action_ground_truth`` is ingesteld.

Alleen objectdetectie krijgt een samengestelde score. De andere taken zijn
niet met die score te vergelijken en krijgen daarom eigen tabellen.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from duplo.action_eval import evaluate_action_csv, load_action_ground_truth
from duplo.benchmark import (
    TASK_3D_HAND_PREDICTION,
    TASK_ACTION_CLASSIFICATION,
    TASK_HAND_DETECTION,
    TASK_LABELS,
    TASK_METRICS,
    TASK_OBJECT_DETECTION,
    TASKS,
    system_info,
)
from duplo.config import load_config

CSV_SUFFIX = "_detections.csv"
TRACKING_SUFFIX = "_tracking_history.csv"
ACTION_SUFFIX = "_actions.csv"
VITRA_ACTION_CSV = "vitra_actions.csv"
HANDPOSE_CSV = "hand_landmarks.csv"
DEFAULT_WEIGHTS = {"coverage": 0.4, "presence": 0.3, "speed": 0.3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output",
                        help="map voor het rapport; standaard <output_dir>/comparison")
    parser.add_argument("--task", choices=("all", *TASKS), default="all",
                        help="beperk het rapport tot één benchmarktaak")
    # Bat-level flag (run_common.bat): zie duplo.cli.parse_args.
    parser.add_argument("--upgrade", nargs="?", const=True, default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def find_runs(output_dir: Path) -> list[Path]:
    """Alle detectie-CSV's van de modeltesten, met summary ernaast."""
    return sorted(output_dir.glob(f"*/*{CSV_SUFFIX}"))


def find_tracking_runs(output_dir: Path) -> list[Path]:
    """Alle ByteTrack-CSV's (`*_history.csv`, bv. `tracking_history.csv`)."""
    seen: dict[str, Path] = {}
    for pattern in ("*/*_history.csv", f"*/*{TRACKING_SUFFIX}"):
        for path in sorted(output_dir.glob(pattern)):
            seen[str(path)] = path
    return sorted(seen.values())


def find_action_runs(output_dir: Path) -> list[Path]:
    """Alle actieclassificatie-CSV's (`*_actions.csv`).

    VITRA schrijft ook een `vitra_actions.csv`, maar dat zijn continue
    handacties (geen clipclassificatie) en horen niet in deze tabel.
    """
    return sorted(path for path in output_dir.glob(f"*/*{ACTION_SUFFIX}")
                  if path.name != VITRA_ACTION_CSV)


def find_handpose_runs(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob(f"*/{HANDPOSE_CSV}"))


def find_vitra_runs(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob(f"*/{VITRA_ACTION_CSV}"))


def load_summary(csv_path: Path) -> dict:
    name = csv_path.name
    candidates: list[Path] = []
    if name.endswith(CSV_SUFFIX):
        candidates.append(csv_path.with_name(
            name[: -len(CSV_SUFFIX)] + "_summary.json"))
    if name.endswith(TRACKING_SUFFIX):
        candidates.append(csv_path.with_name(
            name[: -len(TRACKING_SUFFIX)] + "_summary.json"))
    if name.endswith(ACTION_SUFFIX):
        candidates.append(csv_path.with_name(
            name[: -len(ACTION_SUFFIX)] + "_summary.json"))
    if name.endswith("_history.csv"):
        candidates.append(csv_path.with_name(
            name[: -len("_history.csv")] + "_summary.json"))
    # Fallback: enige *_summary.json in dezelfde map (bv. handpose).
    candidates.extend(sorted(csv_path.parent.glob("*_summary.json")))
    for summary_path in candidates:
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def _class_series(data: pd.DataFrame) -> pd.Series:
    """Klasse-kolom ongeacht schema (`klasse` of `class_name`)."""
    col = "klasse" if "klasse" in data.columns else (
        "class_name" if "class_name" in data.columns else None)
    if col is None:
        return pd.Series([], dtype=str)
    return data[col].astype(str).str.strip().str.lower()


def _frame_series(data: pd.DataFrame) -> pd.Series:
    """Frame-kolom ongeacht schema (`frame` of `frame_number`)."""
    col = "frame" if "frame" in data.columns else (
        "frame_number" if "frame_number" in data.columns else None)
    if col is None:
        return pd.Series([], dtype=int)
    return data[col]


def summarize_run(csv_path: Path, tool_classes: list[str]) -> dict:
    """Bereken de vergelijkingscijfers van één detectiemodel."""
    data = pd.read_csv(csv_path)
    summary = load_summary(csv_path)
    classes = _class_series(data)
    frames = _frame_series(data)
    tool_rows = data[classes.isin(tool_classes)]
    found = sorted(set(classes[classes.isin(tool_classes)]))
    frames_processed = int(summary.get("frames_processed") or
                           (frames.nunique() if len(frames) else 0))
    frame_col = "frame" if "frame" in data.columns else (
        "frame_number" if "frame_number" in data.columns else None)
    tool_frames = (int(tool_rows[frame_col].nunique())
                   if (frame_col is not None and len(tool_rows)) else 0)
    return {
        "stap": csv_path.parent.name,
        "task": TASK_OBJECT_DETECTION,
        "model": summary.get("model", "onbekend (geen summary gevonden)"),
        "frame_stride": summary.get("frame_stride"),
        "frames_processed": frames_processed,
        "detections": len(data),
        "detections_per_frame": round(len(data) / frames_processed, 3)
        if frames_processed else 0.0,
        "tool_classes_found": len(found),
        "tool_coverage": round(len(found) / len(tool_classes), 3)
        if tool_classes else 0.0,
        "tool_detections": len(tool_rows),
        "tool_frames": tool_frames,
        "tool_presence": round(tool_frames / frames_processed, 3)
        if frames_processed else 0.0,
        "tool_confidence_mean": (round(float(tool_rows["confidence"].mean()), 3)
                                 if len(tool_rows) else None),
        "mask_area_mean": (round(float(data["mask_area"].mean()), 1)
                           if "mask_area" in data else None),
        "inference_ms_median": summary.get("inference_ms_median"),
        "inference_fps": summary.get("inference_fps"),
        "pipeline_fps": summary.get("pipeline_fps"),
        "top_classes": ", ".join(
            f"{name} ({count})" for name, count in
            classes.value_counts().head(3).items()),
    }


def add_scores(runs: list[dict], weights: dict, target_fps: float) -> None:
    """Heuristische score: dekking, aanwezigheid en doorvoer (0-100)."""
    for run in runs:
        speed = min(1.0, (run["pipeline_fps"] or 0.0) / target_fps)
        score = (weights["coverage"] * run["tool_coverage"]
                 + weights["presence"] * run["tool_presence"]
                 + weights["speed"] * speed)
        run["score"] = round(100 * score, 1)
        run["speed_target"] = round(speed, 3)
    runs.sort(key=lambda run: run["score"], reverse=True)


def tool_matrix(csv_paths: list[Path], tool_classes: list[str]) -> pd.DataFrame:
    """Detecties per gereedschapsklasse per modeltest."""
    columns = {}
    for csv_path in csv_paths:
        data = pd.read_csv(csv_path)
        classes = _class_series(data)
        columns[csv_path.parent.name] = classes.value_counts()
    matrix = pd.DataFrame(columns).reindex(tool_classes).fillna(0).astype(int)
    return matrix


def summarize_tracking(csv_path: Path) -> dict:
    """Beknopte tracking-statistieken (geen detectie-score)."""
    data = pd.read_csv(csv_path)
    summary = load_summary(csv_path)
    frame_col = "frame_number" if "frame_number" in data.columns else (
        "frame" if "frame" in data.columns else None)
    frames = int(summary.get("frames_processed") or
                 (data[frame_col].nunique() if frame_col else 0))
    tracks = int(data["track_id"].nunique()) if "track_id" in data else 0
    statuses = (data["status"].value_counts().to_dict()
                if "status" in data else {})
    return {
        "stap": csv_path.parent.name,
        "task": TASK_OBJECT_DETECTION,
        "model": summary.get("model", "onbekend (geen summary gevonden)"),
        "frames_processed": frames,
        "tracks": tracks,
        "rows": len(data),
        "rows_per_frame": round(len(data) / frames, 3) if frames else 0.0,
        "status_confirmed": int(statuses.get("confirmed", 0)),
        "status_tentative": int(statuses.get("tentative", 0)),
        "status_occluded": int(statuses.get("occluded", 0)),
        "inference_ms_median": summary.get("inference_ms_median"),
        "pipeline_fps": summary.get("pipeline_fps"),
    }


def summarize_handpose(csv_path: Path) -> dict:
    """Samenvatting van een MediaPipe-handlandmarkrun."""
    data = pd.read_csv(csv_path)
    summary = load_summary(csv_path)
    frame_col = "frame" if "frame" in data.columns else None
    frames = int(summary.get("frames_processed") or
                 (data[frame_col].nunique() if frame_col else 0))
    return {
        "stap": csv_path.parent.name,
        "task": TASK_HAND_DETECTION,
        "model": summary.get("model", "onbekend (geen summary gevonden)"),
        "frames_processed": frames,
        "landmark_rows": len(data),
        "frames_with_hands": (int(data[frame_col].nunique())
                              if frame_col else 0),
        "hands": (int(data["hand"].nunique())
                  if "hand" in data.columns else None),
        "inference_ms_median": summary.get("inference_ms_median"),
        "inference_fps": summary.get("inference_fps"),
        "pipeline_fps": summary.get("pipeline_fps"),
    }


def summarize_vitra(csv_path: Path) -> dict:
    """Samenvatting van een VITRA 3D-handactierun."""
    data = pd.read_csv(csv_path)
    summary = load_summary(csv_path)
    frame_col = "frame" if "frame" in data.columns else None
    frames = int(summary.get("frames_processed") or
                 (data[frame_col].nunique() if frame_col else 0))
    hands = sorted(data["hand"].dropna().unique()) if "hand" in data else []
    steps = int(data["step"].max() + 1) if "step" in data and len(data) else 0
    samples = (int(data["sample"].nunique())
               if "sample" in data and len(data) else 0)
    return {
        "stap": csv_path.parent.name,
        "task": TASK_3D_HAND_PREDICTION,
        "model": summary.get("model", "onbekend (geen summary gevonden)"),
        "frames_processed": frames,
        "action_rows": len(data),
        "hands": ", ".join(str(hand) for hand in hands) or None,
        "chunk_steps": steps,
        "samples": samples,
        "inference_ms_median": summary.get("inference_ms_median"),
        "inference_fps": summary.get("inference_fps"),
        "pipeline_fps": summary.get("pipeline_fps"),
    }


def summarize_action_run(csv_path: Path,
                         ground_truth: list | None = None) -> dict:
    """Actieclassificatie + optionele temporele evaluatie per model."""
    data = pd.read_csv(csv_path)
    summary = load_summary(csv_path)
    classes = _class_series(data)
    clips = int(summary.get("clips_processed") or len(data))
    counts = classes.value_counts()
    top_label = str(counts.index[0]) if len(counts) else "n.v.t."
    top_count = int(counts.iloc[0]) if len(counts) else 0
    confidence = None
    if "confidence" in data and len(data):
        values = pd.to_numeric(data["confidence"], errors="coerce")
        if values.notna().any():
            confidence = round(float(values.mean()), 3)
    evaluation = summary.get("evaluation")
    if evaluation is None and ground_truth:
        fps = summary.get("fps")
        try:
            fps = float(fps) if fps else None
        except (TypeError, ValueError):
            fps = None
        evaluation = evaluate_action_csv(csv_path, ground_truth, fps=fps)
    temporal = bool(summary.get("temporal"))
    temporal_coverage = (evaluation or {}).get("temporal_coverage")
    if "start_s" in data and len(data):
        start_values = pd.to_numeric(data["start_s"], errors="coerce")
        if start_values.notna().any():
            temporal = True
            if temporal_coverage is None:
                temporal_coverage = round(
                    float(start_values.notna().mean()), 3)
    if temporal and temporal_coverage is None:
        temporal_coverage = 0.0
    return {
        "stap": csv_path.parent.name,
        "task": TASK_ACTION_CLASSIFICATION,
        "model": summary.get("model", "onbekend (geen summary gevonden)"),
        "clips": clips,
        "labels_found": int(classes.nunique()),
        "top_label": top_label,
        "top_label_share": round(top_count / clips, 3) if clips else 0.0,
        "confidence_mean": confidence,
        "inference_ms_median": summary.get("inference_ms_median"),
        "inference_fps": summary.get("inference_fps"),
        "pipeline_fps": summary.get("pipeline_fps"),
        "clip_frames": summary.get("clip_frames"),
        "clip_span": summary.get("clip_span"),
        "clip_stride": summary.get("clip_stride"),
        "temporal": temporal,
        "evaluation": evaluation,
        "classification_accuracy": (evaluation or {}).get(
            "classification_accuracy"),
        "temporal_iou_mean": (evaluation or {}).get("temporal_iou_mean"),
        "temporal_recall_0.5": (evaluation or {}).get(
            "temporal_recall_0.5"),
        "temporal_coverage": temporal_coverage,
    }


def action_matrix(csv_paths: list[Path],
                  action_classes: list[str] | None = None) -> pd.DataFrame:
    """Aantal clips per actieklasse per modeltest."""
    columns = {}
    for csv_path in csv_paths:
        data = pd.read_csv(csv_path)
        classes = _class_series(data)
        columns[csv_path.parent.name] = classes.value_counts()
    matrix = pd.DataFrame(columns).fillna(0).astype(int)
    ordered = [str(label).strip().lower() for label in (action_classes or [])]
    if ordered:
        extra = [label for label in matrix.index if label not in ordered]
        matrix = matrix.reindex(ordered + extra).fillna(0).astype(int)
    return matrix


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def _overview_rows(detection_runs: list[dict],
                   tracking_runs: list[dict],
                   handpose_runs: list[dict],
                   vitra_runs: list[dict],
                   action_runs: list[dict]) -> list[list[str]]:
    rows = []
    if detection_runs:
        rows.append([TASK_LABELS[TASK_OBJECT_DETECTION], "detectie/segmentatie",
                     len(detection_runs), TASK_METRICS[TASK_OBJECT_DETECTION]])
    if tracking_runs:
        rows.append([TASK_LABELS[TASK_OBJECT_DETECTION], "tracking",
                     len(tracking_runs), "tracks, rijen/frame, fps"])
    if handpose_runs:
        rows.append([TASK_LABELS[TASK_HAND_DETECTION], "MediaPipe landmarks",
                     len(handpose_runs), TASK_METRICS[TASK_HAND_DETECTION]])
    if vitra_runs:
        rows.append([TASK_LABELS[TASK_3D_HAND_PREDICTION], "VITRA",
                     len(vitra_runs), TASK_METRICS[TASK_3D_HAND_PREDICTION]])
    if action_runs:
        rows.append([TASK_LABELS[TASK_ACTION_CLASSIFICATION], "clip-classificatie",
                     len(action_runs), TASK_METRICS[TASK_ACTION_CLASSIFICATION]])
    return rows


def build_report(detection_runs: list[dict], matrix: pd.DataFrame, cfg: dict,
                 weights: dict, target_fps: float,
                 tracking_runs: list[dict] | None = None,
                 handpose_runs: list[dict] | None = None,
                 vitra_runs: list[dict] | None = None,
                 action_runs: list[dict] | None = None,
                 action_classes_matrix: pd.DataFrame | None = None) -> str:
    tool_classes = cfg.get("tool_classes", [])
    tracking_runs = tracking_runs or []
    handpose_runs = handpose_runs or []
    vitra_runs = vitra_runs or []
    action_runs = action_runs or []

    parts = [
        "# Benchmark: objectdetectie, handdetectie, 3D-hand en acties",
        "",
        (f"Video: `{cfg['video_path']}` · frame_stride "
         f"`{cfg.get('frame_stride', 1)}`"),
        "",
        "## Overzicht per taak",
        "",
        markdown_table(
            ["Taak", "Variant", "Modellen", "Belangrijkste metrics"],
            _overview_rows(detection_runs, tracking_runs, handpose_runs,
                           vitra_runs, action_runs)),
        "",
    ]

    if detection_runs:
        header = ["Model", "Modelnaam", "Toolklassen (gevonden/doel)",
                  "Tool-detecties", "Tool-frames", "Gem. conf.", "Det./frame",
                  "Inferentie (ms)", "Doorvoer (fps)", "Score"]
        rows = []
        for run in detection_runs:
            rows.append([
                run["stap"], run["model"],
                f"{run['tool_classes_found']}/{len(tool_classes)}",
                run["tool_detections"],
                f"{100 * run['tool_presence']:.1f}%",
                "n.v.t." if run["tool_confidence_mean"] is None
                else f"{run['tool_confidence_mean']:.2f}",
                f"{run['detections_per_frame']:.2f}",
                "n.v.t." if run["inference_ms_median"] is None
                else f"{run['inference_ms_median']:.1f}",
                "n.v.t." if run["pipeline_fps"] is None
                else f"{run['pipeline_fps']:.1f}",
                run["score"],
            ])
        matrix_rows = [
            [tool, *[int(matrix.loc[tool, column])
                     for column in matrix.columns]]
            for tool in matrix.index]
        top_rows = [[run["stap"], run["top_classes"]]
                    for run in detection_runs]
        parts.extend([
            "## Objectdetectie / segmentatie",
            "",
            markdown_table(header, rows),
            "",
            ("Score = 100 x ("
             f"{weights['coverage']} x dekking + "
             f"{weights['presence']} x tool-frames + "
             f"{weights['speed']} x min(1, doorvoer/{target_fps:g}))"),
            "",
            "### Detecties per gereedschapsklasse",
            "",
            markdown_table(["Gereedschap", *matrix.columns], matrix_rows),
            "",
            "### Meest gedetecteerde klassen per model",
            "",
            markdown_table(["Model", "Top-3 klassen"], top_rows),
            "",
        ])

    if tracking_runs:
        track_header = ["Model", "Modelnaam", "Tracks", "Rijen/frame",
                        "Confirmed", "Tentative", "Occluded",
                        "Inferentie (ms)", "Doorvoer (fps)"]
        track_rows = [[run["stap"], run["model"], run["tracks"],
                       f"{run['rows_per_frame']:.2f}",
                       run["status_confirmed"], run["status_tentative"],
                       run["status_occluded"],
                       "n.v.t." if run["inference_ms_median"] is None
                       else f"{run['inference_ms_median']:.1f}",
                       "n.v.t." if run["pipeline_fps"] is None
                       else f"{run['pipeline_fps']:.1f}"]
                      for run in tracking_runs]
        parts.extend([
            "## Tracking (ByteTrack, apart van detectie-score)",
            "",
            markdown_table(track_header, track_rows),
            "",
            ("Tracking wordt niet meegewogen in de detectiescore; "
             "`confirmed/tentative/occluded` tellen rijen per status."),
            "",
        ])

    if handpose_runs:
        hand_header = ["Model", "Modelnaam", "Frames", "Landmarkrijen",
                       "Frames met handen", "Handen", "Inferentie (ms)",
                       "Doorvoer (fps)"]
        hand_rows = [[run["stap"], run["model"], run["frames_processed"],
                      run["landmark_rows"], run["frames_with_hands"],
                      "n.v.t." if run["hands"] is None else run["hands"],
                      "n.v.t." if run["inference_ms_median"] is None
                      else f"{run['inference_ms_median']:.1f}",
                      "n.v.t." if run["pipeline_fps"] is None
                      else f"{run['pipeline_fps']:.1f}"]
                     for run in handpose_runs]
        parts.extend([
            "## Handdetectie (MediaPipe)",
            "",
            markdown_table(hand_header, hand_rows),
            "",
            ("Dit is landmarkdetectie, geen actieclassificatie; er is geen "
             "accuracy zonder geannoteerde handposities."),
            "",
        ])

    if vitra_runs:
        vitra_header = ["Model", "Modelnaam", "Frames", "Actierijen",
                        "Handen", "Chunkstappen", "Samples",
                        "Inferentie (ms)", "Doorvoer (fps)"]
        vitra_rows = [[run["stap"], run["model"], run["frames_processed"],
                       run["action_rows"], run["hands"] or "n.v.t.",
                       run["chunk_steps"], run["samples"],
                       "n.v.t." if run["inference_ms_median"] is None
                       else f"{run['inference_ms_median']:.1f}",
                       "n.v.t." if run["pipeline_fps"] is None
                       else f"{run['pipeline_fps']:.1f}"]
                      for run in vitra_runs]
        parts.extend([
            "## 3D-handvoorspelling (VITRA)",
            "",
            markdown_table(vitra_header, vitra_rows),
            "",
            ("VITRA voorspelt een continue actiechunk per gesampled frame. "
             "Zonder ground-truth-handtrajecten zijn alleen aantal, snelheid "
             "en consistentie te vergelijken."),
            "",
        ])

    if action_runs:
        action_header = ["Model", "Modelnaam", "Clips", "Labels",
                         "Top-label", "Aandeel", "Gem. conf.",
                         "Class. acc.", "Temp. coverage", "mIoU", "R@0.5",
                         "Inferentie (ms)", "Clips/s"]
        action_rows = []
        for run in action_runs:
            action_rows.append([
                run["stap"], run["model"], run["clips"],
                run["labels_found"], run["top_label"],
                f"{100 * run['top_label_share']:.1f}%",
                "n.v.t." if run["confidence_mean"] is None
                else f"{run['confidence_mean']:.2f}",
                "n.v.t." if run["classification_accuracy"] is None
                else f"{100 * run['classification_accuracy']:.1f}%",
                "n.v.t." if run["temporal_coverage"] is None
                else f"{100 * run['temporal_coverage']:.1f}%",
                "n.v.t." if run["temporal_iou_mean"] is None
                else f"{run['temporal_iou_mean']:.3f}",
                "n.v.t." if run["temporal_recall_0.5"] is None
                else f"{run['temporal_recall_0.5']:.3f}",
                "n.v.t." if run["inference_ms_median"] is None
                else f"{run['inference_ms_median']:.1f}",
                "n.v.t." if run["pipeline_fps"] is None
                else f"{run['pipeline_fps']:.1f}",
            ])
        parts.extend([
            "## Actieclassificatie (apart van detectie-score)",
            "",
            markdown_table(action_header, action_rows),
            "",
            ("Deze modellen classificeren een hele clip. `Class. acc.`, "
             "`mIoU` en `R@0.5` verschijnen alleen als "
             "`action_ground_truth` is ingesteld. `Temp. coverage` is het "
             "aandeel clips waarvoor het model start/end-tijden gaf en kan "
             "ook zonder ground truth worden getoond."),
            "",
        ])
        if action_classes_matrix is not None and not action_classes_matrix.empty:
            action_matrix_rows = [
                [label, *[int(action_classes_matrix.loc[label, column])
                          for column in action_classes_matrix.columns]]
                for label in action_classes_matrix.index]
            parts.extend([
                "## Clips per actieklasse",
                "",
                markdown_table(["Actie", *action_classes_matrix.columns],
                               action_matrix_rows),
                "",
            ])

    parts.extend([
        "## Kanttekeningen",
        "",
        "* De objectdetectiescore is een heuristic zonder geannoteerde ground",
        "  truth; dekking en tool-frames zijn proxies, geen accuracy.",
        "* Confidence is niet tussen modelfamilies te vergelijken: elke",
        "  familie kalibreert scores anders. Generatieve VLM's geven vaak",
        "  helemaal geen gekalibreerde confidence.",
        "* Qwen3-VL en SmolVLM2 zijn algemene VLM's, geen gespecialiseerde",
        "  temporele actieclassifiers. Meet hun labelkwaliteit en hun",
        "  start/end-nauwkeurigheid daarom apart met `action_ground_truth`.",
        "* Temporele VLM-voorspellingen zijn beperkt tot het clipvenster;",
        "  langere acties over meerdere clips worden niet automatisch",
        "  samengevoegd. Gebruik de temporele metrics als indicatie, niet als",
        "  vervanging van een gespecialiseerde temporal-action-localizer.",
        "* `mask_area` is een pixel-count op frame-resolutie en alleen",
        "  binnen segmentatie onderling vergelijkbaar.",
        "* Bekijk de debug-video's en voorbeeldframes per stap voor een",
        "  kwalitatief oordeel naast deze cijfers.",
        "",
    ])
    return "\n".join(parts)


def _task_models(runs: list[dict]) -> dict:
    return {
        "count": len(runs),
        "steps": [run["stap"] for run in runs],
        "models": [run["model"] for run in runs],
    }


def main(cfg_path: str, output: str | None = None,
         task: str = "all") -> None:
    if task not in ("all", *TASKS):
        raise SystemExit(
            f"Onbekende benchmarktaak: {task!r}. Kies uit: all, "
            f"{', '.join(TASKS)}.")
    cfg = load_config(cfg_path)
    output_dir = Path(cfg["output_dir"])
    include_detection = task in ("all", TASK_OBJECT_DETECTION)
    include_hand = task in ("all", TASK_HAND_DETECTION)
    include_vitra = task in ("all", TASK_3D_HAND_PREDICTION)
    include_action = task in ("all", TASK_ACTION_CLASSIFICATION)

    detection_paths = find_runs(output_dir) if include_detection else []
    tracking_paths = (find_tracking_runs(output_dir)
                      if include_detection else [])
    handpose_paths = find_handpose_runs(output_dir) if include_hand else []
    vitra_paths = find_vitra_runs(output_dir) if include_vitra else []
    action_paths = find_action_runs(output_dir) if include_action else []
    if not (detection_paths or tracking_paths or handpose_paths
            or vitra_paths or action_paths):
        raise SystemExit(
            f"Geen modeltesten gevonden in {output_dir} voor taak "
            f"'{task}'; draai eerst de relevante run_*_test.bat scripts.")

    tool_classes = [str(tool).strip().lower()
                    for tool in cfg.get("tool_classes", [])]
    action_classes = [str(label).strip().lower()
                      for label in cfg.get("action_classes", [])]
    weights = {**DEFAULT_WEIGHTS, **cfg.get("comparison_weights", {})}
    target_fps = float(cfg.get("comparison_target_fps", 15))
    detection_runs = [summarize_run(path, tool_classes)
                      for path in detection_paths]
    add_scores(detection_runs, weights, target_fps)
    tracking_runs = [summarize_tracking(path) for path in tracking_paths]
    handpose_runs = [summarize_handpose(path) for path in handpose_paths]
    vitra_runs = [summarize_vitra(path) for path in vitra_paths]
    ground_truth = None
    if action_paths:
        ground_truth_path = str(cfg.get("action_ground_truth") or "").strip()
        if ground_truth_path:
            ground_truth = load_action_ground_truth(
                ground_truth_path, video=cfg.get("video_path"))
    action_runs = [summarize_action_run(path, ground_truth)
                   for path in action_paths]

    report_dir = Path(output) if output else output_dir / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detection_runs).to_csv(
        report_dir / "model_comparison.csv", index=False)
    if tracking_runs:
        pd.DataFrame(tracking_runs).to_csv(
            report_dir / "tracking_comparison.csv", index=False)
    if action_runs:
        pd.DataFrame(action_runs).to_csv(
            report_dir / "action_comparison.csv", index=False)
    action_classes_table = (action_matrix(action_paths, action_classes)
                            if action_paths else None)
    report = build_report(
        detection_runs, tool_matrix(detection_paths, tool_classes), cfg,
        weights, target_fps, tracking_runs, handpose_runs, vitra_runs,
        action_runs, action_classes_table)
    (report_dir / "model_comparison.md").write_text(report, encoding="utf-8")

    overview = {
        "video": cfg.get("video_path"),
        "task_filter": task,
        "environment": system_info(),
        "tasks": {
            TASK_OBJECT_DETECTION: {
                "detection": _task_models(detection_runs),
                "tracking": _task_models(tracking_runs),
            },
            TASK_HAND_DETECTION: _task_models(handpose_runs),
            TASK_3D_HAND_PREDICTION: _task_models(vitra_runs),
            TASK_ACTION_CLASSIFICATION: _task_models(action_runs),
        },
    }
    (report_dir / "benchmark_overview.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report)
    print(f"Rapport geschreven naar {report_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.output, args.task)
