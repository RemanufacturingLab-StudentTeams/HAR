"""Vergelijkt de modeltesten in de outputmap van een video.

De vergelijking kijkt naar wat het project nodig heeft: gereedschappen
herkennen in disassemblyvideo's (zie PLAN.md fase 7). Zonder geannoteerde
ground truth meet de vergelijking geen accuracy, maar wel:

* dekking van de gereedschapsklassen uit ``tool_classes``;
* aanwezigheid van die klasse over de tijd (aandeel frames met een tool);
* aantal tool-detecties en gemiddelde confidence;
* rekentijd en doorvoer (fps) uit de ``*_summary.json`` naast elke CSV.

De rangschikking is een expliciet weegbare heuristic: dekking, aanwezigheid
en snelheid ten opzichte van ``comparison_target_fps`` (bronfps /
frame_stride). Pas de gewichten aan in de config en lees het rapport met de
kanttekeningen onderaan.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from duplo.config import load_config

CSV_SUFFIX = "_detections.csv"
DEFAULT_WEIGHTS = {"coverage": 0.4, "presence": 0.3, "speed": 0.3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output",
                        help="map voor het rapport; standaard <output_dir>/comparison")
    # Bat-level flag (run_common.bat): zie duplo.cli.parse_args.
    parser.add_argument("--upgrade", nargs="?", const=True, default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


TRACKING_SUFFIX = "_tracking_history.csv"
HANDPOSE_CSV = "hand_landmarks.csv"


def find_runs(output_dir: Path) -> list[Path]:
    """Alle detectie-CSV's van de modeltesten, met summary ernaast.

    Tracking (`*_tracking_history.csv`) en handpose (`hand_landmarks.csv`)
    worden apart gerapporteerd en niet meegewogen in de detectie-score.
    """
    return sorted(output_dir.glob(f"*/*{CSV_SUFFIX}"))


def find_tracking_runs(output_dir: Path) -> list[Path]:
    """Alle ByteTrack-CSV's (`*_history.csv`, bv. `tracking_history.csv`)."""
    seen: dict[str, Path] = {}
    for pattern in ("*/*_history.csv", f"*/*{TRACKING_SUFFIX}"):
        for path in sorted(output_dir.glob(pattern)):
            seen[str(path)] = path
    return sorted(seen.values())


def load_summary(csv_path: Path) -> dict:
    name = csv_path.name
    candidates: list[Path] = []
    if name.endswith(CSV_SUFFIX):
        candidates.append(csv_path.with_name(
            name[: -len(CSV_SUFFIX)] + "_summary.json"))
    if name.endswith(TRACKING_SUFFIX):
        candidates.append(csv_path.with_name(
            name[: -len(TRACKING_SUFFIX)] + "_summary.json"))
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
    """Bereken de vergelijkingscijfers van één modeltest."""
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
    run = {
        "stap": csv_path.parent.name,
        "model": summary.get("model", "onbekend (geen summary gevonden)"),
        "frame_stride": summary.get("frame_stride"),
        "frames_processed": frames_processed,
        "detections": int(len(data)),
        "detections_per_frame": round(len(data) / frames_processed, 3)
        if frames_processed else 0.0,
        "tool_classes_found": len(found),
        "tool_coverage": round(len(found) / len(tool_classes), 3)
        if tool_classes else 0.0,
        "tool_detections": int(len(tool_rows)),
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
    return run


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
        "model": summary.get("model", "onbekend (geen summary gevonden)"),
        "frames_processed": frames,
        "tracks": tracks,
        "rows": int(len(data)),
        "rows_per_frame": round(len(data) / frames, 3) if frames else 0.0,
        "status_confirmed": int(statuses.get("confirmed", 0)),
        "status_tentative": int(statuses.get("tentative", 0)),
        "status_occluded": int(statuses.get("occluded", 0)),
        "inference_ms_median": summary.get("inference_ms_median"),
        "pipeline_fps": summary.get("pipeline_fps"),
    }


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def build_report(runs: list[dict], matrix: pd.DataFrame, cfg: dict,
                 weights: dict, target_fps: float,
                 tracking_runs: list[dict] | None = None) -> str:
    header = ["Model", "Modelnaam", "Toolklassen (gevonden/doel)",
              "Tool-detecties", "Tool-frames", "Gem. conf.", "Det./frame",
              "Inferentie (ms)", "Doorvoer (fps)", "Score"]
    rows = []
    for run in runs:
        rows.append([
            run["stap"], run["model"],
            f"{run['tool_classes_found']}/{len(cfg['tool_classes'])}",
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
    matrix_rows = [[tool, *[int(matrix.loc[tool, column]) for column in matrix.columns]]
                   for tool in matrix.index]
    top_rows = [[run["stap"], run["top_classes"]] for run in runs]
    parts = [
        "# Modelvergelijking toolrecognition",
        "",
        f"Video: `{cfg['video_path']}` · frame_stride "
        f"`{cfg.get('frame_stride', 1)}` · doelgereedschappen: "
        f"{', '.join(cfg['tool_classes'])}",
        "",
        "## Scores",
        "",
        markdown_table(header, rows),
        "",
        "Score = 100 x ("
        f"{weights['coverage']} x dekking + {weights['presence']} x tool-frames "
        f"+ {weights['speed']} x min(1, doorvoer/{target_fps:g}))",
        "",
        "## Detecties per gereedschapsklasse",
        "",
        markdown_table(["Gereedschap", *matrix.columns], matrix_rows),
        "",
        "## Meest gedetecteerde klassen per model",
        "",
        markdown_table(["Model", "Top-3 klassen"], top_rows),
        "",
    ]
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
            "Tracking wordt niet meegewogen in de score hierboven; "
            "`confirmed/tentative/occluded` tellen rijen per status.",
            "",
        ])
    parts.extend([
        "## Kanttekeningen",
        "",
        "* Er is geen geannoteerde ground truth, dus accuracy (precision/recall)",
        "  is niet te meten; dekking en tool-frames zijn proxies. Annotatie van",
        "  een steekproef frames is nodig voor een echte kwaliteitsscore.",
        "* Open-vocabulary modellen (YOLO-World, Grounding DINO, SAM3, SAM 3.1) krijgen de",
        "  doelklassen als prompt en kunnen die dus per definitie vinden;",
        "  closed-vocabulary modellen (YOLO, RF-DETR) zijn beperkt tot hun",
        "  80 COCO-klassen en hebben daar maar enkele gereedschappen in.",
        "* Confidence is niet tussen modellen te vergelijken: elke familie",
        "  kalibreert scores anders.",
        "* `mask_area` is een pixel-count op frame-resolutie (YOLO-seg via",
        "  gerasteriseerd polygoon, RF-DETR-seg/SAM via geresized masker);",
        "  vergelijk alleen binnen segmentatie onderling.",
        "* Klassen met enkele detecties kunnen toevallige treffers of false",
        "  positives zijn; weeg de aantallen per klasse mee in het oordeel.",
        "* Bekijk de debug-video's en voorbeeldframes per stap voor een",
        "  kwalitatief oordeel naast deze cijfers.",
        "",
    ])
    return "\n".join(parts)


def main(cfg_path: str, output: str | None = None) -> None:
    cfg = load_config(cfg_path)
    output_dir = Path(cfg["output_dir"])
    csv_paths = find_runs(output_dir)
    if not csv_paths:
        raise SystemExit(
            f"Geen modeltesten gevonden in {output_dir}; draai eerst de "
            "run_*_test.bat scripts (of de src/run_*_video.py scripts).")

    tool_classes = [str(tool).strip().lower() for tool in cfg["tool_classes"]]
    weights = {**DEFAULT_WEIGHTS, **cfg.get("comparison_weights", {})}
    target_fps = float(cfg.get("comparison_target_fps", 15))
    runs = [summarize_run(csv_path, tool_classes) for csv_path in csv_paths]
    add_scores(runs, weights, target_fps)
    tracking_paths = find_tracking_runs(output_dir)
    tracking_runs = [summarize_tracking(csv_path) for csv_path in tracking_paths]

    report_dir = Path(output) if output else output_dir / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(runs)
    table.to_csv(report_dir / "model_comparison.csv", index=False)
    if tracking_runs:
        pd.DataFrame(tracking_runs).to_csv(
            report_dir / "tracking_comparison.csv", index=False)
    report = build_report(runs, tool_matrix(csv_paths, tool_classes), cfg,
                          weights, target_fps, tracking_runs)
    (report_dir / "model_comparison.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Rapport geschreven naar {report_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.output)
