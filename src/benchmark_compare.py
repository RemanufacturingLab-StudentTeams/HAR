"""Benchmark geselecteerde modellen op één video of over meerdere video's.

Twee gebruikssituaties staan centraal:

* meerdere modellen op één video vergelijken (bijvoorbeeld ActionCLIP vs
  InternVideo2 vs X-CLIP, of Grounding DINO vs SAM 3.1 vs YOLO-World);
* één model op meerdere video's draaien en de metrics middelen.

De orchestrator maakt per model/video een tijdelijke config, start het
bestaande ``run_*`` script, en genereert automatisch:

* een taakbewust per-video rapport in ``<video>/comparison/``;
* een geaggregeerd ``benchmark_report.md``;
* ``benchmark_metrics.csv`` en ``benchmark_average.csv``;
* ``benchmark_overview.json`` met selectie, omgeving en runstatus.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

import compare_models
from duplo.action_eval import load_action_ground_truth
from duplo.benchmark import (
    TASK_3D_HAND_PREDICTION,
    TASK_ACTION_CLASSIFICATION,
    TASK_HAND_DETECTION,
    TASK_LABELS,
    TASK_OBJECT_DETECTION,
    system_info,
)
from duplo.config import load_config, resolve_video_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "benchmarks"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: str
    script: str
    step: str
    csv_name: str
    extra_args: tuple[str, ...] = ()

    @property
    def task(self) -> str:
        return KIND_TASKS[self.kind]


KIND_TASKS = {
    "action": TASK_ACTION_CLASSIFICATION,
    "detection": TASK_OBJECT_DETECTION,
    "tracking": TASK_OBJECT_DETECTION,
    "handpose": TASK_HAND_DETECTION,
    "vitra": TASK_3D_HAND_PREDICTION,
}

MODEL_SPECS: dict[str, ModelSpec] = {
    # Actieclassificatie
    "xclip": ModelSpec("xclip", "action", "src/run_action_test.py", "xclip",
                       "xclip_actions.csv", ("--model", "xclip")),
    "internvideo2": ModelSpec(
        "internvideo2", "action", "src/run_action_test.py", "internvideo2",
        "internvideo2_actions.csv", ("--model", "internvideo2")),
    "actionclip": ModelSpec(
        "actionclip", "action", "src/run_action_test.py", "actionclip",
        "actionclip_actions.csv", ("--model", "actionclip")),
    "smolvlm2": ModelSpec(
        "smolvlm2", "action", "src/run_action_test.py", "smolvlm2",
        "smolvlm2_actions.csv", ("--model", "smolvlm2")),
    "internvideo25": ModelSpec(
        "internvideo25", "action", "src/run_action_test.py",
        "internvideo25", "internvideo25_actions.csv",
        ("--model", "internvideo25")),
    "internvideo3": ModelSpec(
        "internvideo3", "action", "src/run_action_test.py", "internvideo3",
        "internvideo3_actions.csv", ("--model", "internvideo3")),
    # Objectdetectie / segmentatie
    "yolo": ModelSpec("yolo", "detection", "src/run_yolo_video.py", "yolo",
                      "yolo_detections.csv"),
    "rfdetr": ModelSpec("rfdetr", "detection", "src/run_rfdetr_video.py",
                        "rfdetr", "rfdetr_detections.csv"),
    "grounding_dino": ModelSpec(
        "grounding_dino", "detection", "src/run_grounding_dino_video.py",
        "grounding_dino", "grounding_dino_detections.csv"),
    "yolo_world": ModelSpec(
        "yolo_world", "detection", "src/run_yolo_world_video.py",
        "yolo_world", "yolo_world_detections.csv"),
    "yolo_seg": ModelSpec("yolo_seg", "detection",
                          "src/run_yolo_seg_video.py", "yolo_seg",
                          "yolo_seg_detections.csv"),
    "rfdetr_seg": ModelSpec(
        "rfdetr_seg", "detection", "src/run_rfdetr_seg_video.py",
        "rfdetr_seg", "rfdetr_seg_detections.csv"),
    "sam3": ModelSpec("sam3", "detection", "src/run_sam3_video.py", "sam3",
                      "sam3_detections.csv"),
    "sam31": ModelSpec("sam31", "detection", "src/run_sam31_video.py",
                       "sam31", "sam31_detections.csv"),
    # Tracking
    "tracking": ModelSpec("tracking", "tracking", "src/run_tracking.py",
                          "tracking", "tracking_history.csv"),
    "yolo_world_tracking": ModelSpec(
        "yolo_world_tracking", "tracking",
        "src/run_yolo_world_tracking.py", "yolo_world_tracking",
        "yolo_world_tracking_history.csv"),
    "grounding_dino_tracking": ModelSpec(
        "grounding_dino_tracking", "tracking",
        "src/run_grounding_dino_tracking.py", "grounding_dino_tracking",
        "grounding_dino_tracking_history.csv"),
    "sam3_tracking": ModelSpec(
        "sam3_tracking", "tracking", "src/run_sam3_tracking.py",
        "sam3_tracking", "sam3_tracking_history.csv"),
    "sam31_tracking": ModelSpec(
        "sam31_tracking", "tracking", "src/run_sam31_tracking.py",
        "sam31_tracking", "sam31_tracking_history.csv"),
    # Overige taken
    "handpose": ModelSpec("handpose", "handpose", "src/run_handpose.py",
                          "handpose", "hand_landmarks.csv"),
    "vitra": ModelSpec("vitra", "vitra", "src/run_vitra_test.py", "vitra",
                       "vitra_actions.csv"),
}

MODEL_ALIASES = {
    "x-clip": "xclip",
    "x_clip": "xclip",
    "intern-video2": "internvideo2",
    "intern_video2": "internvideo2",
    "action-clip": "actionclip",
    "action_clip": "actionclip",
    "smol-vlm2": "smolvlm2",
    "smol_vlm2": "smolvlm2",
    "internvideo2.5": "internvideo25",
    "internvideo2_5": "internvideo25",
    "intern-video2.5": "internvideo25",
    "intern-video-2.5": "internvideo25",
    "internvideo-2.5": "internvideo25",
    "internvideo3-8b": "internvideo3",
    "internvideo3-8b-instruct": "internvideo3",
    "intervideo3": "internvideo3",
    "intervideo3-8b": "internvideo3",
    "grounding-dino": "grounding_dino",
    "groundingdino": "grounding_dino",
    "yolo-world": "yolo_world",
    "yoloworld": "yolo_world",
    "sam3.1": "sam31",
    "sam-3.1": "sam31",
    "sam_3.1": "sam31",
    "sam_3_1": "sam31",
}

PRESETS = {
    "action": ("actionclip", "internvideo2", "xclip"),
    "actions": ("actionclip", "internvideo2", "xclip"),
    "vlm": ("smolvlm2", "internvideo25", "internvideo3"),
    "generative": ("smolvlm2", "internvideo25", "internvideo3"),
    "detection": ("grounding_dino", "sam31", "yolo_world"),
    "open_vocabulary": ("grounding_dino", "sam31", "yolo_world"),
    "open-vocabulary": ("grounding_dino", "sam31", "yolo_world"),
    "tracking": ("tracking", "grounding_dino_tracking", "sam31_tracking"),
}


@dataclass
class RunRecord:
    model: str
    video: str
    kind: str
    task: str
    status: str
    returncode: int | None = None
    log: str | None = None
    metrics: dict | None = None
    error: str | None = None


def normalize_model_name(name: str) -> str:
    key = str(name).strip().lower()
    return MODEL_ALIASES.get(key, key)


def select_models(preset: str | None, values: list[str] | None = None,
                  single: str | None = None) -> list[str]:
    """Combineer preset en expliciete modelnamen, zonder duplicaten."""
    selected: list[str] = []
    if preset:
        key = str(preset).strip().lower()
        if key not in PRESETS:
            raise SystemExit(
                f"Onbekende preset: {preset!r}. Kies uit: "
                f"{', '.join(sorted(PRESETS))}.")
        selected.extend(PRESETS[key])
    raw = list(values or [])
    if single:
        raw.append(single)
    for value in raw:
        text = str(value).strip()
        compact = re.sub(r"\s+", "_", text.lower())
        if compact in MODEL_SPECS or compact in MODEL_ALIASES:
            selected.append(normalize_model_name(compact))
            continue
        for part in re.split(r"[,\s]+", text):
            if part:
                selected.append(normalize_model_name(part))
    if not selected:
        selected = list(PRESETS["action"])
    unknown = [key for key in selected if key not in MODEL_SPECS]
    if unknown:
        raise SystemExit(
            f"Onbekende model(len): {', '.join(unknown)}. Kies uit: "
            f"{', '.join(sorted(MODEL_SPECS))}.")
    unique: list[str] = []
    for key in selected:
        if key not in unique:
            unique.append(key)
    return unique


def select_videos(explicit: list[str] | None, cfg: dict) -> list[Path]:
    """Gebruik opgegeven video('s), of de video uit de config."""
    raw = [value for value in (explicit or []) if value]
    if not raw:
        resolve_video_path(cfg)
        raw = [str(cfg["video_path"])]
    videos: list[Path] = []
    for value in raw:
        text = str(value)
        if any(char in text for char in "*?[]"):
            pattern = Path(text)
            if not pattern.is_absolute():
                pattern = PROJECT_ROOT / pattern
            matches = sorted(Path(found) for found in glob.glob(
                str(pattern), recursive=True))
            if not matches:
                raise SystemExit(f"Geen video's gevonden voor patroon: {text}")
            videos.extend(matches)
            continue
        path = Path(text)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise SystemExit(f"Video bestaat niet: {text}")
        videos.append(path.resolve())
    unique: list[Path] = []
    for video in videos:
        if video not in unique:
            unique.append(video)
    return unique


def video_slug(video: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video.stem).strip("_") or "video"
    return f"{index:02d}_{stem}"


def benchmark_root(name: str, output: str | None = None) -> Path:
    if output:
        root = Path(output)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return root
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return DEFAULT_OUTPUT_ROOT / (safe_name or "benchmark")


def prepare_config(base_cfg: dict, video: Path, video_dir: Path,
                   config_path: Path, device: str | None,
                   save_videos: bool) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["video_path"] = str(video)
    cfg["output_dir"] = str(video_dir)
    if device:
        cfg["device"] = device
    if not save_videos:
        for key in list(cfg):
            if key.startswith("save_"):
                cfg[key] = False
        cfg["screenshot_every"] = 0
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    return cfg


def _run_streaming(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("  $ " + " ".join(f'"{part}"' if " " in part else part
                            for part in command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def extract_metrics(spec: ModelSpec, csv_path: Path, cfg: dict,
                    ground_truth: list | None) -> dict:
    if spec.kind == "action":
        return compare_models.summarize_action_run(csv_path, ground_truth)
    if spec.kind == "tracking":
        return compare_models.summarize_tracking(csv_path)
    if spec.kind == "handpose":
        return compare_models.summarize_handpose(csv_path)
    if spec.kind == "vitra":
        return compare_models.summarize_vitra(csv_path)
    tool_classes = [str(tool).strip().lower()
                    for tool in cfg.get("tool_classes", [])]
    run = compare_models.summarize_run(csv_path, tool_classes)
    weights = {**compare_models.DEFAULT_WEIGHTS,
               **cfg.get("comparison_weights", {})}
    compare_models.add_scores(
        [run], weights, float(cfg.get("comparison_target_fps", 15)))
    return run


def run_one(spec: ModelSpec, video: Path, video_dir: Path, cfg_path: Path,
            base_cfg: dict, config_dir: Path, log_dir: Path,
            device: str | None, save_videos: bool, reuse: bool,
            ground_truth: list | None) -> RunRecord:
    csv_path = video_dir / spec.step / spec.csv_name
    if reuse and csv_path.is_file():
        try:
            metrics = extract_metrics(spec, csv_path, base_cfg, ground_truth)
            return RunRecord(spec.key, str(video), spec.kind, spec.task,
                             "reused", metrics=metrics)
        except Exception as exc:  # noqa: BLE001 - rapport blijft bruikbaar
            return RunRecord(spec.key, str(video), spec.kind, spec.task,
                             "failed", error=f"metrics: {exc}")
    cfg = prepare_config(base_cfg, video, video_dir, cfg_path, device,
                         save_videos)
    command = [sys.executable, "-u", spec.script, "--config", str(cfg_path),
               *spec.extra_args]
    if device:
        command.extend(["--device", device])
    log_path = log_dir / f"{spec.key}_{video_dir.name}.log"
    try:
        returncode = _run_streaming(command, log_path)
    except OSError as exc:
        return RunRecord(spec.key, str(video), spec.kind, spec.task,
                         "failed", log=str(log_path), error=str(exc))
    if returncode != 0:
        return RunRecord(spec.key, str(video), spec.kind, spec.task,
                         "failed", returncode=returncode, log=str(log_path),
                         error=f"exitcode {returncode}")
    if not csv_path.is_file():
        return RunRecord(spec.key, str(video), spec.kind, spec.task,
                         "failed", returncode=returncode, log=str(log_path),
                         error=f"CSV ontbreekt: {csv_path}")
    try:
        metrics = extract_metrics(spec, csv_path, cfg, ground_truth)
    except Exception as exc:  # noqa: BLE001 - run zelf is wel gelukt
        return RunRecord(spec.key, str(video), spec.kind, spec.task,
                         "failed", returncode=returncode, log=str(log_path),
                         error=f"metrics: {exc}")
    return RunRecord(spec.key, str(video), spec.kind, spec.task, "ok",
                     returncode=returncode, log=str(log_path), metrics=metrics)


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def average_records(records: list[RunRecord]) -> list[dict]:
    """Gemiddelde metrics per model over alle geslaagde videos."""
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.model, record.kind), []).append(record)
    averages = []
    for (model, kind), group in grouped.items():
        successful = [record for record in group
                      if record.status in ("ok", "reused")
                      and record.metrics]
        model_names = [record.metrics.get("model")
                       for record in successful
                       if record.metrics and record.metrics.get("model")]
        averages.append({
            "model": model,
            "model_name": model_names[0] if model_names else model,
            "kind": kind,
            "task": KIND_TASKS[kind],
            "videos": len(group),
            "successful": len(successful),
            "failed": len(group) - len(successful),
            "metrics": _average_metrics([record.metrics
                                         for record in successful]),
        })
    return averages


def _average_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    values: dict[str, list] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key in ("evaluation", "stap", "video", "model"):
                continue
            if _numeric(value):
                values.setdefault(key, []).append(float(value))
            elif key in ("top_label", "hands"):
                values.setdefault(key, []).append(value)
    averaged: dict = {}
    for key, items in values.items():
        if all(_numeric(item) for item in items):
            averaged[key] = round(sum(items) / len(items), 3)
        else:
            counts: dict[str, int] = {}
            for item in items:
                counts[str(item)] = counts.get(str(item), 0) + 1
            averaged[key] = max(counts, key=counts.get)
    averaged["videos"] = len(metrics_list)
    return averaged


REPORT_COLUMNS = {
    "action": [
        ("clips", "Clips"), ("labels_found", "Labels"),
        ("top_label", "Top-label"), ("top_label_share", "Aandeel"),
        ("confidence_mean", "Gem. conf."),
        ("classification_accuracy", "Class. acc."),
        ("temporal_coverage", "Temp. cov."),
        ("temporal_iou_mean", "mIoU"), ("temporal_recall_0.5", "R@0.5"),
        ("inference_ms_median", "Inf. ms"), ("pipeline_fps", "Clips/s"),
    ],
    "detection": [
        ("tool_classes_found", "Tools"), ("tool_coverage", "Dekking"),
        ("tool_detections", "Detecties"), ("tool_presence", "Tool-frames"),
        ("tool_confidence_mean", "Gem. conf."),
        ("inference_ms_median", "Inf. ms"), ("pipeline_fps", "FPS"),
        ("score", "Score"),
    ],
    "tracking": [
        ("tracks", "Tracks"), ("rows_per_frame", "Rijen/frame"),
        ("status_confirmed", "Confirmed"), ("status_tentative", "Tentative"),
        ("status_occluded", "Occluded"),
        ("inference_ms_median", "Inf. ms"), ("pipeline_fps", "FPS"),
    ],
    "handpose": [
        ("frames_processed", "Frames"), ("landmark_rows", "Landmarks"),
        ("frames_with_hands", "Frames met handen"), ("hands", "Handen"),
        ("inference_ms_median", "Inf. ms"), ("pipeline_fps", "FPS"),
    ],
    "vitra": [
        ("frames_processed", "Frames"), ("action_rows", "Actierijen"),
        ("hands", "Handen"), ("chunk_steps", "Chunk"),
        ("samples", "Samples"), ("inference_ms_median", "Inf. ms"),
        ("pipeline_fps", "FPS"),
    ],
}

RATIO_KEYS = {"top_label_share", "classification_accuracy",
              "temporal_coverage", "temporal_recall_0.5", "tool_coverage",
              "tool_presence"}


def _format_value(key: str, value) -> str:
    if value is None or value == "":
        return "n.v.t."
    if key in RATIO_KEYS:
        return f"{100 * float(value):.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def _table(kind: str, rows: list[dict], include_video: bool = False,
           include_count: bool = False) -> str:
    columns = REPORT_COLUMNS[kind]
    header = ["Model"]
    if include_video:
        header.append("Video")
    if include_count:
        header.append("Video's")
    header.extend(label for _, label in columns)
    body = []
    for row in rows:
        line = [row.get("model", "?")]
        if include_video:
            line.append(Path(row.get("video", "")).name)
        if include_count:
            line.append(str(row.get("videos", "n.v.t.")))
        line.extend(_format_value(key, row.get("metrics", {}).get(key))
                    for key, _ in columns)
        body.append(line)
    return compare_models.markdown_table(header, body)


def render_report(name: str, models: list[str], videos: list[Path],
                  records: list[RunRecord], averages: list[dict],
                  device: str | None, root: Path) -> str:
    parts = [
        f"# Benchmark: {name}",
        "",
        f"- Modellen: {', '.join(models)}",
        f"- Video's: {len(videos)}",
        f"- Device: {device or 'config/auto'}",
        f"- Uitvoer: `{root}`",
        "",
        "## Samenvatting",
        "",
    ]
    kinds = []
    for model in models:
        kind = MODEL_SPECS[model].kind
        if kind not in kinds:
            kinds.append(kind)
    summary_rows = []
    for kind in kinds:
        kind_records = [record for record in records if record.kind == kind]
        kind_averages = [row for row in averages if row["kind"] == kind]
        summary_rows.append([
            kind, ", ".join(row["model"] for row in kind_averages),
            len(videos),
            sum(1 for record in kind_records
                if record.status in ("ok", "reused")),
            sum(1 for record in kind_records if record.status == "failed"),
        ])
    parts.extend([
        compare_models.markdown_table(
            ["Soort", "Modellen", "Video's", "Geslaagd", "Mislukt"],
            summary_rows),
        "",
    ])

    for kind in kinds:
        kind_records = [record for record in records if record.kind == kind]
        if not kind_records:
            continue
        label = {
            "action": TASK_ACTION_CLASSIFICATION,
            "detection": TASK_OBJECT_DETECTION,
            "tracking": TASK_OBJECT_DETECTION,
            "handpose": TASK_HAND_DETECTION,
            "vitra": TASK_3D_HAND_PREDICTION,
        }[kind]
        parts.extend([f"## {TASK_LABELS[label]}", ""])
        for video in videos:
            video_records = [record for record in kind_records
                             if record.video == str(video)]
            if not video_records:
                continue
            rows = [{"model": record.model, "video": record.video,
                     "metrics": record.metrics or {}}
                    for record in video_records]
            parts.extend([
                f"### {video.name}",
                "",
                _table(kind, rows, include_video=True),
                "",
            ])
        kind_averages = [row for row in averages if row["kind"] == kind]
        parts.extend([
            "### Gemiddelde over video's",
            "",
            _table(kind, [{"model": row["model"],
                           "videos": row["videos"],
                           "metrics": row["metrics"]}
                          for row in kind_averages], include_count=True),
            "",
        ])

    failed = [record for record in records if record.status == "failed"]
    if failed:
        parts.extend([
            "## Mislukte runs",
            "",
            compare_models.markdown_table(
                ["Model", "Video", "Fout", "Log"],
                [[record.model, Path(record.video).name,
                  record.error or "onbekend", record.log or ""]
                 for record in failed]),
            "",
        ])

    parts.extend([
        "## Detailrapporten per video",
        "",
    ])
    for index, video in enumerate(videos, 1):
        report = root / video_slug(video, index) / "comparison" / (
            "model_comparison.md")
        if report.is_file():
            parts.append(f"- [{video.name}]({report.relative_to(root).as_posix()})")
    parts.extend([
        "",
        "## Kanttekeningen",
        "",
        "* Dit rapport draait alleen de geselecteerde modellen; het is geen",
        "  volledige modelvergelijking.",
        "* De gemiddelde tabel middelt numerieke metrics over de geslaagde",
        "  video's; aantallen clips/detecties zijn dus gemiddelden per video.",
        "* Confidence en snelheid zijn niet tussen modelfamilies te",
        "  vergelijken zonder de kanttekeningen in `Documentation/Benchmarking.md`.",
        "",
    ])
    return "\n".join(parts)


def write_reports(name: str, models: list[str], videos: list[Path],
                  records: list[RunRecord], averages: list[dict],
                  device: str | None, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for record in records:
        row = {
            "model": record.model,
            "kind": record.kind,
            "task": record.task,
            "video": record.video,
            "status": record.status,
            "error": record.error,
        }
        metrics = record.metrics or {}
        row["model_name"] = metrics.get("model")
        for key, value in metrics.items():
            if key in ("evaluation", "model", "stap", "task", "video"):
                continue
            row[key] = value
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(root / "benchmark_metrics.csv",
                                     index=False)
    average_rows = []
    for row in averages:
        average_rows.append({
            "model": row["model"],
            "model_name": row.get("model_name"),
            "kind": row["kind"],
            "task": row["task"],
            "videos": row["videos"],
            "successful": row["successful"],
            "failed": row["failed"],
            **row["metrics"],
        })
    pd.DataFrame(average_rows).to_csv(root / "benchmark_average.csv",
                                      index=False)
    (root / "benchmark_report.md").write_text(
        render_report(name, models, videos, records, averages, device, root),
        encoding="utf-8")
    overview = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "models": models,
        "videos": [str(video) for video in videos],
        "environment": system_info(),
        "runs": [
            {"model": record.model, "video": record.video,
             "kind": record.kind, "status": record.status,
             "returncode": record.returncode, "log": record.log,
             "error": record.error}
            for record in records
        ],
        "averages": averages,
    }
    (root / "benchmark_overview.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_video_report(cfg_path: Path, video_dir: Path) -> None:
    """Genereer het bestaande taakbewuste rapport voor één video."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            compare_models.main(str(cfg_path),
                                output=str(video_dir / "comparison"))
    except SystemExit as exc:
        print(f"  Waarschuwing: per-video rapport overgeslagen ({exc}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--preset", help="action, detection of tracking")
    parser.add_argument("--models", nargs="+",
                        help="modelnamen, kommagescheiden of met spaties")
    parser.add_argument("--model", help="één model voor meerdere video's")
    parser.add_argument("--video", action="append", default=[],
                        help="video (herhaalbaar)")
    parser.add_argument("--videos", nargs="+", default=[],
                        help="meerdere video's of een glob-patroon")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--output", help="hoofdmap voor benchmarkresultaten")
    parser.add_argument("--name", help="naam van deze benchmark")
    parser.add_argument("--reuse", action="store_true",
                        help="bestaande model-CSV's hergebruiken")
    parser.add_argument("--save-videos", action="store_true",
                        help="debug-video's en voorbeeldframes bewaren")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop bij de eerste mislukte modelrun")
    parser.add_argument("--dry-run", action="store_true",
                        help="toon alleen de commando's")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--upgrade", nargs="?", const=True, default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_models:
        for key, spec in sorted(MODEL_SPECS.items()):
            print(f"{key:24s} {spec.kind:10s} {spec.script}")
        return 0
    if args.list_presets:
        for key, value in sorted(PRESETS.items()):
            print(f"{key:24s} {', '.join(value)}")
        return 0

    cfg = load_config(args.config)
    models = select_models(args.preset, args.models, args.model)
    videos = select_videos([*args.videos, *args.video], cfg)
    name = args.name or args.preset or "_".join(models[:3])
    root = benchmark_root(name, args.output)
    print(f"Benchmark '{name}': {len(models)} model(len) x "
          f"{len(videos)} video('s) -> {root}")
    print(f"Modellen: {', '.join(models)}")
    print(f"Video's: {', '.join(video.name for video in videos)}")

    ground_truth = None
    gt_path = str(cfg.get("action_ground_truth") or "").strip()
    if gt_path and any(MODEL_SPECS[model].kind == "action" for model in models):
        ground_truth = load_action_ground_truth(gt_path)
    config_dir = root / "configs"
    log_dir = root / "logs"
    records: list[RunRecord] = []
    video_configs: dict[str, Path] = {}
    stop = False
    for index, video in enumerate(videos, 1):
        slug = video_slug(video, index)
        video_dir = root / slug
        for model in models:
            spec = MODEL_SPECS[model]
            cfg_path = config_dir / f"{spec.key}_{slug}.yaml"
            video_configs[slug] = cfg_path
            if args.dry_run:
                print(f"[dry-run] {spec.key} op {video.name}")
                continue
            print(f"\n[{model} x {video.name}] {spec.script}")
            record = run_one(
                spec, video, video_dir, cfg_path, cfg, config_dir, log_dir,
                args.device, args.save_videos, args.reuse, ground_truth)
            records.append(record)
            if record.status in ("ok", "reused"):
                print(f"  {model}: {record.status}")
            else:
                print(f"  {model}: MISLUKT ({record.error})")
                if args.fail_fast:
                    stop = True
                    break
        if not args.dry_run and video_configs.get(slug, Path()).is_file():
            generate_video_report(video_configs[slug], video_dir)
        if stop:
            break

    if args.dry_run:
        return 0
    averages = average_records(records)
    write_reports(name, models, videos, records, averages, args.device, root)
    print(f"\nRapport: {root / 'benchmark_report.md'}")
    print(f"Gemiddelden: {root / 'benchmark_average.csv'}")
    failures = [record for record in records if record.status == "failed"]
    if failures:
        print(f"Waarschuwing: {len(failures)} run(s) mislukt; zie het rapport.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
