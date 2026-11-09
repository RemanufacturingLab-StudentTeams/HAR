"""Tijdmeting, taakcategorieën en samenvattingen per modelrun.

Elke modeltest schrijft naast de resultaten-CSV een ``<naam>_summary.json``
met dezelfde basisevelden. Sinds de uitbreiding met handdetectie,
3D-handvoorspelling en actieclassificatie bevat elke summary ook een
``task``-categorie, zodat ``compare_models.py`` de juiste metric per taak kan
rapporteren in plaats van alle modellen als detectoren te vergelijken.
"""
import functools
import json
import os
import platform
import statistics
import time
from pathlib import Path

TASK_OBJECT_DETECTION = "object_detection"
TASK_HAND_DETECTION = "hand_detection"
TASK_3D_HAND_PREDICTION = "3d_hand_prediction"
TASK_ACTION_CLASSIFICATION = "action_classification"

TASKS = (TASK_OBJECT_DETECTION, TASK_HAND_DETECTION,
         TASK_3D_HAND_PREDICTION, TASK_ACTION_CLASSIFICATION)

TASK_LABELS = {
    TASK_OBJECT_DETECTION: "Objectdetectie / segmentatie / tracking",
    TASK_HAND_DETECTION: "Handdetectie (landmarks)",
    TASK_3D_HAND_PREDICTION: "3D-handvoorspelling (VITRA)",
    TASK_ACTION_CLASSIFICATION: "Actieclassificatie",
}

TASK_METRICS = {
    TASK_OBJECT_DETECTION: "toolklassen, tool-frames, confidence, fps",
    TASK_HAND_DETECTION: "handen, landmarkrijen, fps",
    TASK_3D_HAND_PREDICTION: "actierijen, handen, fps",
    TASK_ACTION_CLASSIFICATION: "clips, labels, confidence, temporele IoU",
}


@functools.lru_cache(maxsize=1)
def system_info() -> dict:
    """Hardware/softwaremetadata voor reproduceerbare benchmarkresultaten."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": None,
        "cuda": False,
        "gpu_count": 0,
        "gpus": [],
        "cuda_version": None,
        "cudnn_version": None,
    }
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch is optioneel voor pure CSV-tools
        return info
    info["torch"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = [torch.cuda.get_device_name(index)
                            for index in range(info["gpu_count"])]
    except Exception:  # noqa: BLE001, S110 - defecte CUDA-driver
        pass
    return info


class InferenceTimer:
    """Meet de rekentijd per frame/clip, exclusief video-inlezen en schrijven.

    Gebruik als context manager om precies de modelaanroep te meten::

        with timer:
            detections = model.predict(frame)
    """

    def __init__(self) -> None:
        self.times_ms: list[float] = []
        self._start = 0.0

    def __enter__(self) -> "InferenceTimer":  # noqa: PYI034
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.times_ms.append((time.perf_counter() - self._start) * 1000.0)


def percentile(values: list[float], percent: float) -> float:
    """Percentiel van een lijst waarden (zonder interpolatie)."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * percent / 100))
    return ordered[index]


def write_run_summary(out_dir: Path, name: str, *, model: str, device: str,
                      video: str, frames_read: int, frames_processed: int,
                      elapsed_s: float, rows: list[dict],
                      timer: InferenceTimer, csv_file: str,
                      class_field: str = "klasse",
                      task: str = TASK_OBJECT_DETECTION,
                      subtask: str | None = None,
                      extra: dict | None = None) -> dict:
    """Schrijf de samenvatting van deze modelrun als JSON naast de CSV.

    ``name`` is de stapsnaam (bijvoorbeeld ``rfdetr``) en ``model`` de
    gebruikte modelnaam of checkpoint; beide komen in de vergelijking terug.
    ``class_field`` is de kolom met de klassenaam: detectiestappen gebruiken
    ``klasse``, trackingstappen ``class_name``.
    """
    if task not in TASKS:
        raise SystemExit(
            f"Onbekende benchmarktaak: {task!r}. Kies uit: "
            f"{', '.join(TASKS)}.")
    times = timer.times_ms
    summary = {
        "benchmark_version": 2,
        "name": name,
        "model": model,
        "task": task,
        "task_label": TASK_LABELS[task],
        "subtask": subtask,
        "device": device,
        "video": video,
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "elapsed_s": round(elapsed_s, 1),
        "pipeline_fps": (round(frames_processed / elapsed_s, 2)
                         if elapsed_s > 0 else 0.0),
        "inference_ms_mean": round(statistics.fmean(times), 2) if times else 0.0,
        "inference_ms_median": round(statistics.median(times), 2) if times else 0.0,
        "inference_ms_p95": round(percentile(times, 95), 2) if times else 0.0,
        "inference_ms_min": round(min(times), 2) if times else 0.0,
        "inference_ms_max": round(max(times), 2) if times else 0.0,
        "inference_fps": (round(1000.0 / statistics.fmean(times), 2)
                          if times else 0.0),
        "samples": len(times),
        "detections": len(rows),
        "detections_per_frame": (round(len(rows) / frames_processed, 3)
                                 if frames_processed else 0.0),
        "klassen": sorted({str(row[class_field]) for row in rows
                           if class_field in row}),
        "csv": csv_file,
        "environment": system_info(),
    }
    if extra:
        summary.update(extra)
    (out_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
