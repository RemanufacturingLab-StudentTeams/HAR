"""Tijdmeting en samenvatting per modelrun, voor de modelvergelijking.

Elke modeltest schrijft naast de resultaten-CSV een ``<naam>_summary.json``
met dezelfde velden, zodat ``compare_models.py`` de modellen naast elkaar kan
zetten (detecties, rekentijd en doorvoer).
"""
import json
import statistics
import time
from pathlib import Path


class InferenceTimer:
    """Meet de rekentijd per frame, exclusief video-inlezen en wegschrijven.

    Gebruik als context manager om precies de modelaanroep te meten::

        with timer:
            detections = model.predict(frame)
    """

    def __init__(self) -> None:
        self.times_ms: list[float] = []
        self._start = 0.0

    def __enter__(self) -> "InferenceTimer":
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
                      extra: dict | None = None) -> dict:
    """Schrijf de samenvatting van deze modelrun als JSON naast de CSV.

    ``name`` is de stapsnaam (bijvoorbeeld ``rfdetr``) en ``model`` de
    gebruikte modelnaam of checkpoint; beide komen in de vergelijking terug.
    ``class_field`` is de kolom met de klassenaam: detectiestappen gebruiken
    ``klasse``, trackingstappen ``class_name``.
    """
    times = timer.times_ms
    summary = {
        "name": name,
        "model": model,
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
        "inference_fps": (round(1000.0 / statistics.fmean(times), 2)
                          if times else 0.0),
        "detections": len(rows),
        "detections_per_frame": (round(len(rows) / frames_processed, 3)
                                 if frames_processed else 0.0),
        "klassen": sorted({str(row[class_field]) for row in rows}),
        "csv": csv_file,
    }
    if extra:
        summary.update(extra)
    (out_dir / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
