"""Gedeelde runtime-orchestratie voor alle `run_*` scripts.

Centraliseert de ~25 regels boilerplate die in elk script stonden:
device-resolutie + CUDA-setup, fps/stride/screenshot-config, debug-video
writer, `VideoFrames`-loop-skelet, CSV + `*_summary.json` afsluiting.

Detectie-neutraal: drempels, imgsz, gewichten en annotatie-keuzes blijven in
de caller; alleen gedeelde orchestratie staat hier.
"""
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.config import load_config, resolve_device
from duplo.gpu import ensure_cuda_torch, setup_cuda_optimizations
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter


def setup_runtime(cfg: dict, cli_device: str | None = None) -> str:
    """Resolve device + CUDA-setup. Retourneert effectieve `device`."""
    device = resolve_device(cli_device, cfg)
    if device == "gpu":
        ensure_cuda_torch(
            auto_install=bool(cfg.get("auto_install_torch", False)))
    setup_cuda_optimizations(device)
    return device


def load_step_config(cfg_path: str, step: str) -> tuple[dict, Path]:
    """Laad config + maak `output_dir/<step>` aan."""
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, step)
    return cfg, out_dir


def video_params(cfg: dict) -> tuple[float, int, int]:
    """`(fps, stride, screenshot_every)` uit de config + video."""
    fps = video_fps(cfg["video_path"])
    stride = int(cfg.get("frame_stride", 1))
    screenshot_every = int(cfg.get("screenshot_every", 150))
    return fps, stride, screenshot_every


def open_debug_writer(out_dir: Path, filename: str, fps: float, stride: int,
                      cfg: dict, flag_key: str) -> DebugVideoWriter | None:
    """Debug-video writer of `None` als `flag_key` in de config uit staat."""
    if not cfg.get(flag_key, True):
        return None
    effective_fps = fps / stride if stride else fps
    return DebugVideoWriter(out_dir / filename, effective_fps,
                            encoder=cfg.get("video_encoder", "auto"))


def should_screenshot(screenshot_every: int, frame_idx: int) -> bool:
    """Voorbeeldframe nodig op dit frame?"""
    return screenshot_every > 0 and frame_idx % screenshot_every == 0


def finish_detection_run(out_dir: Path, step: str, *, model: str, device: str,
                         video_path: str, frames: VideoFrames,
                         started: float, rows: list[dict], timer: InferenceTimer,
                         csv_file: str, columns: Sequence[str] | None = None,
                         class_field: str = "klasse",
                         extra: dict | None = None) -> dict:
    """Schrijf CSV + summary en print één regel resultaat. Retourneert summary."""
    columns = list(columns) if columns is not None else (
        list(rows[0].keys()) if rows else None)
    write_rows_csv(out_dir, csv_file, rows, columns)
    summary = write_run_summary(
        out_dir, step, model=model, device=device, video=video_path,
        frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file=csv_file, class_field=class_field, extra=extra)
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} rijen, "
          f"{summary['inference_fps']} fps -> {out_dir}")
    return summary


# Type voor annotatie-callbacks: (frame, detections) -> annotated frame.
AnnotateFn = Callable[[Any, Any], Any]


def run_detection_loop(cfg: dict, out_dir: Path, *, desc: str,
                       infer_fn: Callable[[Any], Any],
                       row_fn: Callable[[Any, int, Any], list[dict]],
                       annotate_fn: AnnotateFn | None,
                       csv_file: str, video_file: str, video_flag: str,
                       shot_prefix: str, model_id: str, device: str,
                       columns: Sequence[str] | None = None,
                       extra_summary: dict | None = None,
                       class_field: str = "klasse") -> dict:
    """Generieke detectie-loop (video in, CSV + debug-video uit).

    `infer_fn(frame) -> detections` doet de modelaanroep binnen de timer;
    `row_fn(frame, frame_idx, detections) -> list[dict]` bouwt CSV-rijen;
    `annotate_fn(frame, detections) -> annotated` tekent voor video/shot
    (of `None` als de caller raw `result.plot()` gebruikt — dan liever de
    oude expliciete loop behouden).
    """
    fps, stride, screenshot_every = video_params(cfg)
    timer = InferenceTimer()
    video = open_debug_writer(out_dir, video_file, fps, stride, cfg,
                              video_flag)
    rows: list[dict] = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc=desc)
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            with timer:
                detections = infer_fn(frame)
            rows.extend(row_fn(frame, frame_idx, detections))
            need_video = video is not None
            need_shot = should_screenshot(screenshot_every, frame_idx)
            if (need_video or need_shot) and annotate_fn is not None:
                annotated = annotate_fn(frame, detections)
                if need_video:
                    video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, shot_prefix, frame_idx, annotated)
    finally:
        if video is not None:
            video.release()
    return finish_detection_run(
        out_dir, out_dir.name, model=model_id, device=device,
        video_path=cfg["video_path"], frames=frames, started=started,
        rows=rows, timer=timer, csv_file=csv_file, columns=columns,
        class_field=class_field,
        extra={"conf_threshold": cfg.get("conf_threshold"),
               "frame_stride": stride, **(extra_summary or {})})
