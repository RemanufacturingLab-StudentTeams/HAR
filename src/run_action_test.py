"""Actieclassificatie met X-CLIP, InternVideo2 of ActionCLIP.

De video wordt in clips verdeeld. Elk model classificeert een clip tegen de
actieklassen uit ``action_classes`` en schrijft één rij per clip naar
``<model>_actions.csv``. De debug-video toont per clip het actielabel, de
confidence en de top-k scores.

Kies het model met ``--model xclip``, ``internvideo2`` of ``actionclip`` (of
met ``action_model`` in de config). Alle modellen gebruiken dezelfde
clipindeling en hetzelfde CSV-schema, zodat ``compare_models.py`` ze naast
elkaar kan zetten.
"""
import time

from tqdm import tqdm

from duplo.action_eval import evaluate_action_rows, load_action_ground_truth
from duplo.actions import (
    annotate_action,
    clip_windows,
    create_action_classifier,
    normalize_action_model,
)
from duplo.benchmark import (
    TASK_ACTION_CLASSIFICATION,
    InferenceTimer,
    write_run_summary,
)
from duplo.cli import parse_args
from duplo.config import load_config, resolve_video_path
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.pipeline import setup_runtime
from duplo.schema import ACTION_COLUMNS, action_row
from duplo.video_writer import DebugVideoWriter


def main(cfg_path: str, device: str | None = None,
         model: str | None = None) -> None:
    cfg = load_config(cfg_path)
    resolve_video_path(cfg)
    kind = normalize_action_model(model or cfg.get("action_model", "xclip"))
    out_dir = prepare_output_dir(cfg, kind)
    device = setup_runtime(cfg, device)

    classifier = create_action_classifier(cfg, device, kind)
    print(f"{classifier.model_id} draait op {device.upper()} met "
          f"{len(classifier.classes)} actieklassen")

    num_frames = int(classifier.num_frames)
    span = max(int(cfg.get("action_span", 30)), num_frames)
    stride = max(1, int(cfg.get("action_stride", span)))
    top_k = int(cfg.get("action_top_k", 3))
    screenshot_every = int(cfg.get("screenshot_every", 150))
    repeat_frames = bool(cfg.get("action_debug_repeat_frames", True))

    from duplo.actions import VideoClipReader

    reader = VideoClipReader(cfg["video_path"])
    windows = clip_windows(reader.total_frames, num_frames, span, stride)
    if not windows:
        reader.close()
        raise SystemExit(
            f"Geen clips gevonden in {cfg['video_path']} "
            f"({reader.total_frames} frames).")

    effective_fps = reader.fps if repeat_frames else max(
        1.0, reader.fps / stride)
    video = (DebugVideoWriter(out_dir / f"{kind}_debug.mp4", effective_fps,
                              encoder=cfg.get("video_encoder", "auto"))
             if cfg.get(f"save_{kind}_video", True) else None)
    timer = InferenceTimer()
    rows: list[dict] = []
    started = time.perf_counter()
    try:
        for window in tqdm(windows, desc=kind.upper(), unit="clip",
                          disable=None):
            frames = reader.read(window.indices)
            clip_start_s = window.start / reader.fps
            clip_end_s = window.end / reader.fps
            with timer:
                prediction = classifier.predict(
                    frames, top_k=top_k, clip_start_s=clip_start_s,
                    clip_end_s=clip_end_s)
            rows.append(action_row(
                cfg["video_path"], window.start, window.end, reader.fps,
                prediction.label, prediction.confidence,
                prediction.top_k(top_k), start_s=prediction.start_s,
                end_s=prediction.end_s, raw_response=prediction.raw_text))

            display_idx = window.indices[len(window.indices) // 2]
            display = frames[len(frames) // 2]
            need_video = video is not None
            need_shot = (screenshot_every > 0
                         and window.start % screenshot_every == 0)
            if need_video or need_shot:
                annotated = annotate_action(
                    display, prediction, display_idx, reader.fps,
                    window.start, window.end, top_k=top_k)
                if need_video:
                    repeats = max(1, min(stride,
                                         reader.total_frames - window.start))
                    for _ in range(repeats if repeat_frames else 1):
                        video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, f"{kind}_frame",
                                    window.start, annotated)
    finally:
        reader.close()
        if video is not None:
            video.release()

    csv_file = f"{kind}_actions.csv"
    write_rows_csv(out_dir, csv_file, rows, ACTION_COLUMNS)
    evaluation = None
    ground_truth_path = str(cfg.get("action_ground_truth") or "").strip()
    if ground_truth_path:
        ground_truth = load_action_ground_truth(
            ground_truth_path, video=cfg["video_path"])
        evaluation = evaluate_action_rows(rows, ground_truth, fps=reader.fps)
    summary = write_run_summary(
        out_dir, kind, model=classifier.model_id, device=device,
        video=cfg["video_path"], frames_read=reader.total_frames,
        frames_processed=len(rows), elapsed_s=time.perf_counter() - started,
        rows=rows, timer=timer, csv_file=csv_file, class_field="klasse",
        task=TASK_ACTION_CLASSIFICATION,
        extra={
            "action_classes": list(classifier.classes),
            "prompts": list(classifier.prompt_set.prompts),
            "clip_frames": num_frames,
            "clip_span": span,
            "clip_stride": stride,
            "fps": reader.fps,
            "clips_processed": len(rows),
            "video_frames": reader.total_frames,
            "top_k": top_k,
            "temporal": bool(getattr(classifier, "temporal", False)),
            "evaluation": evaluation,
        })
    print(f"{len(rows)} clips geclassificeerd, {len(rows)} rijen, "
          f"{summary['inference_fps']} clips/s -> {out_dir}")
    if evaluation:
        print("Classificatie-accuracy: "
              f"{100 * evaluation['classification_accuracy']:.1f}% | "
              f"temporele mIoU: {evaluation['temporal_iou_mean']:.3f} | "
              f"R@0.5: {evaluation['temporal_recall_0.5']:.3f}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device, args.model)
