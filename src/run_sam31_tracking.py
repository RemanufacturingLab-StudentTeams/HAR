"""SAM 3.1 + ByteTrack op een video; logt de trackgeschiedenis en schrijft een
debug-video met bboxes, track-ID's en trajectories.

SAM 3.1 segmenteert de gereedschappen uit ``tool_classes`` via de tekstprompt
(Ultralytics ``SAM3SemanticPredictor`` op ``sam3.1_multiplex.pt``); ByteTrack
koppelt de detecties tussen frames aan dezelfde track-ID's, net als in
``run_tracking.py`` voor het YOLO-model. Voor SAM 3.1's eigen videotracker
(Object Multiplex) is er nog geen Ultralytics-ondersteuning, dus deze test
gebruikt per-frame beeldinferentie + ByteTrack, net als ``run_sam3_tracking.py``.
"""
import time

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.detection import Sam31Detector
from duplo.output import write_rows_csv
from duplo.pipeline import load_step_config, setup_runtime
from duplo.tracking import (CSV_COLUMNS, ObjectTracker, build_row, draw_tracks)
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "sam31_tracking")
    device = setup_runtime(cfg, device)
    detector = Sam31Detector(cfg, device)
    tracking_cfg = {**cfg, **cfg.get("sam31_tracking", {})}
    tracker = ObjectTracker(tracking_cfg)
    print(f"SAM 3.1 ({detector.model_id}) + ByteTrack draaien op "
          f"{detector.device.upper()} met prompts: {', '.join(detector.prompts)}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    low_conf = tracking_cfg.get(
        "detection_conf", tracking_cfg.get("track_low_thresh", 0.1))
    timer = InferenceTimer()

    rows = []
    video = (DebugVideoWriter(out_dir / "sam31_tracking_debug.mp4", fps / stride,
                              encoder=cfg.get("video_encoder", "auto"))
             if cfg.get("save_sam31_tracking_video", True) else None)
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="SAM 3.1")
    started = time.perf_counter()
    try:
        for frame_number, frame in frames:
            # Detecties met een lage score gaan mee: ByteTrack gebruikt die in
            # de tweede associatiestap om occlusies te overbruggen.
            with timer:
                detections = detector.detect(frame, conf=low_conf)
            observations = tracker.update(frame_number, frame, detections)
            rows.extend(build_row(cfg["video_path"], frame_number, fps,
                                  observation, detector.class_names)
                        for observation in observations)
            if video is not None:
                draw_tracks(frame, observations, tracker.trajectories,
                            detector.class_names)
                video.write(frame)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "sam31_tracking_history.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "sam31_tracking", model=detector.model_id, device=device,
        video=cfg["video_path"], frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="sam31_tracking_history.csv", class_field="class_name",
        extra={"conf_threshold": detector.conf_threshold, "frame_stride": stride,
               "tekstprompt": detector.prompts, "track_low_thresh": low_conf,
               "tracking_config": cfg.get("sam31_tracking", {})})
    track_ids = {row["track_id"] for row in rows}
    print(f"{frames.frames_read} frames verwerkt, {len(track_ids)} tracks, "
          f"{len(rows)} rijen, {summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
