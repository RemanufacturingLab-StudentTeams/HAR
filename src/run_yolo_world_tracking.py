"""YOLO-World + ByteTrack op een video; logt de trackgeschiedenis en schrijft
een debug-video met bboxes, track-ID's en trajectories.

YOLO-World krijgt de gereedschapsnamen uit ``tool_classes`` als tekstprompt;
ByteTrack koppelt de detecties tussen frames aan dezelfde track-ID's, net als
in ``run_tracking.py`` voor het gesloten-vocabulaire YOLO-model.
"""
import time

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.detection import YoloDetector
from duplo.output import write_rows_csv
from duplo.pipeline import load_step_config, setup_runtime
from duplo.tracking import (CSV_COLUMNS, ObjectTracker, build_row, draw_tracks)
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "yolo_world_tracking")
    device = setup_runtime(cfg, device)
    model_name = cfg.get("yolo_world_model", "models/yolov8x-worldv2.pt")
    detector = YoloDetector(cfg, device, model_key="yolo_world_model",
                            classes=list(cfg["tool_classes"]))
    tracker = ObjectTracker(cfg)
    print(f"YOLO-World ({model_name}) + ByteTrack draaien op {device.upper()} "
          f"met klassen: {', '.join(cfg['tool_classes'])}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    low_conf = cfg.get("track_low_thresh", 0.1)
    timer = InferenceTimer()

    rows = []
    video = (DebugVideoWriter(out_dir / "yolo_world_tracking_debug.mp4",
                              fps / stride,
                              encoder=cfg.get("video_encoder", "auto"))
             if cfg.get("save_yolo_world_tracking_video", True) else None)
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="YOLO-World")
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

    write_rows_csv(out_dir, "yolo_world_tracking_history.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "yolo_world_tracking", model=model_name, device=device,
        video=cfg["video_path"], frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="yolo_world_tracking_history.csv",
        class_field="class_name",
        extra={"conf_threshold": cfg["conf_threshold"], "frame_stride": stride,
               "tekstprompt": list(cfg["tool_classes"]),
               "track_low_thresh": low_conf})
    track_ids = {row["track_id"] for row in rows}
    print(f"{frames.frames_read} frames verwerkt, {len(track_ids)} tracks, "
          f"{len(rows)} rijen, {summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
