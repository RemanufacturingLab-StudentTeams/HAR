"""YOLO-World (open-vocabulary) op een video; logt tool-detecties naar CSV.

YOLO-World krijgt de gereedschapsnamen uit ``tool_classes`` als tekstprompt,
zodat het zonder finetuning op de DUPLO-gereedschappen kan worden getest.
"""
import time

from duplo.benchmark import InferenceTimer
from duplo.cli import parse_args
from duplo.detection import _load_yolo_model
from duplo.gpu import to_cuda_device
from duplo.output import save_screenshot
from duplo.pipeline import (finish_detection_run, load_step_config,
                             open_debug_writer, setup_runtime,
                             should_screenshot, video_params)
from duplo.schema import DETECTION_COLUMNS as CSV_COLUMNS
from duplo.schema import detection_row
from duplo.video import VideoFrames


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "yolo_world")
    device = setup_runtime(cfg, device)
    # Pas importeren na een eventuele torch-installatie, anders draait een
    # reeds geladen CPU-build.
    from ultralytics import YOLO

    yolo_device = to_cuda_device(device)
    use_half = bool(cfg.get("yolo_half", True)) and yolo_device == "cuda"
    model_name = cfg.get("yolo_world_model", "models/yolov8x-worldv2.pt")
    tool_classes = list(cfg["tool_classes"])
    # TensorRT-backend is opt-in via yolo_backend; valt terug op .pt.
    cfg_world = dict(cfg, yolo_world_model=model_name)
    model = _load_yolo_model(YOLO, cfg_world, "yolo_world_model", yolo_device)
    if cfg.get("yolo_fuse", True):
        try:
            model.fuse(verbose=False)
        except Exception as exc:
            print(f"Waarschuwing: YOLO-fuse overgeslagen ({exc}).")
    model.set_classes(tool_classes)
    print(f"YOLO-World ({model_name}) draait op {yolo_device.upper()}"
          f"{' (FP16)' if use_half else ''} "
          f"met klassen: {', '.join(tool_classes)}")

    fps, stride, screenshot_every = video_params(cfg)
    timer = InferenceTimer()
    video = open_debug_writer(out_dir, "yolo_world_debug.mp4", fps, stride,
                              cfg, "save_yolo_world_video")
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="YOLO-World")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            with timer:
                result = model.predict(
                    frame, conf=cfg["conf_threshold"], device=yolo_device,
                    verbose=False, quantize=16 if use_half else None,
                    augment=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                                  int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                rows.append(detection_row(
                    cfg["video_path"], frame_idx, fps,
                    result.names[int(box.cls)], float(box.conf),
                    x1, y1, x2, y2))
            need_video = video is not None
            need_shot = should_screenshot(screenshot_every, frame_idx)
            if need_video or need_shot:
                annotated = result.plot()
                if need_video:
                    video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, "yolo_world_frame", frame_idx,
                                    annotated)
    finally:
        if video is not None:
            video.release()

    finish_detection_run(
        out_dir, "yolo_world", model=model_name, device=device,
        video_path=cfg["video_path"], frames=frames, started=started,
        rows=rows, timer=timer, csv_file="yolo_world_detections.csv",
        columns=CSV_COLUMNS,
        extra={"conf_threshold": cfg["conf_threshold"],
               "frame_stride": stride, "tekstprompt": tool_classes})


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
