"""RF-DETR-objectdetectie op een video; logt detections naar CSV en slaat frames op.

RF-DETR (Roboflow) draait met de pretrained COCO-gewichten van het
``rfdetr``-pakket; het model volgt dezelfde opzet als de YOLO-test.
"""
import time

import cv2

from duplo.annotate import supervision_annotate
from duplo.benchmark import InferenceTimer
from duplo.cli import parse_args
from duplo.gpu import quiet_third_party_init
from duplo.output import save_screenshot
from duplo.pipeline import (finish_detection_run, load_step_config,
                             open_debug_writer, setup_runtime,
                             should_screenshot, video_params)
from duplo.schema import DETECTION_COLUMNS as CSV_COLUMNS
from duplo.schema import detection_row
from duplo.video import VideoFrames


def build_detector(cfg: dict, device: str):
    """RFDETR-model uit de config; pas importeren na een CUDA-installatie."""
    import rfdetr

    model_name = cfg.get("rfdetr_model", "RFDETRLarge")
    return model_name, getattr(rfdetr, model_name)(
        device="cuda" if device == "gpu" else "cpu")


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "rfdetr")
    device = setup_runtime(cfg, device)
    with quiet_third_party_init():
        model_name, model = build_detector(cfg, device)
        if device == "gpu" and cfg.get("rfdetr_optimize", True):
            import torch
            model.inference(dtype=torch.float16)
    print(f"RF-DETR ({model_name}) draait op {device.upper()}")

    import supervision as sv

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    threshold = cfg["conf_threshold"]
    fps, stride, screenshot_every = video_params(cfg)
    timer = InferenceTimer()
    video = open_debug_writer(out_dir, "rfdetr_debug.mp4", fps, stride,
                              cfg, "save_rfdetr_video")
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="RF-DETR")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with timer:
                detections = model.predict(
                    rgb_frame, threshold=threshold,
                    include_source_image=False)
            class_names = detections.data["class_name"]
            for index, (box, confidence) in enumerate(
                    zip(detections.xyxy, detections.confidence)):
                x1, y1, x2, y2 = (int(round(float(v))) for v in box)
                rows.append(detection_row(
                    cfg["video_path"], frame_idx, fps,
                    str(class_names[index]), float(confidence),
                    x1, y1, x2, y2))
            need_video = video is not None
            need_shot = should_screenshot(screenshot_every, frame_idx)
            if need_video or need_shot:
                annotated = supervision_annotate(
                    frame, detections, sv, box_annotator=box_annotator,
                    label_annotator=label_annotator)
                if need_video:
                    video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, "rfdetr_frame", frame_idx,
                                    annotated)
    finally:
        if video is not None:
            video.release()

    finish_detection_run(
        out_dir, "rfdetr", model=model_name, device=device,
        video_path=cfg["video_path"], frames=frames, started=started,
        rows=rows, timer=timer, csv_file="rfdetr_detections.csv",
        columns=CSV_COLUMNS,
        extra={"conf_threshold": threshold, "frame_stride": stride})


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
