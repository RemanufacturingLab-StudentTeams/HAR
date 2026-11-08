"""YOLO-segmentatie (YOLO26-seg) op een video; logt detecties met maskeroppervlak.

Zelfde opzet als de YOLO-detectietest, maar met een segmentatiemodel zodat
de toolmaskers (en hun oppervlak) kunnen worden vergeleken met RF-DETR-seg.
"""
import time

import cv2
import numpy as np

from duplo.benchmark import InferenceTimer
from duplo.cli import parse_args
from duplo.detection import _load_yolo_model
from duplo.gpu import to_cuda_device
from duplo.output import save_screenshot
from duplo.pipeline import (finish_detection_run, load_step_config,
                             open_debug_writer, setup_runtime,
                             should_screenshot, video_params)
from duplo.schema import SEGMENTATION_COLUMNS as CSV_COLUMNS
from duplo.schema import segmentation_row
from duplo.video import VideoFrames


def mask_area(result, index: int, frame_shape=None) -> int:
    """Maskeroppervlak als pixel-count op frame-resolutie (vergelijkbaar).

    Rasterizeert het polygoon naar een frame-groot masker en telt pixels,
    zodat `mask_area` dezelfde definitie heeft als RF-DETR-seg/SAM
    (`mask.sum()` op frame-resolutie) en cross-model vergelijkbaar is.
    """
    if result.masks is None:
        return 0
    polygon = result.masks.xy[index]
    if len(polygon) < 3:
        return 0
    if frame_shape is None:
        return int(cv2.contourArea(polygon))
    height, width = frame_shape[:2]
    canvas = np.zeros((height, width), dtype="uint8")
    pts = polygon.astype(int).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], 1)
    return int(canvas.sum())


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "yolo_seg")
    device = setup_runtime(cfg, device)
    from ultralytics import YOLO

    yolo_device = to_cuda_device(device)
    use_half = bool(cfg.get("yolo_half", True)) and yolo_device == "cuda"
    model_name = cfg.get("yolo_seg_model", "models/yolo26x-seg.pt")
    cfg_seg = dict(cfg, yolo_seg_model=model_name)
    model = _load_yolo_model(YOLO, cfg_seg, "yolo_seg_model", yolo_device)
    if cfg.get("yolo_fuse", True):
        try:
            model.fuse(verbose=False)
        except Exception as exc:
            print(f"Waarschuwing: YOLO-fuse overgeslagen ({exc}).")
    print(f"YOLO-seg ({model_name}) draait op {yolo_device.upper()}"
          f"{' (FP16)' if use_half else ''}")

    fps, stride, screenshot_every = video_params(cfg)
    timer = InferenceTimer()
    video = open_debug_writer(out_dir, "yolo_seg_debug.mp4", fps, stride,
                              cfg, "save_yolo_seg_video")
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="YOLO-seg")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            with timer:
                result = model.predict(
                    frame, conf=cfg["conf_threshold"], device=yolo_device,
                    verbose=False, quantize=16 if use_half else None,
                    augment=False)[0]
            for index, box in enumerate(result.boxes):
                x1, y1, x2, y2 = (int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                                  int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                rows.append(segmentation_row(
                    cfg["video_path"], frame_idx, fps,
                    result.names[int(box.cls)], float(box.conf),
                    x1, y1, x2, y2, mask_area(result, index, frame.shape)))
            need_video = video is not None
            need_shot = should_screenshot(screenshot_every, frame_idx)
            if need_video or need_shot:
                annotated = result.plot()
                if need_video:
                    video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, "yolo_seg_frame", frame_idx,
                                    annotated)
    finally:
        if video is not None:
            video.release()

    finish_detection_run(
        out_dir, "yolo_seg", model=model_name, device=device,
        video_path=cfg["video_path"], frames=frames, started=started,
        rows=rows, timer=timer, csv_file="yolo_seg_detections.csv",
        columns=CSV_COLUMNS,
        extra={"conf_threshold": cfg["conf_threshold"],
               "frame_stride": stride})


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
