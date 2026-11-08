"""MediaPipe Hands op een video; logt 21 landmerken per hand naar CSV."""
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.schema import HANDPOSE_COLUMNS
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def get_model_path(cfg: dict) -> Path:
    model_path = Path(cfg.get("hand_model_path", "models/hand_landmarker.task"))
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Handmodel downloaden naar {model_path}...")
    # Eerst naar een tijdelijk bestand downloaden, zodat een afgebroken
    # download geen half model op de definitieve plek achterlaat.
    temp_path = model_path.with_name(f"{model_path.name}.part")
    try:
        urlretrieve(MODEL_URL, temp_path)
        temp_path.replace(model_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return model_path


def draw_landmarks(frame, hand_landmarks) -> None:
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def draw_all_hands(frame, hand_landmarks):
    """Kopie van het frame met alle gedetecteerde handen erop getekend."""
    annotated = frame.copy()
    for landmarks in hand_landmarks:
        draw_landmarks(annotated, landmarks)
    return annotated


def draw_frame_counter(frame, frame_idx: int):
    """Toon het oorspronkelijke videoframenummer op een debugframe."""
    counter = f"Frame: {frame_idx}"
    origin = (12, 32)
    cv2.putText(frame, counter, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, counter, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def create_landmarker(cfg: dict, model_path: Path, delegate) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path), delegate=delegate),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=cfg["max_hands"],
        min_hand_detection_confidence=cfg["conf_threshold"],
        min_hand_presence_confidence=cfg["conf_threshold"],
        min_tracking_confidence=cfg["conf_threshold"],
    )
    return vision.HandLandmarker.create_from_options(options)


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "handpose")

    device = resolve_device(device, cfg)
    model_path = get_model_path(cfg)
    if device == "gpu" and sys.platform == "win32":
        print("MediaPipe GPU-delegate is niet beschikbaar in de Windows-build; "
              "gebruik CPU.")
        device = "cpu"
        hands = create_landmarker(
            cfg, model_path, python.BaseOptions.Delegate.CPU)
    elif device == "gpu":
        try:
            hands = create_landmarker(
                cfg, model_path, python.BaseOptions.Delegate.GPU)
        except RuntimeError as exc:
            print(f"GPU-delegate niet beschikbaar ({exc}); valt terug op CPU.")
            device = "cpu"
            hands = create_landmarker(
                cfg, model_path, python.BaseOptions.Delegate.CPU)
    else:
        hands = create_landmarker(
            cfg, model_path, python.BaseOptions.Delegate.CPU)
    print(f"MediaPipe draait op {device.upper()}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    screenshot_every = cfg.get("screenshot_every", 150)
    timer = InferenceTimer()
    video = (DebugVideoWriter(out_dir / "handpose_debug.mp4", fps / stride,
                              encoder=cfg.get("video_encoder", "auto"))
             if cfg.get("save_handpose_video", True) else None)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="MediaPipe")
    started = time.perf_counter()
    try:
        with hands:
            for frame_idx, frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_idx * 1000 / fps)
                with timer:
                    res = hands.detect_for_video(image, timestamp_ms)
                has_hands = bool(res.hand_landmarks)
                if has_hands:
                    for h_idx, landmarks in enumerate(res.hand_landmarks):
                        for p_idx, point in enumerate(landmarks):
                            rows.append({
                                "video": cfg["video_path"], "frame": frame_idx,
                                "hand": h_idx, "landmark": p_idx,
                                "x": round(point.x, 4), "y": round(point.y, 4),
                                "z": round(point.z, 4), "zichtbaar": 1.0,
                            })
                    if screenshot_every > 0 and frame_idx % screenshot_every == 0:
                        annotated = draw_all_hands(frame, res.hand_landmarks)
                        save_screenshot(out_dir, "hand_frame", frame_idx,
                                        annotated)
                if video is not None:
                    debug_frame = (draw_all_hands(frame, res.hand_landmarks)
                                   if has_hands else frame.copy())
                    draw_frame_counter(debug_frame, frame_idx)
                    video.write(debug_frame)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "hand_landmarks.csv", rows, HANDPOSE_COLUMNS)
    # Handpose heeft geen 'klasse'-kolom; registreer timing + counts met een
    # dummy-rijenlijst voor write_run_summary (alleen fps/tijd relevant).
    summary = write_run_summary(
        out_dir, "handpose", model=str(model_path), device=device,
        video=cfg["video_path"], frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=[], timer=timer,
        csv_file="hand_landmarks.csv", class_field="hand",
        extra={"conf_threshold": cfg.get("conf_threshold"),
               "frame_stride": stride,
               "landmark_rows": len(rows)})
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} landmarks, "
          f"{summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)