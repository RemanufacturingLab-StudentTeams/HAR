"""Grounding DINO (open-vocabulary) op een video; logt tool-detecties naar CSV.

Grounding DINO krijgt de gereedschapsnamen uit ``tool_classes`` als
tekstprompt ("screwdriver. wrench. ..."), zodat geen finetuning nodig is.
"""
from duplo.annotate import draw_detections
from duplo.cli import parse_args
from duplo.detection import GroundingDinoDetector
from duplo.pipeline import (load_step_config, run_detection_loop,
                             setup_runtime, video_params)
from duplo.schema import DETECTION_COLUMNS as CSV_COLUMNS
from duplo.schema import detection_row


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "grounding_dino")
    device = setup_runtime(cfg, device)
    detector = GroundingDinoDetector(cfg, device)
    print(f"Grounding DINO ({detector.model_id}) draait op "
          f"{detector.torch_device.upper()} met prompt: {detector.prompt}")

    fps, _stride, _shot = video_params(cfg)

    def rows_for(frame, frame_idx, detections):
        rows = []
        for detection in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
            rows.append(detection_row(
                cfg["video_path"], frame_idx, fps, detection.class_name,
                detection.confidence, x1, y1, x2, y2))
        return rows

    run_detection_loop(
        cfg, out_dir, desc="Grounding DINO",
        infer_fn=detector.detect, row_fn=rows_for,
        annotate_fn=draw_detections,
        csv_file="grounding_dino_detections.csv",
        video_file="grounding_dino_debug.mp4",
        video_flag="save_grounding_dino_video",
        shot_prefix="grounding_dino_frame",
        model_id=detector.model_id, device=device, columns=CSV_COLUMNS,
        extra_summary={"conf_threshold": detector.box_threshold,
                       "tekstprompt": detector.prompt})


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
