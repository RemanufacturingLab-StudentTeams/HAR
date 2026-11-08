"""SAM 3.1 (Segment Anything 3.1, Object Multiplex-checkpoint) op een video.

SAM 3.1 segmenteert elk object dat bij een tekstprompt hoort
(open-vocabulary), zonder finetuning. Via Ultralytics draait de
beeldinferentie met ``SAM3SemanticPredictor`` op ``sam3.1_multiplex.pt``;
Object Multiplex-videotracking wordt door Ultralytics nog niet ondersteund,
dus deze test doet per-frame beeldinferentie (net als de SAM3-test).

De gewichten van ``facebook/sam3.1`` staan achter een Hugging
Face-toegangsaanvraag: vraag die aan op
https://huggingface.co/facebook/sam3.1 en log daarna in met
``huggingface-cli login`` (of ``hf auth login``). Daarna downloadt Ultralytics
``sam3.1_multiplex.pt`` automatisch, of zet een lokaal bestand via
``sam31_model`` in de config.
"""
from duplo.annotate import draw_segmentations
from duplo.cli import parse_args
from duplo.detection import Sam31Detector
from duplo.pipeline import (load_step_config, run_detection_loop,
                             setup_runtime, video_params)
from duplo.schema import SEGMENTATION_COLUMNS as CSV_COLUMNS
from duplo.schema import segmentation_row


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "sam31")
    device = setup_runtime(cfg, device)
    detector = Sam31Detector(cfg, device)
    print(f"SAM 3.1 ({detector.model_id}) draait op {detector.device.upper()} "
          f"met prompts: {', '.join(detector.prompts)}")

    fps, _stride, _shot = video_params(cfg)

    def rows_for(frame, frame_idx, detections):
        rows = []
        for detection in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
            rows.append(segmentation_row(
                cfg["video_path"], frame_idx, fps, detection.class_name,
                detection.confidence, x1, y1, x2, y2,
                0 if detection.mask is None else int(detection.mask.sum())))
        return rows

    run_detection_loop(
        cfg, out_dir, desc="SAM 3.1",
        infer_fn=detector.detect, row_fn=rows_for,
        annotate_fn=draw_segmentations,
        csv_file="sam31_detections.csv", video_file="sam31_debug.mp4",
        video_flag="save_sam31_video", shot_prefix="sam31_frame",
        model_id=detector.model_id, device=device, columns=CSV_COLUMNS,
        extra_summary={"tekstprompt": detector.prompts})


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
