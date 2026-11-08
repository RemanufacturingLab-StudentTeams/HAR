"""SAM3 (Segment Anything 3) op een video; logt tool-segmentaties naar CSV.

SAM3 segmenteert elk object dat bij een tekstprompt hoort (open-vocabulary),
zonder finetuning. De gewichten van ``facebook/sam3`` staan achter een
Hugging Face-toegangsaanvraag: vraag die aan op
https://huggingface.co/facebook/sam3 en log daarna in met ``hf auth login``.
Op Windows is daarnaast het community-pakket ``triton-windows`` nodig, omdat
sam3 ``triton`` importeert en het officiële pakket geen Windows-wheels heeft.
"""
from duplo.annotate import draw_segmentations
from duplo.cli import parse_args
from duplo.detection import Sam3Detector
from duplo.pipeline import (load_step_config, run_detection_loop,
                             setup_runtime, video_params)
from duplo.schema import SEGMENTATION_COLUMNS as CSV_COLUMNS
from duplo.schema import segmentation_row


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "sam3")
    device = setup_runtime(cfg, device)
    detector = Sam3Detector(cfg, device)
    print(f"SAM3 ({detector.model_id}) draait op {detector.device.upper()} "
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
        cfg, out_dir, desc="SAM3",
        infer_fn=detector.detect, row_fn=rows_for,
        annotate_fn=draw_segmentations,
        csv_file="sam3_detections.csv", video_file="sam3_debug.mp4",
        video_flag="save_sam3_video", shot_prefix="sam3_frame",
        model_id=detector.model_id, device=device, columns=CSV_COLUMNS,
        extra_summary={"tekstprompt": detector.prompts})


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
