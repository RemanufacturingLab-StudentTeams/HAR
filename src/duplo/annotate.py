"""Gedeelde annotatie-helpers voor debug-video's en voorbeeldframes.

Eén plek voor bbox/masker-tekenen, zodat `run_sam3_video.py` en
`run_sam31_video.py` (voorheen identieke `draw_segmentations`) en de
RF-DETR `annotate()` varianten niet divergeren. YOLO `result.plot()`
blijft YOLO-specifiek en valt hier buiten.
"""
import cv2


def draw_detections(frame, detections):
    """Kopie van het frame met bboxes en labels (groen, 2px)."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        text = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, text, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


def draw_segmentations(frame, detections):
    """Kopie van het frame met maskers (overlay), bboxes en labels."""
    annotated = frame.copy()
    overlay = annotated.copy()
    for detection in detections:
        mask = getattr(detection, "mask", None)
        if mask is not None:
            try:
                overlay[mask] = (0, 200, 0)
            except (IndexError, ValueError):
                # Masker met afwijkende resolutie: overslaan i.p.v. crashen.
                pass
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{detection.class_name} "
                               f"{detection.confidence:.2f}",
                    (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    return cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)


def supervision_annotate(frame, detections, sv, mask_annotator=None,
                         box_annotator=None, label_annotator=None,
                         with_mask: bool = False):
    """Supervision-annotatie voor RF-DETR (detectie of segmentatie).

    `sv` wordt als parameter meegegeven zodat `supervision` pas na een
    eventuele CUDA-torch-installatie geïmporteerd hoeft te worden in de
    caller. Annotators worden hergebruikt als ze zijn meegegeven.
    """
    labels = [f"{name} {float(confidence):.2f}" for name, confidence in
              zip(detections.data["class_name"], detections.confidence)]
    box_annotator = box_annotator or sv.BoxAnnotator(thickness=2)
    label_annotator = (label_annotator
                       or sv.LabelAnnotator(text_scale=0.5, text_thickness=1))
    annotated = frame
    if with_mask:
        mask_annotator = mask_annotator or sv.MaskAnnotator()
        annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    return label_annotator.annotate(annotated, detections, labels)
