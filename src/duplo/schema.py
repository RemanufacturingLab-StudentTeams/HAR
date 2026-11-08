"""Centrale CSV-schema's en rij-builders voor alle pipeline-stappen.

Eén bron voor kolomvolgorde, zodat `run_*` scripts en `compare_models.py`
niet divergeren. Detectie (9 kolommen, met `timestamp`), segmentatie
(10 kolommen, met `mask_area`), tracking (18 kolommen uit `tracking.py`),
handpose (8 kolommen) en VITRA-acties (14 kolommen per voorspelde stap)
staan hier vast.
"""
import numpy as np

DETECTION_COLUMNS = ["video", "frame", "timestamp", "klasse", "confidence",
                     "bbox_x", "bbox_y", "bbox_w", "bbox_h"]

SEGMENTATION_COLUMNS = [*DETECTION_COLUMNS, "mask_area"]

HANDPOSE_COLUMNS = ["video", "frame", "hand", "landmark", "x", "y", "z",
                    "zichtbaar"]

VITRA_COLUMNS = ["video", "frame", "timestamp", "sample", "hand", "step",
                 "transl_x", "transl_y", "transl_z", "rot_x", "rot_y",
                 "rot_z", "transl_norm", "pose_mean"]

VITRA_HAND_SLICES = {"left": slice(0, 51), "right": slice(51, 102)}


def detection_row(video_path: str, frame_idx: int, fps: float,
                  klasse: str, confidence: float,
                  x1: int, y1: int, x2: int, y2: int) -> dict:
    """Eén detectie-rij met `timestamp` en `bbox_x/y/w/h` conventie."""
    return {
        "video": video_path,
        "frame": frame_idx,
        "timestamp": round(frame_idx / fps, 3) if fps else 0.0,
        "klasse": klasse,
        "confidence": round(float(confidence), 3),
        "bbox_x": int(x1),
        "bbox_y": int(y1),
        "bbox_w": int(x2 - x1),
        "bbox_h": int(y2 - y1),
    }


def segmentation_row(video_path: str, frame_idx: int, fps: float,
                     klasse: str, confidence: float,
                     x1: int, y1: int, x2: int, y2: int,
                     mask_area: int) -> dict:
    """Eén segmentatie-rij (`mask_area` als pixel-count op frame-resolutie)."""
    row = detection_row(video_path, frame_idx, fps, klasse, confidence,
                        x1, y1, x2, y2)
    row["mask_area"] = int(mask_area)
    return row


def vitra_action_rows(video_path: str, frame_idx: int, fps: float,
                      actions: np.ndarray,
                      hands: tuple[str, ...]) -> list[dict]:
    """CSV-rijen per voorspelde hand/stap uit VITRA's actiechunk.

    ``actions`` is ``[T, 102]`` of ``[S, T, 102]`` (VITRA's
    ``predict_action`` levert het tweede). Alleen de gekozen handen worden
    gelogd; de 51 dimensies per hand zijn translatie (3), globale rotatie
    (3) en 45 MANO-gewrichtshoeken. ``transl_norm`` en ``pose_mean`` vatten
    de stap samen voor analyse en de debug-overlay.
    """
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] < 102:
        raise ValueError(
            f"VITRA-acties moeten (T, 102) of (S, T, 102) zijn, kreeg "
            f"{array.shape}.")
    unknown = set(hands) - set(VITRA_HAND_SLICES)
    if unknown:
        raise ValueError(f"Onbekende hand(en): {sorted(unknown)}")
    rows = []
    for sample in range(array.shape[0]):
        for hand in hands:
            chunk = array[sample, :, VITRA_HAND_SLICES[hand]]
            for step, action in enumerate(chunk):
                transl = action[:3]
                pose = action[6:51]
                rows.append({
                    "video": video_path,
                    "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 3) if fps else 0.0,
                    "sample": sample,
                    "hand": hand,
                    "step": step,
                    "transl_x": round(float(transl[0]), 6),
                    "transl_y": round(float(transl[1]), 6),
                    "transl_z": round(float(transl[2]), 6),
                    "rot_x": round(float(action[3]), 6),
                    "rot_y": round(float(action[4]), 6),
                    "rot_z": round(float(action[5]), 6),
                    "transl_norm": round(float(np.linalg.norm(transl)), 6),
                    "pose_mean": round(float(np.mean(np.abs(pose))), 6),
                })
    return rows
