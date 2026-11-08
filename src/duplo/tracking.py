"""ByteTrack multi-object tracking: stabiele track-ID's en objectgeschiedenis.

ByteTrack koppelt de YOLO-detecties tussen frames aan elkaar met een
Kalman-filter en IoU-associatie, geeft elk object een uniek track_id en
overbrugt korte occlusies door detecties met een lage confidence in een tweede
associatiestap te gebruiken. Hiervoor wordt de BYTETracker uit ultralytics
gebruikt (versie staat vast in requirements.txt); de tracker draait op de CPU.

Daarnaast staan hier de gedeelde weergave- en CSV-conventies van alle
trackingstappen (`CSV_COLUMNS`, `track_color`, `draw_tracks`, `build_row`),
zodat elke detector met hetzelfde uitvoerformaat werkt.
"""
import types
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from duplo.detection import Detection

if TYPE_CHECKING:
    from ultralytics.trackers.byte_tracker import STrack

CSV_COLUMNS = ["video", "frame_number", "timestamp", "track_id", "class_id",
               "class_name", "confidence", "bbox_x1", "bbox_y1", "bbox_x2",
               "bbox_y2", "center_x", "center_y", "velocity_x", "velocity_y",
               "track_age", "frames_since_detection", "status"]


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministische, per track goed te onderscheiden BGR-kleur."""
    hue = np.uint8([[[track_id * 47 % 180, 255, 255]]])
    blue, green, red = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0][0]
    return int(blue), int(green), int(red)


def draw_tracks(frame, observations: list["TrackObservation"], trajectories: dict,
                class_names: dict[int, str]) -> None:
    """Tekent bboxes, track-ID's, klassen en trajectories voor de debug-video."""
    for observation in observations:
        color = track_color(observation.track_id)
        if observation.bbox is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in observation.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            name = class_names.get(observation.class_id, str(observation.class_id))
            label = f"#{observation.track_id} {name} {observation.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            x1, y1, x2, y2 = (int(round(value))
                              for value in observation.predicted_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(frame,
                        f"#{observation.track_id} occluded "
                        f"({observation.frames_since_detection}f)",
                        (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (128, 128, 128), 1)
        points = trajectories.get(observation.track_id)
        if points and len(points) > 1:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False,
                          color, 2)


def build_row(video_path: str, frame_number: int, fps: float,
              observation: "TrackObservation",
              class_names: dict[int, str]) -> dict:
    """Eén CSV-rij per track per frame; occlusie-rijen hebben geen bbox."""
    bbox = observation.bbox
    center = observation.center
    velocity = observation.velocity
    return {
        "video": video_path,
        "frame_number": frame_number,
        "timestamp": round(frame_number / fps, 3),
        "track_id": observation.track_id,
        "class_id": observation.class_id,
        "class_name": class_names.get(observation.class_id),
        "confidence": (None if observation.confidence is None
                       else round(observation.confidence, 3)),
        "bbox_x1": None if bbox is None else int(round(bbox[0])),
        "bbox_y1": None if bbox is None else int(round(bbox[1])),
        "bbox_x2": None if bbox is None else int(round(bbox[2])),
        "bbox_y2": None if bbox is None else int(round(bbox[3])),
        "center_x": None if center is None else round(center[0], 1),
        "center_y": None if center is None else round(center[1], 1),
        "velocity_x": None if velocity is None else round(velocity[0], 2),
        "velocity_y": None if velocity is None else round(velocity[1], 2),
        "track_age": observation.track_age,
        "frames_since_detection": observation.frames_since_detection,
        "status": observation.status,
    }


@dataclass(frozen=True)
class TrackObservation:
    """Toestand van één track in één frame.

    bbox, center en confidence zijn alleen gevuld als het object in dit frame
    gedetecteerd is (status "confirmed" of "tentative"). Bij status "occluded"
    was het object tijdelijk niet zichtbaar en bevat predicted_bbox de door het
    Kalman-filter voorspelde positie. track_age en frames_since_detection
    gebruiken de bronframe-index, net als frame_number in de CSV.
    """

    track_id: int
    class_id: int | None
    confidence: float | None
    bbox: tuple[float, float, float, float] | None
    predicted_bbox: tuple[float, float, float, float]
    center: tuple[float, float] | None
    velocity: tuple[float, float] | None
    track_age: int
    frames_since_detection: int
    status: str


class _DetectionView:
    """Results-achtige view over Detection-objecten voor de ultralytics BYTETracker.

    De BYTETracker verwacht een object met `conf`, `cls` en `xywh` (pixels) dat
    boolean-indexering ondersteunt; deze klasse biedt dat op basis van onze
    eigen Detection-objecten.
    """

    def __init__(self, conf: np.ndarray, cls: np.ndarray,
                 xywh: np.ndarray) -> None:
        self.conf = conf
        self.cls = cls
        self.xywh = xywh

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> "_DetectionView":
        boxes = np.array([detection.bbox for detection in detections],
                         dtype=np.float32).reshape(-1, 4)
        xywh = np.empty((len(detections), 4), dtype=np.float32)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        return cls(
            conf=np.array([detection.confidence for detection in detections],
                          dtype=np.float32),
            cls=np.array([detection.class_id for detection in detections],
                         dtype=np.float32),
            xywh=xywh,
        )

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "_DetectionView":
        return _DetectionView(self.conf[index], self.cls[index],
                              self.xywh[index])


class ObjectTracker:
    """ByteTrack-wrapper die per track de beweging en geschiedenis bijhoudt."""

    def __init__(self, cfg: dict) -> None:
        # Pas importeren na een eventuele CUDA-torch-installatie, anders draait
        # een reeds geladen CPU-build.
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(types.SimpleNamespace(
            track_high_thresh=cfg.get("track_high_thresh", 0.25),
            track_low_thresh=cfg.get("track_low_thresh", 0.1),
            new_track_thresh=cfg.get("new_track_thresh", 0.25),
            track_buffer=cfg.get("track_buffer", 30),
            match_thresh=cfg.get("match_thresh", 0.8),
            fuse_score=cfg.get("fuse_score", True),
        ))
        self.trajectory_length = cfg.get("track_trajectory_length", 60)
        self._detections: list[Detection] = []
        self._last_seen: dict[int, tuple[int, float, float]] = {}
        self._first_frame: dict[int, int] = {}
        self._last_detection_frame: dict[int, int] = {}
        self.trajectories: dict[int, deque[tuple[float, float]]] = {}

    def update(self, frame_number: int, frame: np.ndarray,
               detections: list[Detection]) -> list[TrackObservation]:
        """Koppelt de detecties van dit frame aan de bestaande tracks.

        De frame-pixels zijn voor ByteTrack niet nodig (associatie gebeurt op
        beweging en IoU); de parameter blijft zodat een tracker die de beelden
        wel gebruikt, zoals BoT-SORT, later uitwisselbaar is.
        """
        self._detections = detections
        self._tracker.update(_DetectionView.from_detections(detections))
        # tracked_stracks bevat de actieve tracks (bevestigd en net gestart),
        # lost_stracks de tracks binnen track_buffer die tijdelijk niet
        # gedetecteerd zijn. Beide zijn gedocumenteerde BYTETracker-attributen.
        observations = [self._observe(frame_number, track, visible=True)
                        for track in self._tracker.tracked_stracks]
        observations += [self._observe(frame_number, track, visible=False)
                         for track in self._tracker.lost_stracks]
        self._detections = []
        return observations

    def _observe(self, frame_number: int, track: "STrack",
                 visible: bool) -> TrackObservation:
        track_id = track.track_id
        predicted_bbox = tuple(float(value) for value in track.xyxy)
        bbox = center = confidence = velocity = None
        if visible:
            detection = self._matched_detection(track)
            # De originele detectie in plaats van de Kalman-schatting, zodat de
            # bbox gelijk blijft aan wat YOLO in dit frame zag.
            bbox = detection.bbox if detection is not None else predicted_bbox
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            confidence = float(track.score)
            previous = self._last_seen.get(track_id)
            if previous is not None:
                elapsed = frame_number - previous[0]
                if elapsed > 0:
                    velocity = ((center[0] - previous[1]) / elapsed,
                                (center[1] - previous[2]) / elapsed)
            self._last_seen[track_id] = (frame_number, center[0], center[1])
            self._first_frame.setdefault(track_id, frame_number)
            self._last_detection_frame[track_id] = frame_number
            self.trajectories.setdefault(
                track_id, deque(maxlen=self.trajectory_length)).append(center)
        if not visible:
            status = "occluded"
        elif track.is_activated:
            status = "confirmed"
        else:
            status = "tentative"
        return TrackObservation(
            track_id=track_id,
            class_id=int(track.cls),
            confidence=confidence,
            bbox=bbox,
            predicted_bbox=predicted_bbox,
            center=center,
            velocity=velocity,
            track_age=frame_number - self._first_frame.get(track_id,
                                                           frame_number),
            frames_since_detection=(
                frame_number - self._last_detection_frame.get(
                    track_id, frame_number)),
            status=status,
        )

    def _matched_detection(self, track: "STrack") -> Detection | None:
        """De detectie uit dit frame die aan de track gekoppeld is (track.idx)."""
        index = int(track.idx)
        if 0 <= index < len(self._detections):
            return self._detections[index]
        return None
