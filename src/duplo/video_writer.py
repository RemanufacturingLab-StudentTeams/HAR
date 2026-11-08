"""Video-encoding voor debug-video's in een breed afspeelbaar formaat.

OpenCV's ingebouwde mp4v-codec (MPEG-4 Part 2) speelt niet af in Chromium
(de VS Code-videospeler en browsers) en wordt door deelplatforms geweigerd.
Deze writer encodeert daarom H.264 (yuv420p) met faststart, via de
ffmpeg-binary die met imageio-ffmpeg wordt meegeleverd.

Op een NVIDIA-GPU (zoals de RTX 5080) wordt bij voorkeur de hardware-encoder
``h264_nvenc`` gebruikt: die ontlast de CPU en verhoogt de pipeline-doorvoer,
zonder dat de detecties (CSV) veranderen. Alleen de debug-mp4 wordt anders
gecodeerd. Bij ontbrekende NVENC volgt automatisch een fallback naar
libx264 en daarna naar mp4v, zodat de pipeline altijd een video oplevert.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

_ENCODER_CACHE: dict[str, bool] = {}


class DebugVideoWriter:
    """Schrijft een H.264-mp4 die in VS Code, browsers en deelplatforms werkt.

    Valt terug op OpenCV's mp4v als de ffmpeg-binary ontbreekt, zodat de
    pipeline altijd een debug-video oplevert. Zonder size worden de
    afmetingen bij het eerste frame bepaald, zodat de pipelines de writer
    niet vooraf hoeven te openen.

    ``encoder`` kiest de ffmpeg-encoder: "auto" (nvenc indien aanwezig,
    anders libx264), "nvenc", "x264" of "mp4v". Alleen de codering van de
    debug-video verandert; de detectieresultaten blijven identiek.
    """

    def __init__(self, path: Path, fps: float,
                 size: tuple[int, int] | None = None,
                 encoder: str = "auto") -> None:
        self._path = path
        self._fps = fps
        self._encoder = (encoder or "auto").lower()
        self._process = None
        self._writer = None
        if size is not None:
            self._open(size)

    def _open(self, size: tuple[int, int]) -> None:
        ffmpeg = self._find_ffmpeg()
        if ffmpeg is not None and self._encoder != "mp4v":
            codec_args = self._pick_codec_args(ffmpeg)
            if codec_args is not None:
                width, height = size
                self._process = subprocess.Popen(
                    [ffmpeg, "-y", "-loglevel", "error",
                     "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{width}x{height}", "-r", str(self._fps), "-i", "-",
                     "-an", *codec_args,
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     str(self._path)],
                    stdin=subprocess.PIPE,
                )
                return
        if self._process is None:
            if ffmpeg is None:
                print("Waarschuwing: imageio-ffmpeg ontbreekt; de debug-video "
                      "krijgt het mp4v-formaat dat niet overal afspeelt.")
            self._writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps,
                size)

    def _pick_codec_args(self, ffmpeg: str) -> list[str] | None:
        """Codec-argumenten voor ffmpeg; None bij expliciet mp4v-verzoek."""
        want_nvenc = self._encoder in ("auto", "nvenc")
        if want_nvenc and self._has_encoder(ffmpeg, "h264_nvenc"):
            if self._encoder == "auto":
                print("Debug-video encodeert met NVENC (h264_nvenc).")
            # p1 = snelste Blackwell-preset; cq 21 ≈ crf 21 voor debug-kwaliteit.
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "21"]
        if self._encoder == "nvenc":
            print("Waarschuwing: h264_nvenc niet gevonden; val terug op libx264.")
        if self._has_encoder(ffmpeg, "libx264"):
            return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
        return None

    @classmethod
    def _has_encoder(cls, ffmpeg: str, name: str) -> bool:
        if name not in _ENCODER_CACHE:
            try:
                probe = subprocess.run(
                    [ffmpeg, "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=15,
                )
                _ENCODER_CACHE[name] = (
                    probe.returncode == 0 and name in probe.stdout)
            except (OSError, subprocess.SubprocessError):
                _ENCODER_CACHE[name] = False
        return _ENCODER_CACHE[name]

    @staticmethod
    def _find_ffmpeg() -> str | None:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return None

    def write(self, frame: np.ndarray) -> None:
        if self._process is None and self._writer is None:
            height, width = frame.shape[:2]
            self._open((width, height))
        if self._process is not None:
            self._process.stdin.write(frame.tobytes())
        else:
            self._writer.write(frame)

    def release(self) -> None:
        if self._process is not None:
            self._process.stdin.close()
            self._process.wait()
            if self._process.returncode != 0:
                print(f"Waarschuwing: ffmpeg stopte met code "
                      f"{self._process.returncode}; de debug-video is "
                      f"mogelijk incompleet.")
            self._process = None
        elif self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "DebugVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
