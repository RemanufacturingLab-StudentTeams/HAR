"""VITRA (Vision-Language-Action) op een video.

Per gesampled frame voorspelt VITRA een chunk toekomstige handacties op basis
van een tekstinstructie (zero-shot, geen finetuning). De resultaten gaan naar
``vitra_actions.csv``; de debug-video toont de instructie, de voorspelde
translatiebaan en een samenvatting per hand.

Met ``vitra_reconstruct_hands: true`` draait eerst een HaWoR + MoGe + MANO-pass
die per frame een echte handstate maakt (``vitra_hand_states.npz``). Daarna
laadt de tweede pass VITRA en gebruikt die states. De twee passes zijn
gescheiden omdat HaWoR/MoGe/MANO en het 3B-model samen niet in 16 GB VRAM
passen.

VITRA is getraind met transformers 4.47.1. Onder transformers 5.x kloppen de
PaliGemma-hidden-states niet meer en voorspelt de actiekop 10-100x te grote
acties (geverifieerd met een A/B-test tegen de officiele stack). Daarom start
dit script zichzelf opnieuw in een subprocess met ``third_party/vitra_runtime``
op de PYTHONPATH; die map bevat alleen de gepinde transformers-versie en laat
de hoofd-venv (transformers 5.x voor SAM3) ongemoeid.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from duplo.benchmark import InferenceTimer
from duplo.cli import parse_args
from duplo.output import save_screenshot
from duplo.pipeline import (finish_detection_run, load_step_config,
                            open_debug_writer, setup_runtime,
                            should_screenshot, video_params)
from duplo.schema import VITRA_COLUMNS, vitra_action_rows
from duplo.video import VideoFrames
from duplo.vitra import STATE_DIM, VitraPredictor, annotate_vitra
from duplo.vitra_state import (FRAME_STATES_FILE, load_frame_states,
                               reconstruct_video_states)

DEFAULT_RUNTIME_DIR = "third_party/vitra_runtime"
ISOLATED_ENV = "DUPLO_VITRA_ISOLATED"


def _transformers_major() -> int | None:
    try:
        import transformers
    except ImportError:
        return None
    try:
        return int(transformers.__version__.split(".")[0])
    except ValueError:
        return None


def ensure_pinned_transformers(cfg: dict) -> None:
    """Start de stap opnieuw met de gepinde VITRA-runtime (transformers 4.47).

    Een kleinere transformersversie in de venv zelf heeft geen isolatie
    nodig; anders is ``third_party/vitra_runtime`` vereist.
    """
    if os.environ.get(ISOLATED_ENV) == "1":
        return
    major = _transformers_major()
    if major is None or major < 5:
        return
    runtime = Path(str(cfg.get("vitra_runtime_path", DEFAULT_RUNTIME_DIR)))
    if not runtime.is_dir():
        raise SystemExit(
            "VITRA vereist transformers 4.47.1, maar de venv heeft een 5.x-"
            "versie.\nInstalleer de aparte runtime (eenmalig):\n"
            "  .venv\\Scripts\\python.exe -m pip install --no-deps --target "
            f"{runtime} -r requirements-vitra.txt\n"
            "Daarna start run_vitra_test.bat de VITRA-stap automatisch in een "
            "subprocess met die runtime.")
    env = os.environ.copy()
    env[ISOLATED_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(runtime.resolve()), env["PYTHONPATH"]]
    ) if env.get("PYTHONPATH") else str(runtime.resolve())
    print(f"VITRA draait in een subprocess met {runtime} "
          "(transformers 4.47.1).")
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, env=env))


def main(cfg_path: str, device: str | None = None) -> None:
    cfg, out_dir = load_step_config(cfg_path, "vitra")
    ensure_pinned_transformers(cfg)
    stride = int(cfg.get("vitra_frame_stride", cfg.get("frame_stride", 1)))
    if stride < 1:
        raise SystemExit("vitra_frame_stride moet minimaal 1 zijn.")
    cfg = {**cfg, "frame_stride": stride}
    device = setup_runtime(cfg, device)

    states_path = cfg.get("vitra_states_path")
    existing_states = out_dir / FRAME_STATES_FILE
    if (not states_path and not cfg.get("vitra_state_path")
            and cfg.get("vitra_reuse_states", True)
            and existing_states.is_file()):
        states_path = str(existing_states)
        print(f"Bestaande handstates hergebruikt: {states_path} "
              "(zet vitra_reuse_states: false voor een nieuwe pass 1)")
    if (not states_path and not cfg.get("vitra_state_path")
            and cfg.get("vitra_reconstruct_hands", False)):
        print("Pass 1/2: HaWoR + MoGe + MANO-handstates maken...")
        states_path = str(reconstruct_video_states(cfg, out_dir, device))
    if states_path:
        cfg = {**cfg, "vitra_states_path": states_path}
    frame_states = load_frame_states(states_path) if states_path else None
    if frame_states is not None:
        print(f"Pass 2/2: VITRA met {len(frame_states)} per-frame states")
    elif cfg.get("vitra_state_path"):
        print("VITRA met één statische handstate")
    else:
        print("Waarschuwing: geen MANO-state of reconstructie; de nul-state "
              "geeft grove voorspellingen.")

    predictor = VitraPredictor(cfg, device)
    print(f"VITRA ({predictor.model_id}) draait op GPU")
    print(f"Instructie: {predictor.instruction}")
    print(f"Handen: {', '.join(predictor.hands)} | "
          f"chunk={predictor.chunk_size} | state={predictor.state_source}")

    fps, _stride, screenshot_every = video_params(cfg)
    timer = InferenceTimer()
    # Eén VITRA-voorspelling per seconde levert anders een video van 1 fps.
    # Standaard wordt elk voorspelframe herhaald tot de bronfps, zodat de
    # debug-video normaal afspeelt en de overlay per seconde bijwerkt.
    repeat_frames = bool(cfg.get("vitra_debug_repeat_frames", True))
    video = open_debug_writer(
        out_dir, "vitra_debug.mp4", fps, 1 if repeat_frames else stride, cfg,
        "save_vitra_video")
    rows: list[dict] = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="VITRA")
    started = time.perf_counter()
    warned_missing_state = False
    try:
        for frame_idx, frame in frames:
            state = fov_x = None
            valid_hands: set[str] = set()
            if frame_states is not None:
                entry = frame_states.get(frame_idx)
                if entry is None:
                    if not warned_missing_state:
                        print("Waarschuwing: geen handstate voor sommige "
                              "frames; daar wordt de nul-state gebruikt.")
                        warned_missing_state = True
                    state = np.zeros(STATE_DIM, dtype=np.float32)
                else:
                    state, fov_x, valid_hands = entry
            with timer:
                actions = predictor.predict(frame, state=state, fov_x=fov_x,
                                            state_hands=valid_hands)
            rows.extend(vitra_action_rows(
                cfg["video_path"], frame_idx, fps, actions, predictor.hands))
            need_video = video is not None
            need_shot = should_screenshot(screenshot_every, frame_idx)
            if need_video or need_shot:
                annotated = annotate_vitra(
                    frame, actions, predictor.instruction, predictor.hands,
                    frame_idx=frame_idx, fps=fps)
                if need_video:
                    for _ in range(stride if repeat_frames else 1):
                        video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, "vitra_frame", frame_idx,
                                    annotated)
    finally:
        if video is not None:
            video.release()

    extra = predictor.describe()
    extra["reconstruct_hands"] = bool(cfg.get("vitra_reconstruct_hands"))
    finish_detection_run(
        out_dir, out_dir.name, model=predictor.model_id, device=device,
        video_path=cfg["video_path"], frames=frames, started=started,
        rows=rows, timer=timer, csv_file="vitra_actions.csv",
        columns=VITRA_COLUMNS, class_field="hand", extra=extra)


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)

