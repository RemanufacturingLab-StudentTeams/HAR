"""HaWoR + MoGe + MANO-handstates voor VITRA.

De officiële VITRA-handreconstructie zit in ``data/tools/hand_recon_core.py``
van de microsoft/VITRA-repo. Deze module start die reconstructie, zet de
uitkomst om naar de 122-dimensionale state die VITRA verwacht en bewaart per
gesampled videoframe een state in een ``.npz``-bestand.

De reconstructie en VITRA-inferentie draaien bewust in twee passes: op een
16 GB-GPU passen HaWoR/MoGe/MANO en het 3B-VITRA-model niet tegelijk. De
eerste pass schrijft alleen states; de tweede pass laadt VITRA en leest die
states weer in.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from duplo.vitra import DEFAULT_REPO_PATH, STATE_DIM, parse_hands, \
    state_from_hand_data

DEFAULT_HAWOR_CHECKPOINT = "weights/hawor/checkpoints/hawor.ckpt"
DEFAULT_HAWOR_DETECTOR = "weights/hawor/external/detector.pt"
DEFAULT_MANO_DIR = "weights/mano"
DEFAULT_MOGE_MODEL = "Ruicheng/moge-2-vitl"
FRAME_STATES_FILE = "vitra_hand_states.npz"

MANO_HELP = (
    "MANO_RIGHT.pkl ontbreekt in {mano_dir}.\n"
    "Download het MANO-model (registratie en licentie via "
    "https://mano.is.tue.mpg.de/; gratis voor niet-commercieel onderzoek) "
    "en plaats:\n"
    "  {mano_right}\n"
    "  {mano_mean}\n"
    "Het bestand mano_mean_params.npz zit al in de VITRA-repo; "
    "MANO_RIGHT.pkl niet vanwege de MANO-licentie."
)


def resolve_hand_recon_paths(cfg: dict) -> dict:
    """Paden voor HaWoR, detector, MANO en MoGe uit de config."""
    repo = Path(str(cfg.get("vitra_repo_path", DEFAULT_REPO_PATH)))
    mano_dir = Path(str(cfg.get("vitra_mano_path",
                                repo / DEFAULT_MANO_DIR)))
    return {
        "repo": repo,
        "hawor": Path(str(cfg.get("vitra_hawor_model_path",
                                  repo / DEFAULT_HAWOR_CHECKPOINT))),
        "detector": Path(str(cfg.get("vitra_hawor_detector_path",
                                     repo / DEFAULT_HAWOR_DETECTOR))),
        "mano_dir": mano_dir,
        "moge_model": str(cfg.get("vitra_moge_model_name",
                                  DEFAULT_MOGE_MODEL)),
    }


def check_hand_recon_assets(paths: dict) -> None:
    """Geef een gerichte melding voordat zware modellen worden geladen."""
    mano_right = paths["mano_dir"] / "MANO_RIGHT.pkl"
    mano_mean = paths["mano_dir"] / "mano_mean_params.npz"
    if not mano_right.is_file() or not mano_mean.is_file():
        raise SystemExit(MANO_HELP.format(
            mano_dir=paths["mano_dir"], mano_right=mano_right,
            mano_mean=mano_mean))
    if not paths["hawor"].is_file():
        raise SystemExit(
            f"HaWoR-checkpoint ontbreekt: {paths['hawor']}.\n"
            "Download het met:\n"
            "  curl -L https://huggingface.co/ThunderVVV/HaWoR/resolve/"
            "main/hawor/checkpoints/hawor.ckpt -o "
            f"\"{paths['hawor']}\"")
    if not paths["detector"].is_file():
        raise SystemExit(
            f"HaWoR-handdetector ontbreekt: {paths['detector']}.\n"
            "Download hem met:\n"
            "  curl -L https://huggingface.co/spaces/rolpotamias/WiLoR/"
            "resolve/main/pretrained_models/detector.pt -o "
            f"\"{paths['detector']}\"")


def add_vitra_paths(repo_path: str | Path) -> None:
    """Zet de VITRA- en HaWoR-repo op ``sys.path`` voor de officiële imports."""
    root = Path(str(repo_path)).resolve()
    for path in (root, root / "thirdparty" / "HaWoR"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_hawor_compat(checkpoint_path: str, mano_dir: Path):
    """VITRA's ``load_hawor`` met absolute MANO-paden en torch.load-compat."""
    from thirdparty.HaWoR.hawor.configs import get_config
    from thirdparty.HaWoR.lib.models.hawor import HAWOR

    checkpoint = Path(checkpoint_path)
    model_cfg = get_config(
        str(checkpoint.parent.parent / "model_config.yaml"),
        update_cachedir=True)
    model_cfg.defrost()
    model_cfg.MANO.DATA_DIR = str(mano_dir)
    model_cfg.MANO.MODEL_PATH = str(mano_dir)
    model_cfg.MANO.MEAN_PARAMS = str(mano_dir / "mano_mean_params.npz")
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit"
            and "BBOX_SHAPE" not in model_cfg.MODEL):
        if model_cfg.MODEL.IMAGE_SIZE != 256:
            raise SystemExit(
                "HaWoR ViT-config verwacht MODEL.IMAGE_SIZE=256, kreeg "
                f"{model_cfg.MODEL.IMAGE_SIZE}.")
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
    model_cfg.freeze()
    # Het checkpoint bevat omegaconf-objecten; het is afkomstig van de
    # officiële HaWoR-repo en kan daarom niet weights_only worden geladen.
    model = HAWOR.load_from_checkpoint(
        str(checkpoint), strict=False, cfg=model_cfg, weights_only=False)
    return model, model_cfg


class HandStateReconstructor:
    """Draait HaWoR + MoGe + MANO voor één frame en levert een VITRA-state."""

    def __init__(self, cfg: dict, device: str = "gpu") -> None:
        if device != "gpu":
            raise SystemExit(
                "De HaWoR/MANO-reconstructie vereist een CUDA-GPU; "
                "gebruik --device gpu.")
        self.hands = parse_hands(cfg.get("vitra_hands", ["right"]))
        self.paths = resolve_hand_recon_paths(cfg)
        check_hand_recon_assets(self.paths)
        import torch

        self._torch = torch
        add_vitra_paths(self.paths["repo"])
        args = SimpleNamespace(
            hawor_model_path=str(self.paths["hawor"]),
            detector_path=str(self.paths["detector"]),
            moge_model_name=self.paths["moge_model"],
            mano_path=str(self.paths["mano_dir"]),
        )
        try:
            import data.tools.utils_hawor as hawor_utils
            from data.tools.hand_recon_core import Config, HandReconstructor

            hawor_utils.load_hawor = lambda path: _load_hawor_compat(
                path, self.paths["mano_dir"])
            self._reconstructor = HandReconstructor(
                Config(args), device=torch.device("cuda"))
        except ImportError as exc:
            raise SystemExit(
                "HaWoR/MANO-dependency ontbreekt: "
                f"{exc}. Installeer de VITRA-handreconstructie-dependencies "
                "uit requirements.txt; chumpy installeer je eenmalig met "
                "'pip install --no-build-isolation --no-deps "
                "chumpy@git+https://github.com/mattloper/chumpy'."
            ) from exc

    def reconstruct(self, frame: np.ndarray
                    ) -> tuple[np.ndarray, float | None, set[str]]:
        data = self._reconstructor.recon([frame])
        return state_from_hand_data(data, self.hands)

    def close(self) -> None:
        self._reconstructor = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def save_frame_states(path: str | Path,
                      states: dict[int, tuple[np.ndarray, float | None,
                                              set[str]]]) -> None:
    """Schrijf per-frame states als gecomprimeerde ``.npz``."""
    path = Path(path)
    frames = np.array(sorted(states), dtype=np.int64)
    if len(frames):
        matrix = np.stack([states[int(frame)][0] for frame in frames])
        fovs = np.array([
            np.nan if states[int(frame)][1] is None
            else float(states[int(frame)][1]) for frame in frames
        ], dtype=np.float32)
        valid = np.array([
            [hand in states[int(frame)][2] for hand in ("left", "right")]
            for frame in frames
        ], dtype=bool)
    else:
        matrix = np.zeros((0, STATE_DIM), dtype=np.float32)
        fovs = np.zeros((0,), dtype=np.float32)
        valid = np.zeros((0, 2), dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=frames, states=matrix, fov_x=fovs,
                        valid=valid)


def load_frame_states(path: str | Path
                      ) -> dict[int, tuple[np.ndarray, float | None, set[str]]]:
    """Lees de per-frame states terug."""
    state_path = Path(path)
    if not state_path.is_file():
        raise SystemExit(f"VITRA-statesbestand niet gevonden: {state_path}")
    with np.load(state_path) as data:
        frames = data["frames"]
        matrix = data["states"]
        fovs = data["fov_x"]
        valid = data["valid"]
    if matrix.ndim != 2 or matrix.shape[1] != STATE_DIM:
        raise SystemExit(
            f"VITRA-statesbestand heeft onverwachte state-shape "
            f"{matrix.shape} (verwacht [N, {STATE_DIM}]).")
    states = {}
    for index, frame in enumerate(frames):
        hands = {hand for hand, flag in zip(("left", "right"), valid[index])
                 if flag}
        fov = None if np.isnan(fovs[index]) else float(fovs[index])
        states[int(frame)] = (matrix[index].astype(np.float32), fov, hands)
    return states


def reconstruct_video_states(cfg: dict, out_dir: str | Path,
                             device: str = "gpu",
                             reconstructor=None) -> Path:
    """Eerste pass: HaWoR/MANO-states voor elk gesampled videoframe."""
    from duplo.video import VideoFrames

    stride = int(cfg.get("vitra_frame_stride", cfg.get("frame_stride", 1)))
    if stride < 1:
        raise SystemExit("vitra_frame_stride moet minimaal 1 zijn.")
    output_path = Path(out_dir) / FRAME_STATES_FILE
    owns_reconstructor = reconstructor is None
    reconstructor = reconstructor or HandStateReconstructor(cfg, device)
    states: dict[int, tuple[np.ndarray, float | None, set[str]]] = {}
    try:
        frames = VideoFrames(cfg["video_path"], stride=stride,
                             desc="HaWoR/MANO")
        for frame_idx, frame in frames:
            state, fov_x, valid = reconstructor.reconstruct(frame)
            states[frame_idx] = (state, fov_x, valid)
    finally:
        if owns_reconstructor:
            reconstructor.close()
    save_frame_states(output_path, states)
    print(f"Handstates voor {len(states)} frames -> {output_path}")
    return output_path
