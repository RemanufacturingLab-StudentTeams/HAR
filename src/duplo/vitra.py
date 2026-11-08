"""VITRA (Vision-Language-Action) als zero-shot handactie-voorspeller.

VITRA is geen detectiemodel: het krijgt één egocentrisch frame plus een
tekstinstructie en voorspelt een chunk toekomstige MANO-handacties
(translatie, globale rotatie en 45 gewrichtshoeken per hand). Deze module
houdt de optionele VITRA-afhankelijkheid buiten de rest van het project:

* de officiële repo wordt via ``sys.path`` geladen (``vitra_repo_path``),
  zodat de training-only dependencies (decord, deepspeed, ...) nooit in de
  project-venv terechtkomen;
* een kleine shim vertaalt de PaliGemma-API van transformers 5.x naar de
  4.47-aanroepen die VITRA gebruikt (het project pint transformers 5.x voor
  SAM3);
* checkpoint-keys van de 4.47-release worden omgezet naar de 5.x-moduleboom.

Rijbouw en annotatie blijven import-licht, zodat ze zonder torch of
modelgewichten te testen zijn.
"""
from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

DEFAULT_REPO_PATH = "third_party/VITRA"
DEFAULT_CONFIG_PATH = "VITRA-VLA/VITRA-VLA-3B"
DEFAULT_INSTRUCTION = (
    "Left hand: None. Right hand: Pick up the tool and disassemble the "
    "DUPLO model."
)
HAND_SLICES = {"left": slice(0, 51), "right": slice(51, 102)}
HAND_COLORS = {"left": (255, 200, 0), "right": (0, 255, 0)}
STATE_DIM = 122
ACTION_DIM = 102


def parse_hands(value) -> tuple[str, ...]:
    """Normaliseer ``vitra_hands`` naar een unieke tuple links/rechts."""
    if value is None:
        value = ["right"]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        raise SystemExit("vitra_hands moet 'left', 'right' of een lijst zijn.")
    hands: list[str] = []
    for hand in value:
        name = str(hand).strip().lower()
        if name not in HAND_SLICES:
            raise SystemExit(
                f"Onbekende hand in vitra_hands: {hand!r} "
                "(gebruik 'left' en/of 'right').")
        if name not in hands:
            hands.append(name)
    if not hands:
        raise SystemExit("vitra_hands is leeg; kies 'left' en/of 'right'.")
    return tuple(hands)


def resolve_model_path(value) -> str | None:
    """Lokaal checkpointbestand, map met één .pt-bestand of HF-repo-id."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value)
    path = Path(text)
    if path.is_dir():
        for name in ("weights.pt", "vitra-vla-3b.pt"):
            candidate = path / name
            if candidate.is_file():
                return str(candidate)
        checkpoints = sorted(path.glob("*.pt"))
        if len(checkpoints) == 1:
            return str(checkpoints[0])
        raise SystemExit(
            f"Geen eenduidig VITRA-checkpoint in {path}; zet vitra_model_path "
            "op het .pt-bestand zelf.")
    return text


def resolve_statistics_path(value) -> str | None:
    """Lokaal statistics-bestand of map met dataset_statistics.json."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value)
    path = Path(text)
    if path.is_dir():
        for candidate in (path / "dataset_statistics.json",
                          path / "statistics" / "dataset_statistics.json",
                          path / "statistics.json"):
            if candidate.is_file():
                return str(candidate)
        raise SystemExit(
            f"Geen dataset_statistics.json gevonden in {path}.")
    return text


def state_from_hand_data(data: dict, hands: Sequence[str],
                         warn_missing: bool = False
                         ) -> tuple[np.ndarray, float | None, set[str]]:
    """[122]-state uit VITRA's ``{left: {0: ...}, right: ..., fov_x}``-dict.

    Handen zonder entry (of met een lege dict, zoals de linkerhand in
    ``examples/0002.npy``) blijven op nullen staan en komen niet in de
    teruggegeven set; zo weet de caller dat er geen echte state is.
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    valid: set[str] = set()
    for hand in hands:
        entry = data.get(hand)
        if not isinstance(entry, dict) or 0 not in entry:
            continue
        offset = 0 if hand == "left" else 61
        entry = entry[0]
        state[offset:offset + 51] = _hand_state_vector(entry, hand)
        state[offset + 51:offset + 61] = _beta_vector(entry, hand)
        valid.add(hand)
    if warn_missing:
        missing = [hand for hand in hands if hand not in valid]
        if missing:
            print(f"Waarschuwing: het statebestand bevat geen bruikbare state "
                  f"voor {', '.join(missing)}; die hand wordt als onbekend "
                  "doorgegeven aan VITRA.")
    fov_x = data.get("fov_x")
    return state, (float(fov_x) if fov_x is not None else None), valid


def load_hand_state(path, hands: Sequence[str]
                    ) -> tuple[np.ndarray, float | None, set[str]]:
    """Lees een VITRA hand-state (.npy) of geef de nul-state terug.

    Het formaat is dat van ``scripts/inference_human_prediction.py
    --save_state_local``: een dict met ``left``/``right`` (MANO-parameters)
    en ``fov_x``. Zonder bestand gebruikt de stap de nul-state; de
    teruggegeven set blijft dan leeg, zodat VITRA de handstate als onbekend
    krijgt in plaats van als een geldige neutrale pose.

    ``np.load`` moet pickles toestaan voor dit dict-formaat; laad daarom
    alleen state-bestanden die je zelf hebt aangemaakt.
    """
    if path is None or str(path).strip() == "":
        return np.zeros(STATE_DIM, dtype=np.float32), None, set()
    state_path = Path(str(path))
    if not state_path.is_file():
        raise SystemExit(f"VITRA-statebestand niet gevonden: {state_path}")
    try:
        loaded = np.load(state_path, allow_pickle=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"Kon het VITRA-statebestand niet lezen: {state_path} ({exc})"
        ) from exc
    data = loaded.item() if loaded.ndim == 0 else loaded
    if not isinstance(data, dict):
        vector = np.asarray(data, dtype=np.float32).reshape(-1)
        if vector.shape != (STATE_DIM,):
            raise SystemExit(
                f"VITRA-state moet {STATE_DIM} waarden bevatten, kreeg "
                f"{vector.shape}.")
        # Een expliciete 122-vector geldt voor alle gevraagde handen.
        return vector, None, set(hands)
    return state_from_hand_data(data, hands, warn_missing=True)


def _hand_state_vector(entry: dict, hand: str) -> np.ndarray:
    """[transl(3), global_orient-euler(3), hand_pose-euler(45)] uit MANO."""
    from scipy.spatial.transform import Rotation

    try:
        transl = np.asarray(entry["transl"], dtype=np.float32).reshape(-1)
        global_orient = np.asarray(entry["global_orient"], dtype=np.float32)
        hand_pose = np.asarray(entry["hand_pose"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"VITRA-state voor {hand} mist transl/global_orient/hand_pose."
        ) from exc
    if transl.shape != (3,):
        raise SystemExit(f"VITRA-state transl voor {hand} moet (3,) zijn.")
    if global_orient.shape != (3, 3):
        raise SystemExit(
            f"VITRA-state global_orient voor {hand} moet (3, 3) zijn.")
    if hand_pose.shape != (15, 3, 3):
        raise SystemExit(
            f"VITRA-state hand_pose voor {hand} moet (15, 3, 3) zijn.")
    global_euler = Rotation.from_matrix(global_orient).as_euler("xyz")
    pose_euler = Rotation.from_matrix(hand_pose).as_euler("xyz").reshape(-1)
    return np.concatenate([transl, global_euler, pose_euler]).astype(np.float32)


def _beta_vector(entry: dict, hand: str) -> np.ndarray:
    beta = np.asarray(entry.get("beta", np.zeros(10)),
                      dtype=np.float32).reshape(-1)
    if beta.shape != (10,):
        raise SystemExit(f"VITRA-state beta voor {hand} moet (10,) zijn.")
    return beta


def pad_state_human(state: np.ndarray, state_mask: np.ndarray, action_dim: int,
                    state_dim: int, unified_state_dim: int):
    """VITRA-padding: per-hand state naar [unified_state_dim] + mask.

    Zelfde contract als ``vitra.datasets.human_dataset.pad_state_human``,
    maar zonder de training-datasetimports (decord/utils3d) te laden.
    """
    import torch

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    mask = np.asarray(state_mask, dtype=bool).reshape(-1)
    if state.shape != (state_dim,) or mask.shape != (2,):
        raise ValueError(
            f"state moet ({state_dim},) en mask (2,) zijn, kreeg "
            f"{state.shape} en {mask.shape}.")
    if state_dim % 2 or action_dim % 2 or unified_state_dim < action_dim:
        raise ValueError("Ongeldige state-/actiedimensies voor VITRA-padding.")
    hand_dim = state_dim // 2
    half = action_dim // 2
    expanded = np.repeat(mask, hand_dim)
    masked = state * expanded
    padded = np.zeros(unified_state_dim, dtype=np.float32)
    padded_mask = np.zeros(unified_state_dim, dtype=bool)
    padded[:half] = masked[:half]
    padded_mask[:half] = expanded[:half]
    padded[half:action_dim] = masked[hand_dim:hand_dim + half]
    padded_mask[half:action_dim] = expanded[hand_dim:hand_dim + half]
    return torch.from_numpy(padded), torch.from_numpy(padded_mask)


def pad_action_mask(action_mask: np.ndarray, action_dim: int,
                    unified_action_dim: int):
    """VITRA-padding: per-hand actiemasker naar [T, unified_action_dim]."""
    import torch

    mask = np.asarray(action_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != 2:
        raise ValueError(f"action_mask moet (T, 2) zijn, kreeg {mask.shape}.")
    if action_dim % 2 or unified_action_dim < action_dim:
        raise ValueError("Ongeldige actiedimensies voor VITRA-padding.")
    half = action_dim // 2
    expanded = np.concatenate([
        np.repeat(mask[:, :1], half, axis=1),
        np.repeat(mask[:, 1:], half, axis=1),
    ], axis=1)
    padded = np.zeros((mask.shape[0], unified_action_dim), dtype=bool)
    padded[:, :action_dim] = expanded
    return torch.from_numpy(padded)


def _as_action_chunk(actions) -> np.ndarray:
    """[T, 102] uit [T, 102] of [samples, T, 102] (eerste sample)."""
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < ACTION_DIM:
        raise ValueError(
            f"acties moeten (T, {ACTION_DIM}) of (S, T, {ACTION_DIM}) zijn, "
            f"kreeg {array.shape}.")
    return array


def _direction_label(vector) -> str:
    """Compacte richting in cameracoordinaten (x rechts, y omlaag, z weg)."""
    x, y, z = (float(value) for value in vector)
    parts = []
    if abs(x) >= 0.02:
        parts.append("rechts" if x > 0 else "links")
    if abs(y) >= 0.02:
        parts.append("omlaag" if y > 0 else "omhoog")
    if abs(z) >= 0.02:
        parts.append("van je af" if z > 0 else "naar je toe")
    return "+".join(parts) if parts else "stil"


def action_summary(actions, hands: Sequence[str]) -> dict[str, dict[str, float]]:
    """Samenvatting per hand voor de debug-overlay en tests."""
    chunk = _as_action_chunk(actions)
    summary: dict[str, dict[str, float]] = {}
    for hand in hands:
        part = chunk[:, HAND_SLICES[hand]]
        if not len(part):
            continue
        first = part[0]
        displacement = part[:, :3].sum(axis=0)
        summary[hand] = {
            "transl_norm": float(np.linalg.norm(first[:3])),
            "pose_mean": float(np.mean(np.abs(first[6:51]))),
            "path_length": float(np.sum(np.linalg.norm(part[:, :3], axis=1))),
            "displacement": float(np.linalg.norm(displacement)),
            "direction": _direction_label(displacement),
            "pose_mean_chunk": float(np.mean(np.abs(part[:, 6:51]))),
        }
    return summary


def _wrap_text(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_panel(frame, lines: list[str]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, pad, margin = 0.5, 1, 8, 10
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    line_height = max(height for _, height in sizes) + 6
    width = max(width for width, _ in sizes) + 2 * pad
    width = min(width, max(1, frame.shape[1] - 2 * margin))
    height = line_height * len(lines) + 2 * pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, margin),
                  (margin + width, margin + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, line in enumerate(lines):
        y = margin + pad + line_height * (index + 1) - 6
        cv2.putText(frame, line, (margin + pad, y), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_trajectory(frame, actions, hands: Sequence[str]) -> None:
    """Projecteer de voorspelde translatie (x/y in cameraruimte) op het beeld."""
    try:
        chunk = _as_action_chunk(actions)
    except (ValueError, TypeError):
        return
    scale = 400.0
    height, width = frame.shape[:2]
    for hand in hands:
        part = chunk[:, HAND_SLICES[hand]][:, :3]
        if len(part) < 2:
            continue
        camera_xy = np.cumsum(part, axis=0)[:, :2]
        offset = -90 if hand == "left" else 90
        origin = np.array([width / 2 + offset, height - 40], dtype=np.float32)
        points = origin + camera_xy * scale
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        points = points.astype(np.int32)
        color = HAND_COLORS[hand]
        cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(points[-1][0]), int(points[-1][1])), 4,
                   color, -1)
        cv2.putText(frame, hand[0].upper(),
                    (int(origin[0]) - 5, int(origin[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def annotate_vitra(frame, actions, instruction: str = "",
                   hands: Sequence[str] = ("right",),
                   frame_idx: int | None = None, fps: float | None = None):
    """Debugframe met invoerinstructie, frametijd en voorspelde handbeweging.

    De eerste regel is de *instructie* die VITRA als input krijgt (statisch);
    de regels per hand zijn de voorspelde beweging over de komende chunk.
    """
    annotated = frame.copy()
    wrap_width = max(24, min(74, frame.shape[1] // 7))
    lines = _wrap_text(f"VITRA: {instruction}".strip(), wrap_width)
    if frame_idx is not None or fps:
        parts = []
        if frame_idx is not None:
            parts.append(f"frame {frame_idx}")
        if frame_idx is not None and fps:
            parts.append(f"t={frame_idx / fps:.1f}s")
        parts.append("voorspelling per seconde")
        lines.append(" | ".join(parts))
    try:
        summary = action_summary(actions, hands)
    except (ValueError, TypeError):
        summary = {}
    for hand, info in summary.items():
        label = "L" if hand == "left" else "R"
        lines.append(
            f"{label}: {info['displacement']:.2f}m {info['direction']} "
            f"(stap 0: {info['transl_norm'] * 1000:.0f}mm, "
            f"gewrichten {info['pose_mean_chunk']:.2f} rad)")
    _draw_panel(annotated, lines)
    _draw_trajectory(annotated, actions, hands)
    return annotated


def _import_vitra(repo_path: str | Path):
    """Importeer ``vitra.models`` uit de repo of een bestaande installatie."""
    try:
        return importlib.import_module("vitra.models")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and not (missing == "vitra" or missing.startswith("vitra.")):
            raise SystemExit(
                f"VITRA is gevonden, maar dependency {missing!r} ontbreekt. "
                "Draai run_vitra_test.bat --upgrade."
            ) from exc
        path = Path(str(repo_path))
        if not path.is_dir():
            raise SystemExit(
                "De VITRA-repo is niet gevonden. Clone hem eenmalig:\n"
                "  git clone https://github.com/microsoft/VITRA.git "
                f"{path}\n"
                "Daarna regelt run_vitra_test.bat --upgrade de runtime-"
                "dependencies."
            ) from exc
        sys.path.insert(0, str(path.resolve()))
        try:
            return importlib.import_module("vitra.models")
        except ModuleNotFoundError as exc2:
            raise SystemExit(
                "VITRA kon niet worden geïmporteerd; ontbrekende module: "
                f"{exc2.name!r}. Draai run_vitra_test.bat --upgrade."
            ) from exc2
    except ImportError as exc:
        raise SystemExit(
            f"VITRA kon niet worden geïmporteerd: {exc}. Controleer de "
            "runtime-dependencies met run_vitra_test.bat --upgrade."
        ) from exc


def _load_model_compat(load_model, configs: dict, torch):
    """Laad VITRA met een expliciete weights_only-checkpointload."""
    import vitra.models.vla_builder as builder

    original = builder.load_vla_checkpoint

    def load_checkpoint(model, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=True)
        model.load_state_dict(checkpoint, strict=True)
        return model

    builder.load_vla_checkpoint = load_checkpoint
    try:
        return load_model(configs)
    finally:
        builder.load_vla_checkpoint = original


def _model_load_error(exc: Exception) -> str:
    return (
        "VITRA-model kon niet worden geladen. Controleer:\n"
        " 1) Hugging Face-toegang tot google/paligemma2-3b-mix-224 en "
        "'hf auth login';\n"
        " 2) schijfruimte voor het 3B-checkpoint (~12 GB in de HF-cache);\n"
        " 3) een NVIDIA-GPU met minimaal 16 GB VRAM.\n"
        f"Oorspronkelijke fout: {type(exc).__name__}: {exc}"
    )


class VitraPredictor:
    """Laadt VITRA één keer en voorspelt per frame een actiechunk."""

    def __init__(self, cfg: dict, device: str) -> None:
        if device != "gpu":
            raise SystemExit(
                "VITRA-inferentie vereist een CUDA-GPU (de modelcode zet "
                "inputs hard op 'cuda'); gebruik --device gpu.")
        import torch

        self._torch = torch
        self.hands = parse_hands(cfg.get("vitra_hands", ["right"]))
        self.instruction = str(
            cfg.get("vitra_instruction", DEFAULT_INSTRUCTION)).strip()
        if not self.instruction:
            raise SystemExit("vitra_instruction is leeg.")
        self.sample_times = int(cfg.get("vitra_sample_times", 1))
        self.num_ddim_steps = int(cfg.get("vitra_ddim_steps", 10))
        self.cfg_scale = float(cfg.get("vitra_cfg_scale", 5.0))
        self.target_short_side = int(cfg.get("vitra_image_size", 224))
        if (self.sample_times < 1 or self.num_ddim_steps < 1
                or self.cfg_scale <= 0 or self.target_short_side < 32):
            raise SystemExit("Ongeldige vitra_sample_times/ddim_steps/cfg_scale/"
                             "image_size in de config.")

        _import_vitra(cfg.get("vitra_repo_path", DEFAULT_REPO_PATH))
        from vitra.datasets.dataset_utils import ActionFeature, StateFeature
        from vitra.models import load_model
        from vitra.utils.config_utils import load_config
        from vitra.utils.data_utils import (load_normalizer,
                                            resize_short_side_to_target)

        self._resize = resize_short_side_to_target
        self.config_path = str(
            cfg.get("vitra_config_path") or DEFAULT_CONFIG_PATH)
        try:
            self.model_config = load_config(self.config_path)
        except Exception as exc:
            raise SystemExit(
                "VITRA-config kon niet worden geladen. Controleer "
                "vitra_config_path (HF-repo-id of lokaal config.json) en je "
                f"Hugging Face-login.\nOorspronkelijke fout: {exc}"
            ) from exc

        model_path = resolve_model_path(cfg.get("vitra_model_path"))
        statistics_path = resolve_statistics_path(
            cfg.get("vitra_statistics_path"))
        if model_path:
            self.model_config["model_load_path"] = model_path
        if statistics_path:
            self.model_config["statistics_path"] = statistics_path
        if not self.model_config.get("model_load_path"):
            raise SystemExit(
                "Geen VITRA-checkpoint opgegeven. Zet vitra_model_path op een "
                "lokaal .pt-bestand of op 'VITRA-VLA/VITRA-VLA-3B', of gebruik "
                "een config.json met model_load_path.")
        if not self.model_config.get("statistics_path"):
            raise SystemExit(
                "Geen VITRA-statistieken opgegeven. Zet vitra_statistics_path "
                "op dataset_statistics.json of gebruik de HF-config.")
        self.model_id = str(self.model_config["model_load_path"])

        try:
            self.model = _load_model_compat(
                load_model, self.model_config, torch).cuda()
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(_model_load_error(exc)) from exc
        self.model.eval()
        self.normalizer = load_normalizer(self.model_config)

        self.chunk_size = int(getattr(
            self.model, "chunk_size",
            self.model_config.get("fwd_pred_next_n", 16)))
        self.state_dim = int(self.normalizer.state_mean.shape[0])
        self.action_dim = int(self.normalizer.action_mean.shape[0])
        if (self.state_dim != STATE_DIM
                or self.action_dim != ACTION_DIM):
            raise SystemExit(
                "Deze stap ondersteunt alleen het human-hand VITRA-checkpoint "
                f"({STATE_DIM} state-, {ACTION_DIM} actiedimensies); kreeg "
                f"{self.state_dim} en {self.action_dim}.")
        self.unified_state_dim = int(StateFeature.ALL_FEATURES[1])
        self.unified_action_dim = int(ActionFeature.ALL_FEATURES[1])

        state_path = cfg.get("vitra_state_path")
        self.states_path = cfg.get("vitra_states_path")
        if self.states_path:
            self.state = np.zeros(STATE_DIM, dtype=np.float32)
            state_fov = None
            self.state_valid_hands: set[str] = set()
            self.state_source = f"per-frame states: {self.states_path}"
        else:
            self.state, state_fov, self.state_valid_hands = load_hand_state(
                state_path, self.hands)
            self.state_source = (
                f"statebestand: {state_path}" if state_path
                else "nul-state (VITRA-minimaalvoorbeeld; zet "
                     "vitra_state_path of vitra_reconstruct_hands)")
        self._fov_from_config = cfg.get("vitra_fov_deg") is not None
        self.fov_x_deg, self.fov_y_deg = self._resolve_fov(
            cfg.get("vitra_fov_deg"), state_fov, bool(self.states_path))

        self.state_mask = np.array(
            [hand in self.hands for hand in ("left", "right")], dtype=bool)
        action_mask = np.tile(self.state_mask, (self.chunk_size, 1))
        self._unified_state, self._unified_state_mask = self._pad_state(
            self.state, self.state_valid_hands)
        self._unified_action_mask = pad_action_mask(
            action_mask, self.action_dim, self.unified_action_dim)

    def _pad_state(self, state: np.ndarray, valid_hands: set[str]):
        """Genormaliseerde state + masker; alleen beschikbare handen tellen."""
        norm_state = self.normalizer.normalize_state(
            np.asarray(state, dtype=np.float32).copy())
        mask = np.array(
            [hand in valid_hands for hand in ("left", "right")], dtype=bool)
        return pad_state_human(
            norm_state, mask, self.action_dim, self.state_dim,
            self.unified_state_dim)

    @staticmethod
    def _resolve_fov(value, state_fov: float | None, dynamic_fov: bool
                     ) -> tuple[float, float | None]:
        if value is None:
            if dynamic_fov:
                # fov_x komt per frame uit de reconstructie; fov_y volgt uit
                # de beeldverhouding in _frame_fov.
                fov: tuple[float, float | None] = (60.0, None)
            elif state_fov is not None:
                fov = (float(state_fov), None)
            else:
                fov = (60.0, 60.0)
        elif isinstance(value, (int, float)):
            fov = (float(value), float(value))
        else:
            if not isinstance(value, Sequence) or len(value) != 2:
                raise SystemExit(
                    "vitra_fov_deg moet null, één getal of [fov_x, fov_y] "
                    "zijn.")
            fov = (float(value[0]), float(value[1]))
        if not all(0.0 < part < 180.0 for part in fov if part is not None):
            raise SystemExit("vitra_fov_deg moet tussen 0 en 180 liggen.")
        return fov

    def _frame_fov(self, frame: np.ndarray, fov_x: float | None = None):
        if self._fov_from_config or fov_x is None:
            fov_x_deg = self.fov_x_deg
        else:
            fov_x_deg = float(fov_x)
        fov_x_rad = math.radians(fov_x_deg)
        if self.fov_y_deg is None:
            height, width = frame.shape[:2]
            fov_y = 2 * math.atan((height / width) * math.tan(fov_x_rad / 2))
        else:
            fov_y = math.radians(self.fov_y_deg)
        return self._torch.tensor([[fov_x_rad, fov_y]],
                                  dtype=self._torch.float32)

    def describe(self) -> dict:
        return {
            "instructie": self.instruction,
            "handen": list(self.hands),
            "chunk_size": self.chunk_size,
            "sample_times": self.sample_times,
            "ddim_steps": self.num_ddim_steps,
            "cfg_scale": self.cfg_scale,
            "state_source": self.state_source,
            "states_path": self.states_path,
            "state_hands": ("per-frame" if self.states_path
                            else sorted(self.state_valid_hands)),
            "fov_deg": None if self.states_path else [
                round(self.fov_x_deg, 1),
                None if self.fov_y_deg is None
                else round(self.fov_y_deg, 1)],
        }

    def predict(self, frame: np.ndarray, state: np.ndarray | None = None,
                fov_x: float | None = None,
                state_hands: set[str] | None = None) -> np.ndarray:
        """Voorspel [samples, chunk, 102] genormaliseerde/absolute acties.

        ``state`` overschrijft de statische state (per-frame reconstructie);
        ``state_hands`` zegt welke handen daarin echt geldig zijn. Ontbreekt
        die info, dan gelden alle gevraagde handen als geldig.
        """
        from PIL import Image

        if state is None:
            unified_state = self._unified_state
            unified_state_mask = self._unified_state_mask
        else:
            valid = self.state_valid_hands if state_hands is None else state_hands
            unified_state, unified_state_mask = self._pad_state(state, valid)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self._resize(image, target=self.target_short_side)
        with self._torch.inference_mode():
            norm_action = self.model.predict_action(
                image=np.asarray(image),
                instruction=self.instruction,
                current_state=unified_state.unsqueeze(0),
                current_state_mask=unified_state_mask.unsqueeze(0),
                action_mask_torch=self._unified_action_mask.unsqueeze(0),
                num_ddim_steps=self.num_ddim_steps,
                cfg_scale=self.cfg_scale,
                fov=self._frame_fov(frame, fov_x),
                sample_times=self.sample_times,
            )
        norm_action = np.asarray(norm_action, dtype=np.float32)
        actions = self.normalizer.unnormalize_action(
            norm_action[:, :, :self.action_dim])
        return np.asarray(actions, dtype=np.float32)
