"""Gedeelde actieclassificatie op videoclips.

InternVideo2, X-CLIP en ActionCLIP zijn geen objectdetectoren maar
classificeren een *clip* (meerdere frames) als één actie. Deze module bevat
alles wat ze gemeen hebben: klassen/prompts uit de config, deterministische
clip-sampling, videolezen, score-aggregatie en annotatie. Alleen de
modeladapter (X-CLIP/InternVideo2 hier, ActionCLIP in
:mod:`duplo.actionclip`) verschilt.

De uitvoer is één rij per clip in ``<model>_actions.csv`` met dezelfde
``frame``/``klasse``/``confidence``-conventie als de detectiestappen,
aangevuld met het clipbereik en de top-k scores.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_MODEL_KINDS = ("xclip", "internvideo2", "actionclip",
                      "smolvlm2", "llavaction", "qwen3vl")
VLM_ACTION_KINDS = ("smolvlm2", "llavaction", "qwen3vl")

DEFAULT_ACTION_TEMPLATES = (
    "a video of a person {}",
    "the person is {}",
    "{}",
)

_ACTION_MODEL_ALIASES = {
    "xclip": "xclip",
    "x-clip": "xclip",
    "x_clip": "xclip",
    "internvideo2": "internvideo2",
    "intern-video2": "internvideo2",
    "intern_video2": "internvideo2",
    "internvideo-2": "internvideo2",
    "intern-video-2": "internvideo2",
    "actionclip": "actionclip",
    "action-clip": "actionclip",
    "action_clip": "actionclip",
    "smolvlm2": "smolvlm2",
    "smol-vlm2": "smolvlm2",
    "smol_vlm2": "smolvlm2",
    "llavaction": "llavaction",
    "llava-action": "llavaction",
    "llava_action": "llavaction",
    "qwen3vl": "qwen3vl",
    "qwen3-vl": "qwen3vl",
    "qwen3_vl": "qwen3vl",
    "qwen3-vl-8b": "qwen3vl",
    "qwen3-vl-8b-instruct": "qwen3vl",
    "qwen/qwen3-vl-8b-instruct": "qwen3vl",
    "huggingfacetb/smolvlm2-2.2b-instruct": "smolvlm2",
    "mladaptiveintelligence/llavaction-7b": "llavaction",
}


@dataclass(frozen=True)
class ActionPrediction:
    """Voorspelling voor één clip, inclusief optionele temporele grenzen.

    ``confidence`` is ``None`` voor generatieve VLM's die geen
    gekalibreerde score geven. ``start_s``/``end_s`` zijn alleen gevuld als
    het model begin- en eindtijd voorspelt (Qwen3-VL, LLaVAction, SmolVLM2).
    ``raw_text`` bewaart de ruwe modeluitvoer voor foutanalyse.
    """

    label: str
    confidence: float | None
    scores: dict[str, float]
    start_s: float | None = None
    end_s: float | None = None
    raw_text: str | None = None

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """De ``k`` hoogste scores, aflopend en deterministisch bij gelijkheid."""
        ranked = sorted(self.scores.items(), key=lambda item: (-item[1],
                                                               item[0]))
        return ranked[:max(1, int(k))]


@dataclass(frozen=True)
class ClipWindow:
    """Eén clip: start/end (end exclusief) en de gesamplede frame-indices."""

    start: int
    end: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class PromptSet:
    """Prompts gegroepeerd per actieklasse.

    Meerdere templates per klasse worden gebruikt om de voorspelling
    robuuster te maken. De logits worden per klasse gemiddeld *voordat* de
    softmax over de klassen gaat, zodat een klasse met meer templates niet
    automatisch meer kans krijgt.
    """

    classes: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(prompt for group in self.groups for prompt in group)

    def aggregate_logits(self, logits: Sequence[float]) -> np.ndarray:
        """Gemiddelde logit per klasse uit de logits van alle prompts."""
        values = np.asarray(logits, dtype=np.float64).reshape(-1)
        if values.size != len(self.prompts):
            raise ValueError(
                f"Verwacht {len(self.prompts)} prompt-scores, kreeg "
                f"{values.size}.")
        means = []
        offset = 0
        for group in self.groups:
            means.append(float(values[offset:offset + len(group)].mean()))
            offset += len(group)
        return np.asarray(means, dtype=np.float64)


def normalize_action_model(name: str | None) -> str:
    """Maak modelnamen als ``X-CLIP``/``ActionCLIP`` uniform."""
    key = str(name or "").strip().lower()
    if key in _ACTION_MODEL_ALIASES:
        return _ACTION_MODEL_ALIASES[key]
    raise SystemExit(
        f"Onbekend actiemodel: {name!r}. Kies uit: "
        f"{', '.join(ACTION_MODEL_KINDS)}.")


def parse_action_classes(value) -> list[str]:
    """Lees ``action_classes`` uit de config (lijst of kommagescheiden tekst)."""
    if value is None:
        raise SystemExit(
            "Geen action_classes in de config. Zet minimaal één actieklasse "
            "in configs/baseline.yaml.")
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise SystemExit(
            "action_classes moet een lijst of kommagescheiden tekst zijn.")
    classes: list[str] = []
    seen: set[str] = set()
    for item in items:
        label = str(item).strip()
        if not label:
            continue
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            classes.append(label)
    if not classes:
        raise SystemExit("action_classes is leeg.")
    return classes


def _as_templates(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise SystemExit(
        "action_templates moet een lijst met teksttemplates zijn.")


def build_prompt_set(classes: Sequence[str],
                     templates: Sequence[str] | str | None = None,
                     prompts: dict | None = None,
                     default_templates: Sequence[str] | str | None = None
                     ) -> PromptSet:
    """Bouw prompts per klasse uit templates en/of per-klasse prompts.

    ``prompts`` heeft voorrang op ``templates``. Een waarde in ``prompts``
    mag één tekst of een lijst teksten zijn. Ontbreekt een klasse in
    ``prompts``, dan valt die terug op de templates.
    """
    class_list = [str(label).strip() for label in classes if str(label).strip()]
    if not class_list:
        raise SystemExit("action_classes is leeg.")
    template_list = _as_templates(
        templates if templates is not None else default_templates)
    if template_list:
        invalid = [template for template in template_list if "{}" not in template]
        if invalid:
            raise SystemExit(
                "Elke action-template moet '{}' bevatten, bijvoorbeeld "
                "'a video of a person {}'.")
    prompt_lookup: dict[str, list[str]] = {}
    if prompts is not None:
        if not isinstance(prompts, dict):
            raise SystemExit(
                "action_prompts moet een mapping van klasse naar prompt(s) zijn.")
        for key, value in prompts.items():
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, (list, tuple)):
                values = [str(item) for item in value]
            else:
                raise SystemExit(
                    f"Prompt voor {key!r} moet tekst of een lijst teksten zijn.")
            cleaned = [item.strip() for item in values if item.strip()]
            if cleaned:
                prompt_lookup[str(key).strip().casefold()] = cleaned

    groups: list[tuple[str, ...]] = []
    for label in class_list:
        custom = prompt_lookup.get(label.casefold())
        if custom:
            groups.append(tuple(custom))
        elif template_list:
            groups.append(tuple(template.format(label)
                                for template in template_list))
        else:
            groups.append((label,))
    return PromptSet(tuple(class_list), tuple(groups))


def softmax(values: Sequence[float]) -> np.ndarray:
    """Numeriek stabiele softmax voor een 1D-array."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    shifted = array - np.max(array)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(array.size, 1.0 / array.size)
    return exp / total


def prediction_from_logits(prompt_set: PromptSet,
                           logits: Sequence[float],
                           top_k: int = 3) -> ActionPrediction:
    """Zet prompt-logits om naar een :class:`ActionPrediction`."""
    class_logits = prompt_set.aggregate_logits(logits)
    probabilities = softmax(class_logits)
    scores = {label: float(probability)
              for label, probability in zip(prompt_set.classes, probabilities)}
    best = int(np.argmax(probabilities))
    return ActionPrediction(
        label=prompt_set.classes[best],
        confidence=float(probabilities[best]),
        scores=scores,
    )


def build_vlm_prompt(classes: Sequence[str], clip_start_s: float,
                     clip_end_s: float) -> str:
    """Prompt voor een VLM: label plus start/end relative aan de clip."""
    duration = max(0.0, clip_end_s - clip_start_s)
    options = "\n".join(f"- {label}" for label in classes)
    return (
        "You are analyzing a short video clip from a larger video.\n"
        f"The clip starts at {clip_start_s:.2f} seconds and lasts "
        f"{duration:.2f} seconds.\n"
        "Choose exactly one action from this list:\n"
        f"{options}\n"
        "Also estimate when the chosen action starts and ends. Give "
        "start_s and end_s as seconds relative to the START OF THIS CLIP "
        f"(0 to {duration:.2f}).\n"
        "Return ONLY valid JSON, without markdown, with exactly these keys: "
        '{"action": "...", "start_s": 0.0, "end_s": 0.0, '
        '"confidence": 0.0}. Use null for the times if you cannot determine '
        "them."
    )


def _json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _value(record: dict, *names):
    lowered = {str(key).strip().lower(): value
               for key, value in record.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _float_value(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", ".")
    if not text or text in ("none", "null", "nan", "n/a", "n.v.t."):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _match_class(value, classes: Sequence[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    ordered = sorted(classes, key=len, reverse=True)
    for label in ordered:
        if text == label.casefold():
            return label
    for label in ordered:
        if label.casefold() in text:
            return label
    return None


def parse_vlm_action_response(text: str, classes: Sequence[str],
                              clip_start_s: float,
                              clip_end_s: float) -> ActionPrediction:
    """Parse een VLM-antwoord naar label, confidence en absolute tijden.

    De prompt vraagt om clip-relatieve tijden; die worden hier omgezet naar
    absolute seconden. Valt het model terug op vrije tekst, dan proberen we
    alsnog een klassenaam en een tijdsbereik te vinden.
    """
    duration = max(0.0, clip_end_s - clip_start_s)
    record = _json_object(text) or {}
    action_value = _value(record, "action", "label", "klasse", "class")
    label = _match_class(action_value, classes)
    if label is None:
        label = _match_class(text, classes)
    label = label or "unknown"
    confidence_value = _value(record, "confidence", "score", "probability")
    if isinstance(confidence_value, str) and confidence_value.strip().endswith("%"):
        try:
            confidence = float(confidence_value.strip().rstrip("%")) / 100.0
        except ValueError:
            confidence = None
    else:
        confidence = _float_value(confidence_value)
    if confidence is not None:
        if confidence > 1.0 and confidence <= 100.0:
            confidence /= 100.0
        confidence = max(0.0, min(1.0, confidence))

    start = _float_value(_value(record, "start_s", "start", "start_time",
                                "start_timestamp"))
    end = _float_value(_value(record, "end_s", "end", "end_time",
                              "end_timestamp"))
    if start is None or end is None:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if start is None and end is None and len(numbers) >= 2:
            start, end = float(numbers[0]), float(numbers[1])
    if start is not None:
        start = max(0.0, min(duration, start)) + clip_start_s
    if end is not None:
        end = max(0.0, min(duration, end)) + clip_start_s
    if start is not None and end is not None and end < start:
        start, end = end, start
    if start is not None and end is not None and end <= start:
        start = end = None

    scores = ({label: confidence}
              if label != "unknown" and confidence is not None else {})
    return ActionPrediction(
        label=label,
        confidence=confidence,
        scores=scores,
        start_s=start,
        end_s=end,
        raw_text=text.strip(),
    )


def clip_windows(total_frames: int, clip_frames: int, clip_span: int,
                 clip_stride: int) -> list[ClipWindow]:
    """Deterministische clipindeling voor een video.

    ``clip_frames`` frames worden gelijkmatig verdeeld over ``clip_span``
    videoframes; de volgende clip start ``clip_stride`` frames later. De
    laatste clip wordt naar het video-einde geschoven zodat de staart altijd
    meedoet. Bij een korte video worden frame-indices geclipt (herhaling van
    het laatste frame, zoals bij de meeste actiemodellen).
    """
    total = int(total_frames)
    frames = int(clip_frames)
    span = int(clip_span)
    stride = int(clip_stride)
    if total <= 0 or frames <= 0 or span <= 0 or stride <= 0:
        return []
    span = min(span, total)
    starts = list(range(0, max(1, total - span + 1), stride))
    if starts[-1] + span < total:
        starts.append(max(0, total - span))
    windows: list[ClipWindow] = []
    seen_starts: set[int] = set()
    for start in starts:
        if start in seen_starts:
            continue
        seen_starts.add(start)
        end = min(start + span, total)
        indices = np.linspace(start, end - 1, num=frames)
        indices = np.clip(np.rint(indices).astype(int), start, end - 1)
        windows.append(ClipWindow(start=start, end=end,
                                  indices=tuple(int(i) for i in indices)))
    return windows


class VideoClipReader:
    """Leest frames op willekeurige indices met één open VideoCapture.

    De clips worden op volgorde verwerkt, dus frames worden zoveel mogelijk
    sequentieel gedecodeerd (``grab`` voor overgeslagen frames). Alleen bij
    een sprong terug wordt opnieuw ge-seekt.
    """

    def __init__(self, video_path: str | Path) -> None:
        import cv2

        self._cv2 = cv2
        self.video_path = str(video_path)
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            self.capture.release()
            raise SystemExit(
                f"Kon de video niet openen: {video_path} (controleer het pad "
                "en of het bestand een leesbaar videoformaat heeft)")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.total_frames <= 0:
            # Sommige codecs/containers melden geen framecount; tel ze eenmalig
            # en open de video daarna opnieuw (seeken naar 0 is niet overal
            # betrouwbaar na het einde van de stream).
            counted = 0
            while self.capture.grab():
                counted += 1
            self.total_frames = counted
            self.capture.release()
            self.capture = cv2.VideoCapture(self.video_path)
            if not self.capture.isOpened():
                self.capture.release()
                raise SystemExit(
                    f"Kon de video niet opnieuw openen: {video_path}")
        self._next_frame = 0
        self._last_index = -1
        self._last_frame: np.ndarray | None = None

    def _read_at(self, index: int) -> np.ndarray:
        index = int(index)
        if index < 0 or (self.total_frames and index >= self.total_frames):
            raise ValueError(
                f"Frame {index} valt buiten de video (0.."
                f"{max(0, self.total_frames - 1)}).")
        if index == self._last_index and self._last_frame is not None:
            return self._last_frame.copy()
        if index < self._next_frame:
            self.capture.set(self._cv2.CAP_PROP_POS_FRAMES, index)
            self._next_frame = index
        while self._next_frame < index:
            if not self.capture.grab():
                raise SystemExit(
                    f"Kon frame {self._next_frame} van {self.video_path} "
                    "niet lezen.")
            self._next_frame += 1
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise SystemExit(
                f"Kon frame {index} van {self.video_path} niet lezen.")
        self._next_frame = index + 1
        self._last_index = index
        self._last_frame = frame
        return frame

    def read(self, indices: Iterable[int]) -> list[np.ndarray]:
        return [self._read_at(index) for index in indices]

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> VideoClipReader:  # noqa: PYI034
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def video_info(video_path: str | Path) -> tuple[float, int]:
    """``(fps, aantal frames)`` zonder een model te laden."""
    reader = VideoClipReader(video_path)
    try:
        return reader.fps, reader.total_frames
    finally:
        reader.close()


def annotate_action(frame: np.ndarray, prediction: ActionPrediction,
                    frame_idx: int, fps: float, clip_start: int,
                    clip_end: int, top_k: int = 3) -> np.ndarray:
    """Kopie van het frame met actielabel, clipbereik en top-k scores."""
    import cv2

    annotated = frame.copy()
    confidence = ("n.v.t." if prediction.confidence is None
                  else f"{prediction.confidence:.2f}")
    lines = [
        f"Action: {prediction.label} ({confidence})",
        (f"Clip: {clip_start}-{clip_end} frames @ "
         f"{(frame_idx / fps) if fps else 0.0:.1f}s"),
    ]
    if prediction.start_s is not None and prediction.end_s is not None:
        lines.append(f"Times: {prediction.start_s:.2f}-"
                     f"{prediction.end_s:.2f}s")
    lines.extend(f"{label}: {score:.2f}"
                 for label, score in prediction.top_k(top_k))
    line_height = 24
    width = min(annotated.shape[1] - 20,
                max(240, int(max(len(line) for line in lines) * 9.5)))
    height = len(lines) * line_height + 14
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
    for row, line in enumerate(lines):
        cv2.putText(annotated, line, (16, 32 + row * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return annotated


def _prompt_set_for(cfg: dict, model_key: str,
                    default_templates: Sequence[str] | str | None = None
                    ) -> PromptSet:
    classes = parse_action_classes(cfg.get("action_classes"))
    templates = cfg.get(f"{model_key}_templates", cfg.get("action_templates"))
    prompts = cfg.get(f"{model_key}_prompts", cfg.get("action_prompts"))
    return build_prompt_set(classes, templates, prompts, default_templates)


class XClipClassifier:
    """X-CLIP (Microsoft VideoX) via Hugging Face transformers."""

    temporal = False

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        from transformers import AutoModel, AutoProcessor

        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.torch_device = "cuda" if device == "gpu" else "cpu"
        self.model_id = str(cfg.get(
            "xclip_model", "microsoft/xclip-base-patch16-zero-shot"))
        self.prompt_set = _prompt_set_for(cfg, "xclip",
                                          DEFAULT_ACTION_TEMPLATES)
        self.top_k = int(cfg.get("action_top_k", 3))
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.to(self.torch_device).eval()
        # Het checkpoint bepaalt het aantal frames (8/16/32), niet de config:
        # een afwijkend aantal geeft anders een shape-fout in de MIT.
        vision_config = getattr(self.model.config, "vision_config", None)
        self.num_frames = int(getattr(
            vision_config, "num_frames", cfg.get("action_frames", 8)))

    @property
    def classes(self) -> tuple[str, ...]:
        return self.prompt_set.classes

    def predict(self, frames: Sequence[np.ndarray],
                top_k: int | None = None,
                clip_start_s: float | None = None,
                clip_end_s: float | None = None) -> ActionPrediction:
        import cv2
        import torch

        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                      for frame in frames]
        inputs = self.processor(
            text=list(self.prompt_set.prompts), videos=rgb_frames,
            return_tensors="pt", padding=True)
        inputs = {key: (value.to(self.torch_device)
                        if hasattr(value, "to") else value)
                  for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits_per_video[0]
        return prediction_from_logits(
            self.prompt_set, logits.float().cpu().numpy(),
            self.top_k if top_k is None else top_k)


def _patch_transformers_for_remote_code() -> None:
    """Vang kleine API-verschillen op voor oude remote-code-modellen.

    InternVideo2's Hugging Face-code is geschreven voor transformers 4.x en
    gebruikt ``all_tied_weights_keys``; transformers 5 heeft die naam
    vervangen door ``get_expanded_tied_weights_keys()``. Zonder deze
    compatibiliteitsproperty stopt ``from_pretrained`` na het laden van de
    gewichten met een AttributeError.
    """
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        getter = getattr(PreTrainedModel, "get_expanded_tied_weights_keys",
                         None)
        PreTrainedModel.all_tied_weights_keys = property(
            (lambda self: getter(self)) if getter is not None
            else (lambda self: {}))


def _install_flash_attn_shim() -> None:
    """Laat InternVideo2's remote code importeren zonder flash-attn.

    Het Hugging Face-model importeert ``flash_attn`` onvoorwaardelijk, terwijl
    de configuratie ``use_flash_attn=False`` gebruikt. Op Windows/laptops is
    flash-attn vaak niet te installeren; deze shim voorziet alleen de namen
    die bij import nodig zijn. Wordt flash-attn alsnog gebruikt, dan geeft de
    placeholder een duidelijke fout in plaats van een ImportError bij import.
    """
    try:
        import flash_attn  # noqa: F401
        return
    except (ImportError, OSError, RuntimeError):
        # Ook een half geinstalleerde/beschadigde flash-attn valt hieronder;
        # de configuratie gebruikt hem niet, dus de shim is veilig.
        pass

    import importlib.util
    import sys
    import types

    from torch import nn

    class _MissingFlashAttention(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "Dit model gebruikt flash-attn, maar het pakket is niet "
                "geinstalleerd. Installeer flash-attn of kies een model/"
                "configuratie zonder flash-attn.")

    def _missing_flash_function(*args, **kwargs):
        raise RuntimeError(
            "Dit model gebruikt flash-attn, maar het pakket is niet "
            "geinstalleerd. Installeer flash-attn of kies een model/"
            "configuratie zonder flash-attn.")

    module_names = (
        "flash_attn",
        "flash_attn.modules",
        "flash_attn.modules.mlp",
        "flash_attn.ops",
        "flash_attn.ops.rms_norm",
        "flash_attn.flash_attn_interface",
        "flash_attn.bert_padding",
    )
    for name in module_names:
        module = types.ModuleType(name)
        # Een echte ModuleSpec voorkomt dat transformers'
        # importlib.util.find_spec() op een module met __spec__=None stuit.
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        module.__loader__ = None
        module.__package__ = name.rpartition(".")[0]
        sys.modules[name] = module
    sys.modules["flash_attn.modules.mlp"].FusedMLP = _MissingFlashAttention
    sys.modules["flash_attn.ops.rms_norm"].DropoutAddRMSNorm = (
        _MissingFlashAttention)
    sys.modules["flash_attn.flash_attn_interface"].flash_attn_varlen_qkvpacked_func = (
        _missing_flash_function)
    sys.modules["flash_attn.bert_padding"].unpad_input = _missing_flash_function
    sys.modules["flash_attn.bert_padding"].pad_input = _missing_flash_function
    # Maak de submodules ook als attribuut bereikbaar voor ``import
    # flash_attn.modules.mlp`` in oudere importpaden.
    sys.modules["flash_attn"].modules = sys.modules["flash_attn.modules"]
    sys.modules["flash_attn.modules"].mlp = sys.modules[
        "flash_attn.modules.mlp"]
    sys.modules["flash_attn"].ops = sys.modules["flash_attn.ops"]


class InternVideo2Classifier:
    """InternVideo2-CLIP via het Hugging Face-model met remote code."""

    temporal = False

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        # Eerst transformers importeren: transformers controleert bij import
        # of flash-attn beschikbaar is. De shim daarna installeren voorkomt
        # dat die controle onze placeholder-module tegenkomt.
        from transformers import AutoConfig, AutoModel

        _patch_transformers_for_remote_code()
        _install_flash_attn_shim()
        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.torch_device = "cuda" if device == "gpu" else "cpu"
        self.model_id = str(cfg.get(
            "internvideo2_model", "OpenGVLab/InternVideo2_CLIP_S"))
        self.prompt_set = _prompt_set_for(cfg, "internvideo2",
                                          DEFAULT_ACTION_TEMPLATES)
        self.top_k = int(cfg.get("action_top_k", 3))
        trust_remote_code = bool(
            cfg.get("internvideo2_trust_remote_code", True))
        revision = cfg.get("internvideo2_revision") or None
        config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=trust_remote_code,
            revision=revision)
        self.model = AutoModel.from_pretrained(
            self.model_id, config=config,
            trust_remote_code=trust_remote_code, revision=revision)
        self.model.to(self.torch_device).eval()
        # Het HF-config.json pint fp16, maar de remote tekstencoder cast zijn
        # invoer naar float32 terwijl de LayerNorm-gewichten half blijven; dat
        # geeft op CUDA "expected scalar type Float but found Half". Float32
        # is voor dit model klein genoeg en werkt op CPU en GPU identiek.
        self.model.float()
        # Het checkpoint bepaalt het aantal frames; action_frames is alleen
        # een fallback voor modellen zonder num_frames in de config.
        self.num_frames = int(getattr(
            config, "num_frames", cfg.get("action_frames", 8)))
        self.dtype = next(self.model.parameters()).dtype

    @property
    def classes(self) -> tuple[str, ...]:
        return self.prompt_set.classes

    def predict(self, frames: Sequence[np.ndarray],
                top_k: int | None = None,
                clip_start_s: float | None = None,
                clip_end_s: float | None = None) -> ActionPrediction:
        import cv2
        import torch
        import torch.nn.functional as F

        rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        for frame in frames], axis=0)
        video = torch.from_numpy(rgb).permute(0, 3, 1, 2)
        video = self.model.transform(video).unsqueeze(0)
        video = video.to(device=self.torch_device, dtype=self.dtype)
        text = self.model.tokenizer(list(self.prompt_set.prompts))
        if not torch.is_tensor(text):
            text = torch.as_tensor(text)
        text = text.to(self.torch_device)
        with torch.inference_mode():
            video_features = self.model.encode_vision(video, test=True)
            text_features = self.model.encode_text(text)
            video_features = F.normalize(video_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)
            logits = (text_features @ video_features.T).squeeze(-1) * 100.0
        return prediction_from_logits(
            self.prompt_set, logits.float().cpu().numpy(),
            self.top_k if top_k is None else top_k)


def create_action_classifier(cfg: dict, device: str,
                             kind: str | None = None):
    """Factory voor de drie actiemodellen (lazy imports per model)."""
    kind = normalize_action_model(kind or cfg.get("action_model", "xclip"))
    if kind == "xclip":
        return XClipClassifier(cfg, device)
    if kind == "internvideo2":
        return InternVideo2Classifier(cfg, device)
    if kind == "actionclip":
        from duplo.actionclip import ActionClipClassifier

        return ActionClipClassifier(cfg, device)
    if kind in VLM_ACTION_KINDS:
        from duplo.vlm_actions import create_vlm_classifier

        return create_vlm_classifier(cfg, device, kind)
    raise SystemExit(f"Onbekend actiemodel: {kind!r}.")
