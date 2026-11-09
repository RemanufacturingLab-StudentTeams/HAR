"""ActionCLIP-inferentie met de officiële/MMAction2-checkpoints.

ActionCLIP gebruikt een OpenAI-CLIP ViT-backbone met een getrainde
transformer-adapter over de gehele clip (8, 16 of 32 frames). De checkpoints
van de MMAction2-port zijn stabiel downloadbaar en worden eenmalig in
``models/`` geplaatst. De ``clip``-dependency staat al in requirements.txt
(de Ultralytics/OpenAI-compatible fork).

Deze module bevat bewust geen mmengine/mmaction2: alleen de architectuur die
nodig is voor inferentie, zodat het op een gewone laptop/Windows-pc werkt.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from duplo.actions import _prompt_set_for, prediction_from_logits

ACTIONCLIP_TEMPLATES = (
    "a photo of action {}",
    "a picture of action {}",
    "Human action of {}",
    "{}, an action",
    "{} this is an action",
    "{}, a video of action",
    "Playing action of {}",
    "{}",
    "Playing a kind of action, {}",
    "Doing a kind of action, {}",
    "Look, the human is {}",
    "Can you recognize the action of {}?",
    "Video classification of {}",
    "A video of {}",
    "The man is {}",
    "The woman is {}",
)

# Stable MMAction2-port checkpoints (ported from the official ActionCLIP repo).
ACTIONCLIP_CHECKPOINTS = {
    "ViT-B/32-8": (
        "https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/"
        "actionclip_vit-base-p32-res224-clip-pre_1x1x8_k400-rgb/"
        "vit-b-32-8f.pth"
    ),
    "ViT-B/16-8": (
        "https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/"
        "actionclip_vit-base-p16-res224-clip-pre_1x1x8_k400-rgb/"
        "vit-b-16-8f.pth"
    ),
    "ViT-B/16-16": (
        "https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/"
        "actionclip_vit-base-p16-res224-clip-pre_1x1x16_k400-rgb/"
        "vit-b-16-16f.pth"
    ),
    "ViT-B/16-32": (
        "https://download.openmmlab.com/mmaction/v1.0/projects/actionclip/"
        "actionclip_vit-base-p16-res224-clip-pre_1x1x32_k400-rgb/"
        "vit-b-16-32f.pth"
    ),
}

# MMAction2 ActionDataPreprocessor: RGB, 0-255 (dus geen /255).
ACTIONCLIP_MEAN = (122.771, 116.746, 104.093)
ACTIONCLIP_STD = (68.500, 66.632, 70.323)
ACTIONCLIP_CROP = 224
ACTIONCLIP_RESIZE = 256


class TransformerAdapter(nn.Module):
    """Temporale transformer bovenop de CLIP-framefuncties (ActionCLIP)."""

    def __init__(self, clip_model, num_segs: int, num_layers: int = 6) -> None:
        super().__init__()
        from clip.model import Transformer

        embed_dim = clip_model.text_projection.shape[1]
        transformer_width = clip_model.ln_final.weight.shape[0]
        transformer_heads = transformer_width // 64
        self.frame_position_embeddings = nn.Embedding(num_segs, embed_dim)
        self.transformer = Transformer(
            width=embed_dim, layers=num_layers, heads=transformer_heads)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        original = features
        position_ids = torch.arange(
            features.shape[1], dtype=torch.long, device=features.device)
        embeddings = self.frame_position_embeddings(position_ids)
        features = features + embeddings.unsqueeze(0)
        features = features.transpose(0, 1)  # NLD -> LND
        features = self.transformer(features)
        features = features.transpose(0, 1)  # LND -> NLD
        features = features.type(original.dtype) + original
        return features.mean(dim=1)


class ActionClipNetwork(nn.Module):
    """CLIP-backbone + ActionCLIP-adapter, met dezelfde keynamen als het checkpoint."""

    def __init__(self, clip_arch: str, num_segs: int, num_layers: int,
                 download_root: str | None = None) -> None:
        super().__init__()
        try:
            import clip
        except ImportError as exc:
            raise SystemExit(
                "Het 'clip'-pakket ontbreekt. Installeer de requirements "
                "opnieuw met run_actionclip_test.bat --upgrade "
                "(of pip install -r requirements.txt)."
            ) from exc

        self.clip, _ = clip.load(clip_arch, device="cpu",
                                 download_root=download_root)
        self.adapter = TransformerAdapter(self.clip, num_segs, num_layers)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        flat = video.view(-1, channels, height, width)
        features = self.clip.encode_image(flat)
        features = features.view(batch, frames, -1)
        return self.adapter(features)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(text)


def preprocess_actionclip_frames(frames) -> torch.Tensor:
    """BGR-frames -> genormaliseerde ``[1, T, 3, 224, 224]``-tensor.

    Volgt de MMAction2-testpipeline: resize korte zijde naar 256,
    center-crop 224, RGB, daarna normaliseren met de ActionCLIP-mean/std op
    een 0-255-schaal.
    """
    processed = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = ACTIONCLIP_RESIZE / float(min(height, width))
        new_width = max(ACTIONCLIP_CROP, round(width * scale))
        new_height = max(ACTIONCLIP_CROP, round(height * scale))
        resized = cv2.resize(rgb, (new_width, new_height),
                             interpolation=cv2.INTER_LINEAR)
        left = max(0, (new_width - ACTIONCLIP_CROP) // 2)
        top = max(0, (new_height - ACTIONCLIP_CROP) // 2)
        crop = resized[top:top + ACTIONCLIP_CROP,
                       left:left + ACTIONCLIP_CROP]
        processed.append(crop)
    array = np.stack(processed).astype(np.float32)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    mean = torch.tensor(ACTIONCLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(ACTIONCLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return target
    temp = target.with_name(f"{target.name}.part")
    print(f"ActionCLIP-checkpoint downloaden naar {target}...")
    try:
        urlretrieve(url, temp)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def _resolve_checkpoint(cfg: dict, name: str) -> Path:
    configured = str(cfg.get("actionclip_checkpoint") or "").strip()
    if configured:
        if configured.startswith(("http://", "https://")):
            target = Path("models") / Path(configured).name
            return _download(configured, target)
        path = Path(configured)
        if path.is_file():
            return path
        raise SystemExit(
            f"actionclip_checkpoint bestaat niet: {path}. Laat de sleutel "
            "leeg om het checkpoint automatisch te downloaden.")
    if name not in ACTIONCLIP_CHECKPOINTS:
        raise SystemExit(
            f"Onbekend ActionCLIP-model: {name!r}. Kies uit: "
            f"{', '.join(ACTIONCLIP_CHECKPOINTS)} of zet een lokaal "
            "'actionclip_checkpoint' in de config.")
    url = ACTIONCLIP_CHECKPOINTS[name]
    return _download(url, Path("models") / Path(url).name)


def _load_state_dict(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Oudere torch-versies kennen weights_only niet.
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:  # noqa: BLE001 - checkpointformaat is niet voorspelbaar
        # Checkpoints met metadata die niet weights_only-compatibel is; het
        # bestand komt van een vertrouwde model-URL of is zelf opgegeven.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise SystemExit(f"Onbekend checkpointformaat: {path}")
    return {str(key).removeprefix("module."): value
            for key, value in state.items()}


def _infer_clip_arch(state: dict) -> str:
    weight = state.get("clip.visual.conv1.weight")
    if weight is None:
        raise SystemExit(
            "Het ActionCLIP-checkpoint bevat geen 'clip.visual'-gewichten; "
            "is dit wel een ActionCLIP/MMAction2-checkpoint?")
    patch = int(weight.shape[-1])
    if patch not in (16, 32):
        raise SystemExit(
            f"Onbekende CLIP-patchgrootte {patch}; alleen ViT-B/16 en "
            "ViT-B/32 worden ondersteund.")
    return f"ViT-B/{patch}"


def _infer_adapter(state: dict) -> tuple[int, int]:
    embedding = state.get("adapter.frame_position_embeddings.weight")
    if embedding is None:
        raise SystemExit(
            "Het ActionCLIP-checkpoint bevat geen adaptergewichten.")
    num_segs = int(embedding.shape[0])
    layers = {int(key.split(".")[3])
              for key in state
              if key.startswith("adapter.transformer.resblocks.")
              and key.split(".")[3].isdigit()}
    return num_segs, (max(layers) + 1 if layers else 6)


class ActionClipClassifier:
    """ActionCLIP-classifier met dezelfde interface als de andere actiemodellen."""

    temporal = False

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.torch_device = "cuda" if device == "gpu" else "cpu"
        self.name = str(cfg.get("actionclip_model", "ViT-B/32-8"))
        checkpoint = _resolve_checkpoint(cfg, self.name)
        state = _load_state_dict(checkpoint)
        clip_arch = _infer_clip_arch(state)
        num_segs, num_layers = _infer_adapter(state)
        self.model_id = f"ActionCLIP {clip_arch}-{num_segs}f"
        self.prompt_set = _prompt_set_for(cfg, "actionclip",
                                          ACTIONCLIP_TEMPLATES)
        self.top_k = int(cfg.get("action_top_k", 3))
        self.num_frames = num_segs
        self.model = ActionClipNetwork(
            clip_arch, num_segs, num_layers,
            download_root=str(cfg.get("actionclip_clip_cache", "models/clip")))
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise SystemExit(
                "Het ActionCLIP-checkpoint mist gewichten voor deze "
                f"architectuur: {', '.join(incompatible.missing_keys[:5])}"
                f"{' ...' if len(incompatible.missing_keys) > 5 else ''}")
        if incompatible.unexpected_keys:
            print(f"Waarschuwing: {len(incompatible.unexpected_keys)} "
                  "onbekende checkpoint-keys genegeerd.")
        self.model.to(self.torch_device).eval()
        self._text_features = self._encode_text()

    @property
    def classes(self) -> tuple[str, ...]:
        return self.prompt_set.classes

    def _encode_text(self) -> torch.Tensor:
        import clip

        tokens = clip.tokenize(list(self.prompt_set.prompts))
        with torch.inference_mode():
            features = self.model.encode_text(tokens.to(self.torch_device))
            return F.normalize(features.float(), dim=-1)

    def predict(self, frames, top_k: int | None = None,
                clip_start_s: float | None = None,
                clip_end_s: float | None = None):
        video = preprocess_actionclip_frames(frames).to(self.torch_device)
        with torch.inference_mode():
            features = self.model.encode_video(video)
            features = F.normalize(features.float(), dim=-1)
            logits = 100.0 * features @ self._text_features.T
        return prediction_from_logits(
            self.prompt_set, logits[0].cpu().numpy(),
            self.top_k if top_k is None else top_k)
