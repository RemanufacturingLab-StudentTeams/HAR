"""Generatieve video-VLM's als actieclassifiers.

SmolVLM2, InternVideo2.5 en InternVideo3 zijn algemene video-language-
modellen, geen gespecialiseerde temporele actieclassifiers. Ze krijgen
dezelfde clipframes plus een strak JSON-prompt voor zowel het actielabel als
de begin-/eindtijd. De temporele voorspellingen zijn clip-relatief (0 = start
van de clip) en worden hier omgerekend naar absolute seconden. Of dat
betrouwbaar is, meet :mod:`duplo.action_eval` apart tegen ground truth.

De drie modellen hebben ieder hun eigen inferencepad:

* SmolVLM2 gebruikt de standaard transformers-processor met een videoblok;
* InternVideo2.5 (InternVL-stijl) verwacht losse, dynamisch getegelde
  afbeeldingen en draait vanwege zijn oude remote code in de gepinde
  transformers 4.47-runtime;
* InternVideo3 gebruikt een Qwen-VL-achtige processor met videotokens.

De adapters hieronder verbergen die verschillen achter dezelfde
``predict``-interface als de gespecialiseerde classifiers.
"""
from __future__ import annotations

from duplo.actions import (
    DEFAULT_ACTION_TEMPLATES,
    _patch_transformers_for_remote_code,
    _prompt_set_for,
    build_vlm_prompt,
    parse_vlm_action_response,
)

_SMOLVLM2_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
_INTERNVIDEO25_MODEL = "OpenGVLab/InternVideo2_5_Chat_8B"
_INTERNVIDEO3_MODEL = "yanziang/InternVideo3-8B-Instruct"

_INTERNVIDEO25_MEAN = (0.485, 0.456, 0.406)
_INTERNVIDEO25_STD = (0.229, 0.224, 0.225)


def _load_remote_config(model_id: str):
    """Laad een remote config zonder de kapotte repr/logging van oudere code.

    InternVideo2.5's ``InternVLChatConfig.to_dict()`` crasht op een default-
    instantie (``llm_config`` ontbreekt). Transformers 5 logt de config tijdens
    ``from_pretrained`` en evalueert daarbij meteen ``repr(config)``; door de
    repr tijdelijk te vervangen blijft het laden werken.
    """
    from transformers import AutoConfig
    from transformers.configuration_utils import PretrainedConfig

    original_repr = PretrainedConfig.__repr__
    PretrainedConfig.__repr__ = lambda self: self.__class__.__name__
    try:
        return AutoConfig.from_pretrained(
            model_id, trust_remote_code=True)
    finally:
        PretrainedConfig.__repr__ = original_repr


def _patch_internvideo3_transformers() -> None:
    """Vang twee transformers 5-wijzigingen op voor InternVideo3's remote code.

    * ``check_model_inputs`` was vroeger een decoratorfactory (``@...()``) en
      is nu een gewone decorator;
    * de ``default``-RoPE-variant is uit ``ROPE_INIT_FUNCTIONS`` verdwenen,
      terwijl de config van InternVideo3 die naam gebruikt.
    """
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.utils import generic

    original = generic.check_model_inputs
    if not getattr(original, "_duplo_compat", False):
        def check_model_inputs(func=None, **kwargs):
            if func is None:
                return lambda inner: original(inner)
            return original(func)

        check_model_inputs._duplo_compat = True
        generic.check_model_inputs = check_model_inputs

    if "default" not in ROPE_INIT_FUNCTIONS:
        def default_rope(config=None, device=None, seq_len=None,
                         layer_type=None):
            base = float(getattr(config, "rope_theta", 10000.0))
            head_dim = (getattr(config, "head_dim", None)
                        or config.hidden_size // config.num_attention_heads)
            partial = float(getattr(config, "partial_rotary_factor", 1.0))
            dim = max(2, int(head_dim * partial))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                         .to(device=device, dtype=torch.float) / dim))
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = default_rope


def _patch_missing_pad_token(config) -> None:
    """Oudere configs verwachten een pad_token_id-attribuut dat 5.x weggelaten."""
    for sub_config in (config, getattr(config, "text_config", None),
                       getattr(config, "vision_config", None)):
        if sub_config is not None and not hasattr(sub_config, "pad_token_id"):
            sub_config.pad_token_id = None


def _load_remote_tokenizer(model_id: str):
    """Laad een remote tokenizer achter dezelfde repr-compatibiliteitspatch."""
    from transformers import AutoTokenizer
    from transformers.configuration_utils import PretrainedConfig

    original_repr = PretrainedConfig.__repr__
    PretrainedConfig.__repr__ = lambda self: self.__class__.__name__
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True)
    finally:
        PretrainedConfig.__repr__ = original_repr
    if not tokenizer:
        # transformers 4.x geeft False terug als de SentencePiece-load faalt
        # (het tokenizer.model van InternVideo2.5 bevat een NUL-piece).
        raise RuntimeError("tokenizer kon niet geladen worden")
    return tokenizer


def _sanitize_internvideo25_tokenizer(model_id: str, cfg: dict):
    """Maak een laadbare kopie van InternVideo2.5's SentencePiece-model.

    Het tokenizer.model van InternVideo2.5 bevat één piece met een letterlijke
    NUL-byte. SentencePiece 0.2 weigert dat ("piece must not include null
    character"), terwijl het model zelf gewoon bruikbaar is. De piece wordt
    daarom in een lokale kopie vervangen door een ongebruikte placeholder.
    """
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    try:
        from sentencepiece import sentencepiece_model_pb2
    except ImportError as exc:
        raise SystemExit(
            "Het 'protobuf'-pakket ontbreekt; nodig om het tokenizer.model "
            "van InternVideo2.5 te saneren. Draai run_internvideo25_test.bat "
            "--upgrade of pip install protobuf.") from exc

    target = Path(str(cfg.get(
        "internvideo25_tokenizer_dir",
        "models/internvideo25_tokenizer"))).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    file_names = (
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenization_internlm2.py",
        "config.json",
    )
    for name in file_names:
        try:
            source = hf_hub_download(model_id, name)
        except Exception:  # noqa: BLE001 - optioneel bestand mag ontbreken
            continue
        shutil.copyfile(source, target / name)

    model_file = target / "tokenizer.model"
    if not model_file.is_file():
        raise SystemExit(
            "Kon tokenizer.model van InternVideo2.5 niet downloaden.")
    model_proto = sentencepiece_model_pb2.ModelProto()
    model_proto.ParseFromString(model_file.read_bytes())
    used = {piece.piece for piece in model_proto.pieces}
    placeholder = "<duplo_null>"
    while placeholder in used:
        placeholder = f"_{placeholder}"
    changed = 0
    for piece in model_proto.pieces:
        if "\x00" in piece.piece:
            piece.piece = placeholder
            changed += 1
    if changed:
        model_file.write_bytes(model_proto.SerializeToString())
    return target


def _load_internvideo25_tokenizer(model_id: str, cfg: dict):
    """Tokenizer met fallback voor de NUL-byte in InternVideo2.5's model."""
    try:
        return _load_remote_tokenizer(model_id)
    except Exception as exc:  # noqa: BLE001 - remote tokenizers verschillen
        message = str(exc)
        if ("null character" not in message
                and "tokenizer kon niet geladen worden" not in message):
            raise
    path = _sanitize_internvideo25_tokenizer(model_id, cfg)
    return _load_remote_tokenizer(str(path))


def _flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        import torch
    except (ImportError, OSError, RuntimeError):
        return False
    return bool(torch.cuda.is_available())


def _flash_attention_enabled(cfg: dict, torch_device: str) -> bool:
    if not _flash_attention_available():
        return False
    import torch

    return _torch_dtype(cfg, torch_device) in (torch.float16, torch.bfloat16)


def _torch_dtype(cfg: dict, torch_device: str):
    """Modeldtype: bf16 op CUDA en fp32 op CPU, tenzij expliciet gezet."""
    import torch

    name = str(cfg.get("vlm_dtype", "auto")).strip().lower()
    if name in ("", "auto"):
        return torch.bfloat16 if torch_device == "cuda" else torch.float32
    if not hasattr(torch, name):
        raise SystemExit(f"Onbekende vlm_dtype: {cfg.get('vlm_dtype')!r}.")
    return getattr(torch, name)


def _load_pretrained(model_class, model_id: str, cfg: dict,
                     torch_device: str, **extra):
    """Laad een HF-model met dtype/device_map en een transformers 4.x-fallback."""
    from transformers.configuration_utils import PretrainedConfig

    dtype = _torch_dtype(cfg, torch_device)
    kwargs = {"dtype": dtype, **extra}
    if torch_device == "cuda":
        kwargs["device_map"] = str(cfg.get("vlm_device_map", "auto"))
    original_repr = PretrainedConfig.__repr__
    PretrainedConfig.__repr__ = lambda self: self.__class__.__name__
    try:
        try:
            return model_class.from_pretrained(model_id, **kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            # Oudere transformers-versies noemen dit torch_dtype.
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_class.from_pretrained(model_id, **kwargs)
    finally:
        PretrainedConfig.__repr__ = original_repr


def _prepare_loaded_model(model, cfg: dict, torch_device: str) -> tuple:
    """Zet het model in eval-modus en bepaal invoerdevice en -dtype."""
    dtype = _torch_dtype(cfg, torch_device)
    if torch_device != "cuda":
        model.to("cpu")
    elif not getattr(model, "hf_device_map", None):
        # Zonder device_map kan de dtype nog ontbreken; met accelerate-offload
        # mag model.to() niet meer worden aangeroepen.
        try:
            model.to(dtype=dtype)
        except (RuntimeError, ValueError):
            pass
    model.eval()
    input_device = getattr(model, "device", torch_device)
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = dtype
    return input_device, model_dtype


def _move_inputs(inputs, device, dtype) -> dict:
    """Verplaats tensorinvoer naar het model en filter processor-metadata."""
    import torch

    skipped = {"video_metadata", "rows", "cols"}
    moved = {}
    for key, value in inputs.items():
        if key in skipped:
            continue
        if torch.is_tensor(value):
            if value.is_floating_point():
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
        moved[key] = value
    return moved


def _clip_times(clip_start_s: float | None,
                clip_end_s: float | None) -> tuple[float, float]:
    start = 0.0 if clip_start_s is None else float(clip_start_s)
    end = start + 1.0 if clip_end_s is None else float(clip_end_s)
    return start, max(start, end)


def _pil_frames(frames):
    """BGR-numpyframes naar RGB-PIL-afbeeldingen."""
    import cv2
    from PIL import Image

    return [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in frames]


def _video_metadata(frames, clip_start_s: float, clip_end_s: float):
    """Metadata voor processors die frametimestamps in de prompt zetten.

    De frames zijn al gelijkmatig over de clip verdeeld. Door ``fps`` zo te
    kiezen dat de gehele clip wordt afgedekt, kloppen de timestamps in de
    processor-prompt ongeveer met de cliprelatieve tijd.
    """
    from transformers.video_utils import VideoMetadata

    count = len(frames)
    duration = max(0.0, clip_end_s - clip_start_s)
    fps = ((count - 1) / duration) if duration > 0 and count > 1 else 1.0
    return VideoMetadata(
        total_num_frames=count,
        fps=fps,
        duration=duration,
        frames_indices=list(range(count)),
    )


def _decode_generated(processor, generated, input_length: int) -> str:
    tokens = generated[:, input_length:]
    texts = processor.batch_decode(
        tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return texts[0].strip()


class _BaseVlmActionClassifier:
    """Gedeelde configuratie en promptopbouw voor generatieve VLM's."""

    temporal = True

    def _configure(self, cfg: dict, device: str, kind: str, model_key: str,
                   default_model: str) -> None:
        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.kind = kind
        self.torch_device = "cuda" if device == "gpu" else "cpu"
        self.model_id = str(cfg.get(f"{model_key}_model") or default_model)
        self.prompt_set = _prompt_set_for(cfg, model_key,
                                          DEFAULT_ACTION_TEMPLATES)
        self.top_k = int(cfg.get("action_top_k", 3))
        self.max_new_tokens = max(1, int(cfg.get("vlm_max_new_tokens", 256)))
        self.num_frames = max(1, int(cfg.get(
            "vlm_frames", cfg.get("action_frames", 8))))
        self.input_device = self.torch_device
        self.model_dtype = None
        self.attn_implementation = None

    @property
    def classes(self) -> tuple[str, ...]:
        return self.prompt_set.classes

    def _parse(self, text: str, clip_start_s: float,
               clip_end_s: float, top_k: int | None = None):
        return parse_vlm_action_response(
            text, self.classes, clip_start_s, clip_end_s)

    def _prompt(self, clip_start_s: float, clip_end_s: float) -> str:
        return build_vlm_prompt(self.classes, clip_start_s, clip_end_s)


class SmolVlm2Classifier(_BaseVlmActionClassifier):
    """SmolVLM2 (2.2B/500M/256M) via de transformers chat-template-API."""

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._configure(cfg, device, "smolvlm2", "smolvlm2",
                        _SMOLVLM2_MODEL)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        load_kwargs = {}
        if _flash_attention_enabled(cfg, self.torch_device):
            load_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = _load_pretrained(
            AutoModelForImageTextToText, self.model_id, cfg,
            self.torch_device, **load_kwargs)
        self.attn_implementation = getattr(
            self.model.config, "_attn_implementation", None)
        self.input_device, self.model_dtype = _prepare_loaded_model(
            self.model, cfg, self.torch_device)

    def predict(self, frames, top_k: int | None = None,
                clip_start_s: float | None = None,
                clip_end_s: float | None = None):
        import torch

        start, end = _clip_times(clip_start_s, clip_end_s)
        pil_frames = _pil_frames(frames)
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames},
                {"type": "text", "text": self._prompt(start, end)},
            ],
        }]
        processor_kwargs = {
            "do_sample_frames": False,
            "video_metadata": [_video_metadata(frames, start, end)],
        }
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs=processor_kwargs)
        inputs = _move_inputs(inputs, self.input_device, self.model_dtype)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = _decode_generated(self.processor, generated, input_length)
        return self._parse(text, start, end, top_k)


def _find_closest_aspect_ratio(aspect_ratio: float,
                               target_ratios: list[tuple[int, int]],
                               width: int, height: int,
                               image_size: int) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, image_size: int = 448, max_num: int = 1,
                        use_thumbnail: bool = False):
    """InternVL-dynamic-preprocess: één of meer vierkante tiles per frame."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(1, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if 1 <= i * j <= max_num},
        key=lambda item: item[0] * item[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for index in range(blocks):
        box = ((index % (target_width // image_size)) * image_size,
               (index // (target_width // image_size)) * image_size,
               ((index % (target_width // image_size)) + 1) * image_size,
               ((index // (target_width // image_size)) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _internvl_transform(input_size: int):
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose([
        transforms.Lambda(
            lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((input_size, input_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=_INTERNVIDEO25_MEAN,
                             std=_INTERNVIDEO25_STD),
    ])


class InternVideo25Classifier(_BaseVlmActionClassifier):
    """InternVideo2.5-Chat-8B via InternVL remote code met eigen greedy-decode."""

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        import os

        import torch
        import transformers
        from transformers import AutoModel

        if (int(transformers.__version__.split(".")[0]) >= 5
                and os.environ.get("DUPLO_INTERNVIDEO25_ISOLATED") != "1"):
            raise SystemExit(
                "InternVideo2.5 vereist de transformers 4.47-runtime; start "
                "deze stap via run_internvideo25_test.bat of "
                "src/run_action_test.py (die isoleren automatisch).")

        self._configure(cfg, device, "internvideo25", "internvideo25",
                        _INTERNVIDEO25_MODEL)
        _patch_transformers_for_remote_code()
        config = _load_remote_config(self.model_id)
        self._configure_attention(config, cfg, self.torch_device)
        self.attn_implementation = getattr(
            getattr(config, "llm_config", None), "attn_implementation",
            "eager")
        self.tokenizer = _load_internvideo25_tokenizer(self.model_id, cfg)
        self.model = _load_pretrained(
            AutoModel, self.model_id, cfg, self.torch_device,
            trust_remote_code=True, config=config)
        self.input_device, self.model_dtype = _prepare_loaded_model(
            self.model, cfg, self.torch_device)
        self.num_local_frames = max(
            1, int(getattr(self.model, "local_num_frames", 4)))
        self.image_size = int(getattr(config, "force_image_size", 448) or 448)
        self.max_num = max(
            1, int(cfg.get("internvideo25_max_num", 1)))
        self.transform = _internvl_transform(self.image_size)
        self._torch = torch

    @staticmethod
    def _configure_attention(config, cfg: dict, torch_device: str) -> None:
        """Gebruik flash-attn als die beschikbaar is, anders eager attention."""
        requested = str(cfg.get(
            "internvideo25_attn_implementation", "auto")).strip().lower()
        if requested in ("", "auto"):
            use_flash = _flash_attention_enabled(cfg, torch_device)
        else:
            use_flash = (requested in ("flash_attention_2", "flash")
                         and _flash_attention_enabled(cfg, torch_device))
        implementation = "flash_attention_2" if use_flash else "eager"
        llm_config = getattr(config, "llm_config", None)
        if llm_config is not None:
            llm_config.attn_implementation = implementation
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            vision_config.use_flash_attn = use_flash
        try:
            config._attn_implementation = implementation
        except (AttributeError, ValueError):
            pass

    def _load_image(self, frame):
        from PIL import Image
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tiles = _dynamic_preprocess(
            image, image_size=self.image_size, max_num=self.max_num,
            use_thumbnail=self.max_num > 1)
        return self._torch.stack([self.transform(tile) for tile in tiles])

    def _build_query(self, question: str, num_patches_list) -> str:
        import copy

        template = copy.deepcopy(self.model.conv_template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for num_patches in num_patches_list:
            image_tokens = (
                "<img>"
                + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches
                + "</img>")
            query = query.replace("<image>", image_tokens, 1)
        return query

    def _generate(self, pixel_values, question: str,
                  num_patches_list) -> str:
        """Eigen greedy-decode tegen de remote forward (zonder GenerationMixin)."""
        import torch

        tokenizer = self.tokenizer
        model = self.model
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>")
        query = self._build_query(question, num_patches_list)
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(model.device)
        attention_mask = model_inputs["attention_mask"].to(model.device)
        with torch.inference_mode():
            vit_embeds = model.extract_feature(pixel_values)
            input_embeds = model.language_model.get_input_embeddings()(
                input_ids)
            selected = input_ids == model.img_context_token_id
            expected = vit_embeds.shape[0] * vit_embeds.shape[1]
            if int(selected.sum()) != expected:
                raise SystemExit(
                    "InternVideo2.5: aantal beeldtokens komt niet overeen "
                    f"({int(selected.sum())} vs {expected}).")
            input_embeds[selected] = vit_embeds.reshape(
                -1, vit_embeds.shape[-1]).to(
                    device=input_embeds.device, dtype=input_embeds.dtype)

        eos_ids = {tokenizer.eos_token_id}
        for token in ("<|im_end|>", "</s>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                eos_ids.add(token_id)

        generated: list[int] = []
        past_key_values = None
        current_embeds = input_embeds
        next_token = None
        with torch.inference_mode():
            for step in range(self.max_new_tokens):
                outputs = model.language_model(
                    input_ids=None if step == 0 else next_token,
                    inputs_embeds=current_embeds if step == 0 else None,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True)
                next_token = torch.argmax(
                    outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
                token_id = int(next_token.item())
                if token_id in eos_ids:
                    break
                generated.append(token_id)
                past_key_values = outputs.past_key_values
                attention_mask = torch.cat(
                    [attention_mask,
                     attention_mask.new_ones((1, 1))], dim=-1)
                current_embeds = None
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def predict(self, frames, top_k: int | None = None,
                clip_start_s: float | None = None,
                clip_end_s: float | None = None):
        start, end = _clip_times(clip_start_s, clip_end_s)
        pixel_values = [self._load_image(frame) for frame in frames]
        num_patches_list = [int(value.shape[0]) for value in pixel_values]
        pixel_values = self._torch.cat(pixel_values, dim=0)
        # InternVideo2.5 comprimeert telkens local_num_frames frames samen.
        padding = (-len(num_patches_list)) % self.num_local_frames
        if padding:
            pixel_values = self._torch.cat(
                [pixel_values,
                 pixel_values[-1:].repeat(padding, 1, 1, 1)], dim=0)
            num_patches_list.extend([num_patches_list[-1]] * padding)
        pixel_values = pixel_values.to(
            device=self.input_device, dtype=self.model_dtype)
        video_prefix = "".join(
            f"Frame{index + 1}: <image>\n"
            for index in range(len(num_patches_list)))
        question = video_prefix + self._prompt(start, end)
        text = self._generate(pixel_values, question, num_patches_list)
        return self._parse(text, start, end, top_k)


class InternVideo3Classifier(_BaseVlmActionClassifier):
    """InternVideo3-8B-Instruct via de Qwen-VL-achtige processor."""

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._configure(cfg, device, "internvideo3", "internvideo3",
                        _INTERNVIDEO3_MODEL)
        _patch_transformers_for_remote_code()
        _patch_internvideo3_transformers()
        config = _load_remote_config(self.model_id)
        _patch_missing_pad_token(config)
        attention = str(cfg.get("vlm_attn_implementation", "sdpa"))
        if attention.strip().lower() in ("", "auto"):
            attention = ("flash_attention_2"
                         if _flash_attention_enabled(cfg, self.torch_device)
                         else "sdpa")
        self.attn_implementation = attention
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)
        self.model = _load_pretrained(
            AutoModelForCausalLM, self.model_id, cfg, self.torch_device,
            trust_remote_code=True, config=config,
            attn_implementation=attention)
        self.input_device, self.model_dtype = _prepare_loaded_model(
            self.model, cfg, self.torch_device)
        # Beperk het totale aantal videopixels, anders maakt de processor voor
        # elk frame bijna de volledige resolutie (~duizenden tokens per clip).
        max_pixels = max(1, int(cfg.get(
            "internvideo3_max_pixels", 128 * 32 * 32)))
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is not None:
            video_processor.size = {
                "longest_edge": max_pixels * self.num_frames,
                "shortest_edge": max_pixels,
            }

    def predict(self, frames, top_k: int | None = None,
                clip_start_s: float | None = None,
                clip_end_s: float | None = None):
        import torch

        start, end = _clip_times(clip_start_s, clip_end_s)
        pil_frames = _pil_frames(frames)
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames},
                {"type": "text", "text": self._prompt(start, end)},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text, videos=[pil_frames],
            video_metadata=[_video_metadata(frames, start, end)],
            do_sample_frames=False, return_tensors="pt")
        inputs = _move_inputs(inputs, self.input_device, self.model_dtype)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, use_cache=True)
        output = _decode_generated(self.processor, generated, input_length)
        return self._parse(output, start, end, top_k)


def create_vlm_classifier(cfg: dict, device: str, kind: str):
    if kind == "smolvlm2":
        return SmolVlm2Classifier(cfg, device)
    if kind == "internvideo25":
        return InternVideo25Classifier(cfg, device)
    if kind == "internvideo3":
        return InternVideo3Classifier(cfg, device)
    raise SystemExit(f"Onbekend VLM-actiemodel: {kind!r}.")
