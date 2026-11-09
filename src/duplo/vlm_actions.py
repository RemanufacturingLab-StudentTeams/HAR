"""Generatieve VLM's als actieclassifiers.

SmolVLM2 en Qwen3-VL zijn algemene video-language-modellen, geen
gespecialiseerde temporele actieclassifiers. Ze krijgen daarom dezelfde
clip-beelden plus een strak JSON-prompt voor zowel het actielabel als de
begin-/eindtijd. LLaVAction is wel op acties gefinetuned en gebruikt zijn
eigen LLaVA-Qwen inferencepad.

De temporele voorspellingen zijn clip-relatief (0 = start van de clip) en
worden hier omgerekend naar absolute seconden. Of dat betrouwbaar is, meet
``duplo.action_eval`` apart tegen ground truth.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from duplo.actions import (
    DEFAULT_ACTION_TEMPLATES,
    _patch_transformers_for_remote_code,
    _prompt_set_for,
    build_vlm_prompt,
    parse_vlm_action_response,
)

_LLAVACTION_MODEL = "MLAdaptiveIntelligence/LLaVAction-7B"
_SMOLVLM2_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
_QWEN3VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _torch_dtype(cfg: dict, torch_device: str):
    name = str(cfg.get("vlm_dtype", "auto")).strip().lower()
    if name in ("", "auto"):
        return torch.float32 if torch_device != "cuda" else torch.bfloat16
    if not hasattr(torch, name):
        raise SystemExit(f"Onbekende vlm_dtype: {cfg.get('vlm_dtype')!r}.")
    return getattr(torch, name)


def _load_generative_model(model_id: str, torch_device: str, cfg: dict):
    from transformers import AutoProcessor

    classes = []
    try:
        from transformers import AutoModelForImageTextToText as AutoModel
        classes.append(AutoModel)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForMultimodalLM as AutoModel
        classes.append(AutoModel)
    except ImportError:
        pass
    if not classes:
        raise SystemExit(
            "Deze transformers-versie heeft geen AutoModelForImageTextToText/"
            "AutoModelForMultimodalLM. Installeer transformers 5.x.")
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = _torch_dtype(cfg, torch_device)
    load_kwargs = {"dtype": dtype}
    if torch_device == "cuda":
        load_kwargs["device_map"] = "auto"
    last_error = None
    for model_class in classes:
        try:
            model = model_class.from_pretrained(model_id, **load_kwargs)
            break
        except TypeError:
            # Oudere transformers kent 'dtype' nog niet.
            fallback = {"torch_dtype": dtype}
            if torch_device == "cuda":
                fallback["device_map"] = "auto"
            try:
                model = model_class.from_pretrained(model_id, **fallback)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        raise SystemExit(
            f"Kon VLM {model_id} niet laden: {last_error}. Controleer je "
            "Hugging Face-login/toegang en de transformers-versie.") from last_error
    if torch_device != "cuda":
        model.to("cpu")
    else:
        # Sommige modellen negeren dtype/device_map bij het laden; forceer de
        # gekozen dtype alsnog zodat een 8B-model in bf16 past.
        try:
            model.to(dtype=dtype)
        except (RuntimeError, ValueError):
            pass
    model.eval()
    return processor, model


class GenericVlmActionClassifier:
    """SmolVLM2/Qwen3-VL met een gezamenlijk beeld+tekst JSON-promptpad."""

    temporal = True

    def __init__(self, cfg: dict, device: str = "cpu", kind: str = "qwen3vl"
                 ) -> None:
        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.kind = kind
        self.torch_device = "cuda" if device == "gpu" else "cpu"
        if kind == "smolvlm2":
            self.model_id = str(cfg.get("smolvlm2_model", _SMOLVLM2_MODEL))
            model_key = "smolvlm2"
        else:
            self.model_id = str(cfg.get("qwen3vl_model", _QWEN3VL_MODEL))
            model_key = "qwen3vl"
        self.prompt_set = _prompt_set_for(cfg, model_key,
                                          DEFAULT_ACTION_TEMPLATES)
        self.num_frames = int(cfg.get("vlm_frames",
                                       cfg.get("action_frames", 8)))
        self.max_new_tokens = int(cfg.get("vlm_max_new_tokens", 256))
        self.top_k = int(cfg.get("action_top_k", 3))
        self.processor, self.model = _load_generative_model(
            self.model_id, self.torch_device, cfg)
        self.input_device = getattr(self.model, "device", self.torch_device)
        self.model_dtype = next(self.model.parameters()).dtype

    @property
    def classes(self) -> tuple[str, ...]:
        return self.prompt_set.classes

    def _messages(self, frames, prompt: str):
        import cv2
        from PIL import Image

        content = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            content.append({"type": "image", "image": Image.fromarray(rgb)})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def predict(self, frames, top_k: int | None = None,
                clip_start_s: float | None = None,
                clip_end_s: float | None = None):
        start = 0.0 if clip_start_s is None else float(clip_start_s)
        end = (start + 1.0 if clip_end_s is None else float(clip_end_s))
        prompt = build_vlm_prompt(self.classes, start, end)
        messages = self._messages(frames, prompt)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.input_device)
        else:
            inputs = {key: (value.to(self.input_device)
                            if hasattr(value, "to") else value)
                      for key, value in inputs.items()}
        # Beeldtensoren naar de modeldtype; tekst-ids blijven integers.
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                inputs[key] = value.to(dtype=self.model_dtype)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        input_length = inputs["input_ids"].shape[-1]
        text = self.processor.batch_decode(
            generated[:, input_length:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        return parse_vlm_action_response(text, self.classes, start, end)


def _patch_llavaction_transformers() -> None:
    """Vul utilities aan die transformers 5 uit ``modeling_utils`` verwijderde.

    De LLaVAction-package (0.0.1) is tegen transformers 4.x geschreven. Alleen
    de qformer-module importeert deze drie helpers; de configuratie van
    LLaVAction-7B gebruikt geen resampler, dus de implementaties worden niet
    uitgevoerd. Ze zijn nodig om de package te kunnen importeren.
    """
    import torch
    from transformers import modeling_utils

    _patch_transformers_for_remote_code()

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim,
                                      *input_tensors):
            if not input_tensors:
                return forward_fn()
            if not chunk_size:
                return forward_fn(*input_tensors)
            tensor_shape = input_tensors[0].shape[chunk_dim]
            chunks = []
            for index in range(0, tensor_shape, chunk_size):
                size = min(chunk_size, tensor_shape - index)
                chunks.append(forward_fn(*[
                    tensor.narrow(chunk_dim, index, size)
                    for tensor in input_tensors]))
            return torch.cat(chunks, dim=chunk_dim)

        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size,
                                            already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head -= sum(1 if item < head else 0
                            for item in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            return heads, torch.arange(len(mask))[mask].long()

        modeling_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices)

    if not hasattr(modeling_utils, "prune_linear_layer"):
        def prune_linear_layer(layer, index, dim=0):
            index = index.to(layer.weight.device)
            weight = layer.weight.index_select(dim, index).clone().detach()
            bias = None
            if layer.bias is not None:
                bias = (layer.bias.clone().detach() if dim == 1
                        else layer.bias[index].clone().detach())
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = torch.nn.Linear(
                new_size[1], new_size[0],
                bias=layer.bias is not None).to(layer.weight.device)
            with torch.no_grad():
                new_layer.weight.copy_(weight.contiguous())
                if bias is not None:
                    new_layer.bias.copy_(bias.contiguous())
            return new_layer

        modeling_utils.prune_linear_layer = prune_linear_layer


class LlavaActionClassifier:
    """LLaVAction-7B via de officiële ``llavaction`` inferencebuilder."""

    temporal = True

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.torch_device = "cuda" if device == "gpu" else "cpu"
        if self.torch_device != "cuda":
            raise SystemExit(
                "LLaVAction-7B vereist een CUDA-GPU (~15 GB). De officiële "
                "llavaction-inferencecode ondersteunt geen gewone CPU-run; "
                "gebruik --device gpu.")
        _patch_llavaction_transformers()
        try:
            from llavaction.model.builder import load_pretrained_model
        except ImportError as exc:
            raise SystemExit(
                "Het 'llavaction'-pakket ontbreekt. Draai "
                "run_llavaction_test.bat --upgrade of installeer "
                "requirements.txt.") from exc
        self.model_id = str(cfg.get("llavaction_model", _LLAVACTION_MODEL))
        self.prompt_set = _prompt_set_for(cfg, "llavaction",
                                          DEFAULT_ACTION_TEMPLATES)
        self.num_frames = min(64, int(cfg.get(
            "llavaction_frames", cfg.get("vlm_frames", 8))))
        self.max_new_tokens = int(cfg.get("vlm_max_new_tokens", 256))
        self.top_k = int(cfg.get("action_top_k", 3))
        dtype = str(cfg.get("llavaction_dtype", "bfloat16")).lower()
        if dtype not in ("float16", "bfloat16"):
            raise SystemExit(
                "llavaction_dtype moet 'float16' of 'bfloat16' zijn.")
        attn = str(cfg.get("llavaction_attn_implementation", "sdpa"))
        # Niet 'auto': de oude LLaVA-code gaat uit van een volledig geladen
        # model. Met accelerate-offload blijven parameters op 'meta' staan en
        # crasht de vision tower. Laad daarom in één keer op cuda:0.
        device_map = cfg.get("llavaction_device_map") or {"": 0}
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*non-meta parameter.*")
            warnings.filterwarnings(
                "ignore", message=r".*device_map keys do not match.*")
            self.tokenizer, self.model, self.image_processor, _ = (
                load_pretrained_model(
                    self.model_id, None, "llava_qwen",
                    torch_dtype=dtype, device_map=device_map,
                    attn_implementation=attn))
        self.model.eval()
        self.input_device = getattr(self.model, "device", self.torch_device)
        self.model_dtype = next(self.model.parameters()).dtype

    @property
    def classes(self) -> tuple[str, ...]:
        return self.prompt_set.classes

    def predict(self, frames, top_k: int | None = None,
                clip_start_s: float | None = None,
                clip_end_s: float | None = None):
        import cv2
        from llavaction.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llavaction.conversation import conv_templates
        from llavaction.mm_utils import tokenizer_image_token

        start = 0.0 if clip_start_s is None else float(clip_start_s)
        end = (start + 1.0 if clip_end_s is None else float(clip_end_s))
        duration = max(0.0, end - start)
        rgb_frames = np.stack([
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in frames], axis=0)
        pixel_values = self.image_processor.preprocess(
            rgb_frames, return_tensors="pt")["pixel_values"]
        pixel_values = [pixel_values.to(
            device=getattr(self.model, "device", self.torch_device),
            dtype=self.model_dtype)]
        time_instruction = (
            f"The clip starts at {start:.2f} seconds and lasts "
            f"{duration:.2f} seconds; {len(frames)} frames are uniformly "
            "sampled from it. ")
        question = (DEFAULT_IMAGE_TOKEN + "\n" + time_instruction + "\n"
                    + build_vlm_prompt(self.classes, start, end))
        conversation = copy.deepcopy(conv_templates["qwen_1_5"])
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors="pt").unsqueeze(0).to(
                getattr(self.model, "device", self.torch_device))
        with torch.inference_mode():
            output = self.model.generate(
                input_ids, images=pixel_values, modalities=["video"],
                do_sample=False, temperature=0,
                max_new_tokens=self.max_new_tokens)
        text = self.tokenizer.batch_decode(
            output, skip_special_tokens=True)[0].strip()
        return parse_vlm_action_response(text, self.classes, start, end)


def create_vlm_classifier(cfg: dict, device: str, kind: str):
    if kind in ("smolvlm2", "qwen3vl"):
        return GenericVlmActionClassifier(cfg, device, kind)
    if kind == "llavaction":
        return LlavaActionClassifier(cfg, device)
    raise SystemExit(f"Onbekend VLM-actiemodel: {kind!r}.")
