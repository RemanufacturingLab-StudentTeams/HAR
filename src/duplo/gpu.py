"""Detectie en installatie van CUDA-ondersteuning voor torch."""
import contextlib
import subprocess
import sys

CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def cuda_available() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "True"


def to_cuda_device(device: str) -> str:
    """Map `cpu/gpu` (config/CLI) naar `cpu/cuda` (torch/ultralytics)."""
    return "cuda" if device == "gpu" else "cpu"


def ensure_cuda_torch(auto_install: bool = False) -> None:
    """Controleer CUDA-torch; installeer alleen met expliciete opt-in.

    Zonder `auto_install` volgt een duidelijke fout met het installatie-
    commando i.p.v. een ~2GB pip-install mid-run. Met
    `auto_install=True` (opt-in via `auto_install_torch: true` in de config)
    blijft het oude gedrag behouden.

    Draait installatie in een subprocess zodat een reeds geladen CPU-build
    het vervangen van de DLL's niet blokkeert; importeer torch/ultralytics
    pas hierna.
    """
    if cuda_available():
        return
    if not auto_install:
        raise SystemExit(
            "GPU gevraagd, maar CUDA-torch is niet beschikbaar. "
            "Installeer handmatig met: "
            f'pip install -U torch torchvision --index-url {CUDA_TORCH_INDEX} '
            "of zet 'auto_install_torch: true' in de config voor automatische "
            "installatie (download ~2 GB, kan enkele minuten duren)."
        )
    print("CUDA-torch niet gevonden; CUDA-build van torch wordt geinstalleerd "
          "(kan enkele minuten duren)...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-U", "torch", "torchvision",
        "--index-url", CUDA_TORCH_INDEX,
    ])
    if result.returncode != 0 or not cuda_available():
        raise SystemExit(
            "GPU gevraagd, maar CUDA is niet beschikbaar "
            "(installatie mislukt of geen NVIDIA-GPU aanwezig)."
        )


def setup_cuda_optimizations(device: str) -> None:
    """Veilige, detectie-neutrale GPU-instellingen (o.a. RTX 5080/Blackwell).

    Schakelt TF32 + cudnn-benchmark + hoge matmul-precisie in. Dit verandert
    geen drempels, imgsz of modelgewichten; het versnelt alleen kernels.
    No-op op CPU of zonder CUDA. Nooit een harde fout: bij problemen volgt
    een waarschuwing en draait de pipeline met standaardinstellingen verder.
    """
    if device != "gpu":
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        # TF32 is op Blackwell/Hopper/Ampere vrijwel verliesvrij voor
        # detectie en scheelt meetbaar in conv/matmul-tijd.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except (AttributeError, RuntimeError):
            pass
        # Kies per vorm de snelste cudnn-kernel; frames hebben een vaste
        # resolutie, dus een eenmalige autotune volstaat (geen urenlange sweep).
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cudnn.deterministic = False
        except (AttributeError, RuntimeError):
            pass
    except Exception as exc:  # pragma: no cover - alleen een waarschuwing
        print(f"Waarschuwing: CUDA-optimalisaties overgeslagen ({exc}).")


@contextlib.contextmanager
def quiet_third_party_init():
    """Onderdruk bekende onschuldige third-party meldingen bij model-init.

    RF-DETR's bouw + ``model.inference()`` (trace/compile-route) produceert
    eenmalig veel ``TracerWarning``-regels uit torch (data-afhankelijke
    vertakkingen die als constanten worden getracet), ``torch.jit``-
    deprecation-meldingen en rfdetr-logregels (o.a. over niet-geladen
    DINOv2-gewichten, onschuldig bij inferentie met COCO-gewichten), en laadt
    intern een transformers-config zonder ``loss_type`` (irrelevant bij
    inferentie). Alles is cosmetisch: detecties veranderen er niet door.
    Echte fouten (exceptions, logging op ERROR-niveau) komen hier altijd nog
    doorheen; alleen de waarschuwingsuitvoer wordt tijdens het ``with``-blok
    gedempt en daarna hersteld.
    """
    import logging
    import os
    import warnings

    try:
        import torch
        tracer_warning = torch.jit.TracerWarning
    except (ImportError, AttributeError):
        class _NeverRaisedWarning(Warning):
            pass
        tracer_warning = _NeverRaisedWarning
    previous_verbosity = None
    try:
        from transformers import logging as hf_logging
        previous_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    except Exception:
        previous_verbosity = None
    # rfdetr leest LOG_LEVEL bij het aanmaken van zijn logger; zonder dit
    # zet de constructor het niveau zelf terug naar INFO.
    previous_log_level = os.environ.get("LOG_LEVEL")
    os.environ["LOG_LEVEL"] = "ERROR"
    rfdetr_logger = logging.getLogger("rf-detr")
    previous_level = rfdetr_logger.level
    rfdetr_logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", tracer_warning)
        warnings.filterwarnings("ignore", message=r".*torch\.jit.*",
                                category=FutureWarning)
        try:
            yield
        finally:
            rfdetr_logger.setLevel(previous_level)
            if previous_log_level is None:
                os.environ.pop("LOG_LEVEL", None)
            else:
                os.environ["LOG_LEVEL"] = previous_log_level
            if previous_verbosity is not None:
                try:
                    from transformers import logging as hf_logging
                    hf_logging.set_verbosity(previous_verbosity)
                except Exception:
                    pass
