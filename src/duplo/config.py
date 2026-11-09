"""Configuratie laden, device-keuze en invoervideo vinden."""
from pathlib import Path

import yaml

DEVICES = ("auto", "cpu", "gpu")


def load_config(cfg_path: str) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"Configuratiebestand is leeg of ongeldig: {cfg_path}")
    return cfg


def cuda_available() -> bool:
    """True als de geïnstalleerde torch-build een werkende NVIDIA-GPU ziet."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - ontbrekende of beschadigde torch-build
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - defecte CUDA-driver
        return False


def resolve_device(cli_device: str | None, cfg: dict) -> str:
    """Het --device argument heeft voorrang op de 'device' waarde in de config.

    ``auto`` (de standaard) kiest GPU als CUDA beschikbaar is en anders CPU,
    zodat dezelfde config op een laptop zonder NVIDIA-GPU werkt.
    """
    device = str(cli_device or cfg.get("device", "auto")).strip().lower()
    if device == "auto":
        return "gpu" if cuda_available() else "cpu"
    if device not in DEVICES:
        raise SystemExit(
            f"Onbekend device: {device!r}. Kies uit: {', '.join(DEVICES)}.")
    return device


def resolve_video_path(cfg: dict) -> Path:
    """Controleer ``video_path`` en val terug op de enige video in ``data/``.

    Een verse clone bevat geen ``data/*.mp4`` (die staat in .gitignore). Dit
    geeft een begrijpelijke melding in plaats van een OpenCV-fout, en kiest
    automatisch de video als er precies één in ``data/`` staat.
    """
    configured = str(cfg.get("video_path", "")).strip()
    path = Path(configured) if configured else Path()
    if configured and path.is_file():
        return path
    candidates = sorted(Path("data").glob("*.mp4"))
    if len(candidates) == 1:
        print(f"Video niet gevonden op '{configured}'; gebruik "
              f"'{candidates[0]}'.")
        cfg["video_path"] = str(candidates[0])
        return candidates[0]
    if candidates:
        names = ", ".join(candidate.name for candidate in candidates)
        raise SystemExit(
            f"Video '{configured}' bestaat niet en data/ bevat meerdere "
            f"video's ({names}). Zet video_path in de config.")
    raise SystemExit(
        f"Video '{configured or '(leeg)'}' bestaat niet en data/ bevat geen "
        "mp4-bestand.\nZet een video in data/ (bijvoorbeeld "
        "data/Disassembly1.mp4) of pas video_path in de config aan.")
