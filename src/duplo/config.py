"""Configuratie laden en de effectieve device-keuze bepalen."""
from pathlib import Path

import yaml

DEVICES = ("cpu", "gpu")


def load_config(cfg_path: str) -> dict:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"Configuratiebestand is leeg of ongeldig: {cfg_path}")
    return cfg


def resolve_device(cli_device: str | None, cfg: dict) -> str:
    """Het --device argument heeft voorrang op de 'device' waarde in de config."""
    return cli_device or cfg.get("device", "cpu")
