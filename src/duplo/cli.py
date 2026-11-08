"""Command-line interface voor de pipeline-scripts."""
import argparse

from .config import DEVICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", choices=DEVICES,
                        help="cpu of gpu; overschrijft 'device' uit de config")
    # Bat-level flag (run_common.bat): .bat-scripts geven alle argumenten
    # met %* door, inclusief --upgrade. Batch splitst --upgrade=<waarde> in
    # twee tokens, dus filteren in .bat is onbetrouwbaar; accepteer de flag
    # hier en negeer hem (herinstallatie gebeurt in run_common.bat, of via
    # 'auto_install_torch: true' bij direct Python-gebruik).
    parser.add_argument("--upgrade", nargs="?", const=True, default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()
