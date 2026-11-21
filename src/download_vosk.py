"""Download en pak het Nederlandse Vosk-model uit in de map models/ (eenmalig, ~40 MB)."""
import urllib.request
import zipfile
from pathlib import Path

NAAM = "vosk-model-small-nl-0.22"
URL = f"https://alphacephei.com/vosk/models/{NAAM}.zip"
models = Path(__file__).resolve().parents[1] / "models"  # models/ in de projectmap
doel = models / NAAM

if doel.exists():
    print(f"Al aanwezig: {doel}")
else:
    models.mkdir(exist_ok=True)
    zip_pad = models / f"{NAAM}.zip"
    print(f"Downloaden: {URL}")
    urllib.request.urlretrieve(URL, zip_pad)
    with zipfile.ZipFile(zip_pad) as z:
        z.extractall(models)
    zip_pad.unlink()
    print(f"Klaar: {doel}")
