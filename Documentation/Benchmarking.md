# Benchmarken per modeltaak

Dit document beschrijft hoe de modeltesten worden gemeten en vergeleken. Het
project bevat meerdere soorten modellen; ze mogen niet allemaal met dezelfde
detectiescore worden beoordeeld.

## 1. Taken en uitvoer

| Taak | Modellen | Uitvoer | Belangrijkste metrics |
| --- | --- | --- | --- |
| `object_detection` | YOLO, RF-DETR, Grounding DINO, YOLO-World, YOLO-seg, RF-DETR-seg, SAM3, SAM 3.1, ByteTrack-varianten | `*_detections.csv` / `*_tracking_history.csv` | tooldekking, tool-frames, confidence, fps |
| `hand_detection` | MediaPipe Hands | `hand_landmarks.csv` | frames met handen, landmarkrijen, fps |
| `3d_hand_prediction` | VITRA | `vitra_actions.csv` | frames, actierijen, handen, chunk, fps |
| `action_classification` | X-CLIP, InternVideo2, ActionCLIP, SmolVLM2, LLaVAction, Qwen3-VL | `*_actions.csv` | clips, labels, confidence, classificatie-accuracy, temporele IoU |

LLaVAction draait vanwege de oude `llavaction`-package in een aparte
transformers 4.x-runtime (`third_party/llavaction_runtime`, geïnstalleerd door
`run_llavaction_test.bat`) en vereist een CUDA-GPU. Qwen3-VL en SmolVLM2 zijn
algemene VLM's; meet hun classificatie en temporele grenzen expliciet apart.

Elke run schrijft daarnaast `<model>_summary.json`. Sinds benchmarkversie 2
bevat die samenvatting:

- `task` en `task_label`;
- `subtask` (bijvoorbeeld `tracking`);
- `environment` met platform, Python, torch, CUDA en GPU-naam;
- timing (mean/mediaan/p95/min/max, inference-fps en pipeline-fps);
- het aantal samples en de gebruikte CSV.

## 2. Rapport maken

```bat
run_benchmark.bat
run_compare_models.bat
run_benchmark.bat --task action_classification
run_benchmark.bat --task hand_detection
```

`src/compare_models.py` schrijft naar `outputs/<video>/comparison/`:

- `model_comparison.md` – leesbaar rapport met een overzicht per taak;
- `model_comparison.csv` – detectiemodellen;
- `tracking_comparison.csv` – tracking (indien aanwezig);
- `action_comparison.csv` – actieclassificatie (indien aanwezig);
- `benchmark_overview.json` – machineleesbaar overzicht per taak plus
  hardware/softwaremetadata.

De samengestelde score blijft alleen voor objectdetectie gelden. Handdetectie,
3D-handvoorspelling en actieclassificatie hebben eigen tabellen en eigen
metrics.

## 3. Actieclassificatie en temporele localisatie

Gespecialiseerde classifiers (X-CLIP, InternVideo2, ActionCLIP) geven alleen
een label per clip. Generatieve VLM's (SmolVLM2, Qwen3-VL) en LLaVAction
kunnen ook begin- en eindtijden genereren. Die twee prestaties worden apart
gemeten.

### Ground truth

Zet in `configs/baseline.yaml`:

```yaml
action_ground_truth: "data/action_ground_truth.csv"
```

Het CSV-bestand heeft deze kolommen:

```csv
video,action,start_s,end_s
Disassembly1.mp4,unscrewing a screw,12.50,18.20
Disassembly1.mp4,picking up a tool,18.20,21.00
Disassembly1.mp4,idle,21.00,24.50
```

- `video` is optioneel; leeg betekent: geldt voor elke video.
- `action`/`klasse` moet overeenkomen met `action_classes` (hoofdletter-
  ongevoelig).
- JSON met dezelfde velden wordt ook ondersteund.

### Metrics

`src/duplo/action_eval.py` berekent:

- **classificatie-accuracy** – per clip wordt de ground-truth-actie met de
  grootste overlap met het clipvenster gekozen; het voorspelde label moet
  daarmee overeenkomen;
- **temporele coverage** – aandeel clips waarvoor het model start/end gaf;
- **temporele IoU (mIoU)** – gemiddelde IoU van één-op-één gematchte
  voorspelde en ground-truth-intervallen;
- **R@IoU 0.3/0.5/0.7** – recall bij vaste IoU-drempels;
- **start/end MAE** – gemiddelde absolute fout van de grenzen in seconden;
- **temporele labelaccuracy** – labelcorrectheid van de gematchte intervallen.

De temporele voorspellingen van een VLM zijn beperkt tot het clipvenster. Een
actie die over meerdere clips loopt wordt niet automatisch samengevoegd; de
metrics zijn daarom een indicatie en geen vervanging van een gespecialiseerde
temporal-action-localizer.

## 4. Reproduceerbaarheid

- Gebruik dezelfde video, `frame_stride`, clipinstellingen en prompts.
- De eerste inferentie kan trager zijn door modeldownload/cache; meet bij
  voorkeur een tweede run.
- `environment` in de summary legt hardware en software vast.
- Vergelijk geen confidence tussen modelfamilies: elke familie kalibreert
  anders. Generatieve VLM's geven vaak helemaal geen confidence.
- Draai `run_benchmark_suite.bat` voor unittests + benchmarkrapport in één
  keer.
