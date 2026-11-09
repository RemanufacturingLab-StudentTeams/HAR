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
| `action_classification` | X-CLIP, InternVideo2, ActionCLIP, SmolVLM2, InternVideo2.5, InternVideo3 | `*_actions.csv` | clips, labels, confidence, classificatie-accuracy, temporele IoU |

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

## 3. Geselecteerde modellen en meerdere video's

`src/benchmark_compare.py` is de orchestrator voor benchmarks waarbij je zelf
de modellen en video's kiest:

```bat
run_benchmark_compare.bat --preset action --video data\Disassembly1.mp4
run_benchmark_compare.bat --preset vlm --video data\Disassembly1.mp4
run_benchmark_compare.bat --preset detection --video data\Disassembly1.mp4
run_benchmark_compare.bat --model xclip --videos data\v1.mp4 data\v2.mp4
```

- `--preset action` = ActionCLIP vs InternVideo2 vs X-CLIP.
- `--preset vlm`/`generative` = SmolVLM2 vs InternVideo2.5 vs InternVideo3.
- `--preset detection` = Grounding DINO vs SAM 3.1 vs YOLO-World.
- `--model MODEL --videos ...` draait één model op meerdere video's en
  berekent het gemiddelde per numerieke metric.
- `--models a,b,c` en `--videos "data\*.mp4"` ondersteunen eigen selecties.
- `--reuse` hergebruikt bestaande CSV's; `--save-videos` bewaart debugvideo's.
- `--dry-run` toont alleen de commando's.

De orchestrator schrijft automatisch naar `outputs/benchmarks/<naam>/`:
`benchmark_report.md`, `benchmark_metrics.csv`, `benchmark_average.csv`,
`benchmark_overview.json` en per video een gedetailleerd
`<video>/comparison/model_comparison.md`. De gemiddelde tabel gebruikt alleen
geslaagde runs; mislukte runs staan expliciet in het rapport en in
`benchmark_overview.json`.

## 4. Actieclassificatie en temporele localisatie

Gespecialiseerde classifiers (X-CLIP, InternVideo2, ActionCLIP) geven alleen
een label per clip. De generatieve VLM's (SmolVLM2, InternVideo2.5,
InternVideo3) krijgen een JSON-prompt en kunnen daarnaast begin- en eindtijden
voorspellen. `action_ground_truth` voegt classificatie-accuracy en temporele
metrics toe voor beide groepen.

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

Temporele voorspellingen zijn beperkt tot het clipvenster. Een
actie die over meerdere clips loopt wordt niet automatisch samengevoegd; de
metrics zijn daarom een indicatie en geen vervanging van een gespecialiseerde
temporal-action-localizer.

## 5. Reproduceerbaarheid

- Gebruik dezelfde video, `frame_stride`, clipinstellingen en prompts.
- InternVideo2.5 draait in de gedeelde transformers 4.47-runtime
  (`third_party/vitra_runtime`) omdat de remote code onder transformers 5
  degenererende output geeft; de benchmark start die run automatisch in een
  subprocess.
- De eerste inferentie kan trager zijn door modeldownload/cache; meet bij
  voorkeur een tweede run.
- `environment` in de summary legt hardware en software vast.
- Vergelijk geen confidence tussen modelfamilies: elke familie kalibreert
  anders. Niet elk model geeft een gekalibreerde confidence.
- Draai `run_benchmark_suite.bat` voor unittests + benchmarkrapport in één
  keer.
