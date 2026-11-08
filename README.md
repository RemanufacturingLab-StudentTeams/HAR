# Project DUPLO

Proof of concept voor het automatisch analyseren van DUPLO-disassemblyvideo's. De scope van het project is **tool recognition** (gereedschappen herkennen en volgen) en **human action recognition** (menselijke handelingen herkennen). Het finetunen van YOLO tot een generiek disassemblymodel is te moeilijk en te tijdrovend gebleken en valt buiten dit project; in plaats daarvan gebruiken we pretrained modellen voor tooldetectie en richten we ons eigen modelleerwerk op de actieherkenning. Zie [Documentation/Opdracht.md](Documentation/Opdracht.md) en [Documentation/PLAN.md](Documentation/PLAN.md) voor het volledige kader.

## Projectstructuur

- `configs/` – instellingen per experiment (video, model, drempels, device)
- `data/` – invoervideo's (niet in git)
- `Documentation/` – [Opdracht.md](Documentation/Opdracht.md) (opdrachtomschrijving), [PLAN.md](Documentation/PLAN.md) (technisch plan), [Modelvergelijking.md](Documentation/Modelvergelijking.md) (testrapport van de modelvergelijking) en [MediaPipe_annotering.md](Documentation/MediaPipe_annotering.md) (handleiding voor handmatige actieannotatie)
- `models/` – gedownloade modelgewichten (`yolo26x.pt`, `yolo26x-seg.pt`, `yolov8x-worldv2.pt`, `hand_landmarker.task`, `sam3.1_multiplex.pt` en optioneel `sam3.pt`; niet in git, worden waar nodig automatisch gedownload)
- `third_party/VITRA/` – optionele clone van de officiële microsoft/VITRA-repo (niet in git; zie de VITRA-sectie)
- `outputs/` – resultaten per video, per stap een eigen submap (`yolo/`, `handpose/`, `tracking/`, `yolo_world_tracking/`, `grounding_dino_tracking/`, `sam3_tracking/`, `sam31_tracking/`, `rfdetr/`, `rfdetr_seg/`, `grounding_dino/`, `yolo_world/`, `yolo_seg/`, `sam3/`, `sam31/`, `vitra/` en `comparison/`) met CSV's, voorbeeldframes en debug-video; de trackingvarianten verschijnen pas na een run
- `src/duplo/` – gedeelde componenten: CLI, config, GPU-setup (`gpu.py`, incl. `to_cuda_device`), video-inlezen (`video.py`), output (`output.py`), detectie (`detection.py` met `Detector`-protocol + `create_detector`), ByteTrack-tracking (`tracking.py`), benchmark-samenvatting (`benchmark.py`), CSV-schema's (`schema.py`), annotatie (`annotate.py`), VITRA-adapter (`vitra.py`, `vitra_state.py` voor HaWoR/MoGe/MANO) en pipeline-orchestratie (`pipeline.py` met `setup_runtime/run_detection_loop`)
- `src/run_yolo_video.py` – YOLO-pipeline (detectie, in dit project gericht op gereedschappen)
- `src/run_handpose.py` – MediaPipe-pipeline (handlandmarks, met `handpose_summary.json` voor timing)
- `src/run_vitra_test.py` – VITRA-pipeline (zero-shot handactie-voorspelling, met optionele HaWoR + MoGe + MANO-statepass)
- `src/run_tracking.py` – YOLO + ByteTrack-pipeline (objecten volgen over frames, met `tracking_summary.json`)
- `src/run_yolo_world_tracking.py` / `src/run_grounding_dino_tracking.py` / `src/run_sam3_tracking.py` / `src/run_sam31_tracking.py` – open-vocabulary detector + ByteTrack (zelfde tracking-CSV als `run_tracking.py`)
- `src/run_rfdetr_video.py` – RF-DETR-detectie (COCO-gewichten)
- `src/run_rfdetr_seg_video.py` – RF-DETR-segmentatie (maskers)
- `src/run_grounding_dino_video.py` – Grounding DINO (open-vocabulary, tekstprompt met toolklassen)
- `src/run_yolo_world_video.py` – YOLO-World (open-vocabulary, tekstprompt met toolklassen)
- `src/run_yolo_seg_video.py` – YOLO26-segmentatie (maskers)
- `src/run_sam3_video.py` – SAM3-segmentatie (open-vocabulary, vereist Hugging Face-toegang)
- `src/run_sam31_video.py` – SAM 3.1-segmentatie via Ultralytics (open-vocabulary, `sam3.1_multiplex.pt`, vereist Hugging Face-toegang)
- `src/compare_models.py` – vergelijkt de modeltesten op tooldetectie en snelheid (detectie-score + aparte tracking-tabel `tracking_comparison.csv`)
- `run_common.bat` – gedeelde venv + pip fast-check (hash van `requirements.txt`; `pip install` wordt overgeslagen bij ongewijzigde hash; forceer met `--upgrade`)
- `run_yolo_test.bat` / `run_handpose_test.bat` / `run_tracking_test.bat` / `run_rfdetr_test.bat` / `run_rfdetr_seg_test.bat` / `run_grounding_dino_test.bat` / `run_yolo_world_test.bat` / `run_yolo_seg_test.bat` / `run_sam3_test.bat` / `run_sam31_test.bat` / `run_sam31_tracking_test.bat` / `run_vitra_test.bat` / `run_compare_models.bat` – dunne startscripts voor Windows (roepen `run_common.bat` aan)

SAM 3.1 gebruikt in `configs/baseline.yaml` een eigen `sam31_tracking`-blok:
de lage `detection_conf`-drempel levert kandidaten voor ByteTrack, terwijl
`track_high_thresh` en `new_track_thresh` voorkomen dat zwakke hypotheses nieuwe
tracks starten. `track_buffer` en `match_thresh` zijn bewust alleen voor SAM 3.1
ingesteld en wijzigen de andere trackingvarianten niet.

## Eerste YOLO-test

Dubbelklik op `run_yolo_test.bat` vanuit de projectmap. Via `run_common.bat` wordt `.venv` aangemaakt als die nog niet bestaat, worden packages uit `requirements.txt` alleen (opnieuw) geïnstalleerd als de hash is gewijzigd (met `run_yolo_test.bat --upgrade` sla je de hash-check over; elke `--upgrade*`-vorm werkt en wordt daarna door `duplo/cli.py` genegeerd), worden de imports gecontroleerd en wordt daarna de YOLO-test uitgevoerd op `data/Disassembly1.mp4`.

De eerste run downloadt automatisch het pretrained model `models/yolo26x.pt` als dat nog ontbreekt. De resultaten komen in `outputs/Disassembly1/yolo/`, met detections in `yolo_detections.csv`, voorbeeldframes als JPG-bestanden en een `yolo_debug.mp4` met de getekende bboxes en labels (uit te zetten met `save_yolo_video: false` in de config). Daarnaast schrijft de stap `yolo_summary.json` met de rekentijd per frame, zodat YOLO in `compare_models.py` als baseline meekan.

## Modelvergelijking: toolrecognition

Voor de toolrecognition zijn naast YOLO zeven alternatieven toegevoegd met dezelfde opzet (video in, CSV + voorbeeldframes + debug-video uit). Elk model schrijft in een eigen submap van `outputs/Disassembly1/` en levert naast de CSV ook een `<model>_summary.json` met frames, detections en rekentijd:

| Startscript | Modelstap | Soort | Submap |
| --- | --- | --- | --- |
| `run_yolo_test.bat` | YOLO26x (baseline) | gesloten vocabulaire (80 COCO-klassen) | `yolo/` |
| `run_rfdetr_test.bat` | RF-DETR Large | gesloten vocabulaire (80 COCO-klassen) | `rfdetr/` |
| `run_grounding_dino_test.bat` | Grounding DINO base | open-vocabulary (tekstprompt) | `grounding_dino/` |
| `run_yolo_world_test.bat` | YOLO-World v2 X | open-vocabulary (tekstprompt) | `yolo_world/` |
| `run_yolo_seg_test.bat` | YOLO26x-seg | segmentatie | `yolo_seg/` |
| `run_rfdetr_seg_test.bat` | RF-DETR Seg 2XLarge | segmentatie | `rfdetr_seg/` |
| `run_sam3_test.bat` | SAM3 | segmentatie, open-vocabulary | `sam3/` |
| `run_sam31_test.bat` | SAM 3.1 (Object Multiplex-checkpoint) | segmentatie, open-vocabulary | `sam31/` |

De doelgereedschappen staan als `tool_classes` in `configs/baseline.yaml` (schroevendraaier, steeksleutel, tang, snijder, hamer). De open-vocabulary modellen krijgen die lijst als tekstprompt; de andere modellen gebruiken hun eigen COCO-klassen en kunnen dus alleen gereedschappen vinden die in COCO zitten. De segmentatiestappen loggen ook `mask_area` (maskeroppervlak in pixels). VITRA is geen detectiemodel en doet daarom niet mee in deze detectie-score; die stap staat apart beschreven in [VITRA: zero-shot handactie-voorspelling](#vitra-zero-shot-handactie-voorspelling).

Alle detectiestappen gebruiken hetzelfde schema (`video`, `frame`, `timestamp`, `klasse`, `confidence`, `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`; centraal vastgelegd in `src/duplo/schema.py`), bij segmentatie aangevuld met `mask_area` als pixel-count op frame-resolutie (YOLO-seg via gerasteriseerd polygoon, RF-DETR-seg/SAM via geresized masker — alleen onderling vergelijkbaar binnen segmentatie). Tracking gebruikt `tracking_history.csv` (`frame_number`, `track_id`, `class_name`, `bbox_x1/y1/x2/y2`, `center`, `velocity`, `track_age`, `frames_since_detection`, `status`; zie `src/duplo/tracking.py:CSV_COLUMNS`), handpose `hand_landmarks.csv` (`video`, `frame`, `hand`, `landmark`, `x/y/z`). `compare_models.py` normaliseert `frame/frame_number` en `klasse/class_name`, zodat oude en nieuwe CSV's beide lezen.

**SAM3:** de gewichten van `facebook/sam3` staan achter een Hugging Face-toegangsaanvraag: vraag toegang aan op https://huggingface.co/facebook/sam3 en log daarna in met `hf auth login`. Op Windows is daarnaast `triton-windows` nodig (sam3 importeert `triton`; het officiële pakket heeft geen Windows-wheels) — beide packages staan in `requirements.txt`. Het pip-pakket mist het BPE-vocabulaire voor de tekstencoder; [run_sam3_video.py](src/run_sam3_video.py) haalt dat eenmalig uit de sam3-repo en zet het in `models/`.

**SAM 3.1:** de gewichten van `facebook/sam3.1` (`sam3.1_multiplex.pt`, Object Multiplex-checkpoint) staan achter een Hugging Face-toegangsaanvraag: vraag toegang aan op https://huggingface.co/facebook/sam3.1 en log daarna in met `huggingface-cli login` (of `hf auth login`). Daarna downloadt Ultralytics het checkpoint automatisch bij `run_sam31_test.bat`, of zet een lokaal bestand via `sam31_model` in `configs/baseline.yaml` (standaard `models/sam3.1_multiplex.pt`, niet in git — handmatig downloaden mag ook). De test gebruikt Ultralytics `SAM3SemanticPredictor` met de `tool_classes` als tekstprompt (per-frame beeldinferentie met maskers); Object Multiplex-videotracking wordt door Ultralytics nog niet ondersteund. Voor het native videomodel (gezamenlijk multi-object tracken met gedeeld geheugen, ~7x sneller bij veel objecten) is de originele repo https://github.com/facebookresearch/sam3 nodig (Python 3.12+, CUDA 12.6+, `sam3.1_video_predictor_example.ipynb`); zie ook `RELEASE_SAM3p1.md` daar. Bij `TypeError: 'SimpleTokenizer' object is not callable` is het verkeerde `clip`-pakket geïnstalleerd: draai dan `pip uninstall clip -y` en `pip install git+https://github.com/ultralytics/CLIP.git`.

De SAM 3.1-checkpoints (Object Multiplex, `sam3.1_multiplex.pt`) verbeteren volgens Meta vooral het *videomodel* (~7x sneller bij veel objecten); Ultralytics laadt hetzelfde checkpoint ook voor beeldinferentie, dus `run_sam31_test.bat` vergelijkt de actuele SAM 3.1-beelddetector met de SAM3-test (`sam3.pt`). De repositoryversie van de SAM3-code pint `numpy<2` en vraagt `pkg_resources` (`setuptools<81`); `requirements.txt` gebruikt daarom bewust de PyPI-release (`sam3==0.1.4`, werkt met numpy 2.x op Python 3.13) en niet de git-versie — gebruik voor die git-versie een aparte venv.

### Vergelijken

Draai eerst de testen die je wilt vergelijken en daarna:

```bat
run_compare_models.bat
```

`src/compare_models.py` leest alle detectiestappen in `outputs/Disassembly1/` (`*_detections.csv` + summary) en schrijft `outputs/Disassembly1/comparison/model_comparison.csv` en `model_comparison.md` met daarin per model: gevonden toolklassen, tool-detecties, aandeel frames met een tool, gemiddelde confidence, detecties per frame, rekentijd en doorvoer. Tracking (`*_tracking_history.csv`) krijgt een aparte tabel `tracking_comparison.csv` (tracks, rijen/frame, `confirmed/tentative/occluded`, timing) en weegt niet mee in de detectie-score. De rangschikking is een expliciet weegbare heuristic:

```text
score = 100 x (coverage x dekking + presence x tool-frames + speed x min(1, doorvoer/target_fps))
```

De gewichten (`comparison_weights`) en `comparison_target_fps` (de doorvoer die nodig is om real-time te draaien: bronfps / frame_stride) staan in `configs/baseline.yaml`.

Het volledige testrapport met de resultaten van alle modellen staat in [Documentation/Modelvergelijking.md](Documentation/Modelvergelijking.md).

Let op de kanttekeningen in het rapport: er is geen geannoteerde ground truth, dus accuracy (precision/recall) kan niet worden gemeten; dekking en tool-frames zijn proxies en bevoordelen open-vocabulary modellen, die hun promptklassen per definitie kunnen terugvinden. Confidence is niet tussen modelfamilies te vergelijken. Gebruik de debug-video's en voorbeeldframes per stap voor het kwalitatieve oordeel.

## Eerste MediaPipe-test

Dubbelklik op `run_handpose_test.bat` om MediaPipe handpose op dezelfde video uit te voeren. De handlandmarks komen in `outputs/Disassembly1/handpose/hand_landmarks.csv` en geannoteerde voorbeeldframes worden als JPG-bestanden opgeslagen. Daarnaast schrijft de stap een `handpose_debug.mp4` met de getekende handlandmarks (uit te zetten met `save_handpose_video: false` in de config) en een `handpose_summary.json` met timing (geen detectie-score).

## VITRA: zero-shot handactie-voorspelling

[VITRA](https://github.com/microsoft/VITRA) (Microsoft Research) is een 3B Vision-Language-Action-model dat vanuit één egocentrisch frame plus een tekstinstructie een chunk toekomstige MANO-handacties voorspelt: translatie (3), globale rotatie (3) en 45 gewrichtshoeken per hand. In dit project gebruiken we het als aanvulling op MediaPipe: `run_vitra_test.bat` voorspelt per gesampled frame de rechterhand (of beide handen) en logt de actiechunk. VITRA is geen classifier en geeft dus geen actielabel; de CSV bevat de continue voorspelling waarmee je later acties kunt analyseren of vergelijken.

### Eenmalige setup

VITRA's eigen `pyproject.toml` pint `torch==2.3.0`, `transformers==4.47.1` en `numpy<2`. Dat is onverenigbaar met de rest van dit project (torch 2.14/CUDA 13, transformers 5.x voor SAM3). Installeer VITRA daarom **niet** met `pip install -e`; clone de repo en laat de project-venv hem via `sys.path` laden:

```bat
git clone https://github.com/microsoft/VITRA.git third_party\VITRA
git -C third_party\VITRA submodule update --init --recursive
run_vitra_test.bat --upgrade
.venv\Scripts\pip install --no-build-isolation --no-deps chumpy@git+https://github.com/mattloper/chumpy
```

`run_vitra_test.bat --upgrade` installeert de runtime-dependencies uit `requirements.txt`, inclusief de HaWoR/MoGe/MANO-packages (`smplx`, `yacs`, `pytorch_lightning`, `omegaconf`, `scikit-image`, `trimesh`, `utils3d`). `chumpy` staat bewust niet in `requirements.txt`, omdat het alleen met `--no-build-isolation` installeert; de losse regel hierboven doet dat eenmalig. De geneste DROID-SLAM/eigen-submodule van HaWoR is niet nodig voor handreconstructie; als die clone faalt, kun je de fout negeren.

**Waarom een aparte transformers-runtime:** VITRA 0.0.1 is getraind met `transformers==4.47.1`, terwijl de hoofd-venv 5.x gebruikt voor SAM3. Een A/B-test met identieke inputs en gewichten laat zien dat de 5.x-PaliGemma 10–100× te grote acties oplevert (cognition-token-norm 96 i.p.v. 7, chunk-pad 9 m i.p.v. 0,2 m). Daarom installeert `run_vitra_test.bat` eenmalig de pins uit `requirements-vitra.txt` in `third_party/vitra_runtime` en start de VITRA-stap zichzelf opnieuw in een subprocess met die map op de `PYTHONPATH`. De hoofd-venv blijft ongemoeid. Handmatig installeren kan met:

```bat
.venv\Scripts\python.exe -m pip install --no-deps --target third_party\vitra_runtime -r requirements-vitra.txt
```

Daarnaast is Hugging Face-toegang nodig:

1. vraag toegang aan voor `google/paligemma2-3b-mix-224` (VITRA is daarop gefinetuned);
2. log in met `hf auth login` (of `huggingface-cli login`);
3. het VITRA-checkpoint (`VITRA-VLA/VITRA-VLA-3B`, ~12 GB) wordt bij de eerste run automatisch naar de HF-cache gedownload.

Voor de HaWoR/MANO-reconstructie (pass 1) zijn daarnaast deze gewichten nodig:

```bat
curl -L https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -o third_party\VITRA\weights\hawor\checkpoints\hawor.ckpt
curl -L https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -o third_party\VITRA\weights\hawor\external\detector.pt
```

en **`MANO_RIGHT.pkl`**. Dat bestand valt onder de MANO-licentie van het Max Planck Instituut en kan niet automatisch worden gedownload: registreer je op https://mano.is.tue.mpg.de/, download het MANO-modelpakket (`mano_v1_2.zip`) en kopieer **alleen** `models/MANO_RIGHT.pkl` naar `third_party\VITRA\weights\mano\` (het bestand `mano_mean_params.npz` staat daar al; niet overschrijven). `MANO_LEFT.pkl` en de `MANO_UV_*`-varianten zijn niet nodig. MANO is gratis voor niet-commercieel onderzoek; voor commercieel gebruik is een aparte licentie nodig.

MANO is van oorsprong een **rechtermodel**: `MANO_LEFT.pkl` is de gespiegelde kopie met dezelfde topologie en dezelfde 51-dimensionale pose-ruimte. HaWoR en VITRA spiegelen de linkerhand daarom expliciet in plaats van een tweede model te laden: `data/tools/hand_recon_core.py` flipt de x-as van vertices en wrist-joint, en `data/tools/utils_hawor.py` zet `do_flip=True` en spiegelt de globale rotatie. Daarmee kun je met alleen `MANO_RIGHT.pkl` toch **beide handen** voorspellen: zet `vitra_hands: ["left", "right"]` (VITRA's actiekop heeft per hand een eigen 51-dimensionaal masker). Alleen voor HaWoR's eigen linkerhand-renderpad (`_DATA/data_left/mano_left/`) zou je `MANO_LEFT.pkl` nodig hebben; dat pad gebruikt deze pipeline niet.

VITRA-inferentie vereist een NVIDIA-GPU met minimaal ~16 GB VRAM; `--device cpu` wordt met een duidelijke melding geweigerd omdat de modelcode inputs hard op `cuda` zet. Op een RTX 5080 (16 GB) laadt het 3B-model in fp32 en haalt de stap met de standaardinstellingen ongeveer 6 fps na de eenmalige download. De HaWoR/MoGe/MANO-modellen en VITRA passen niet tegelijk in 16 GB; daarom draait de reconstructie in een aparte eerste pass en wordt die daarna volledig van de GPU verwijderd.

### Draaien en resultaten

```bat
run_vitra_test.bat
run_vitra_test.bat --device gpu
.venv\Scripts\python.exe src\run_vitra_test.py --config configs\baseline.yaml
```

Met `vitra_reconstruct_hands: true` (de standaard in `configs/baseline.yaml`) draait eerst **pass 1**: HaWoR + MoGe + MANO maken per gesampled frame een echte handstate en schrijven die naar `vitra_hand_states.npz`. Daarna draait **pass 2**: VITRA laadt het 3B-model en gebruikt de states per frame. Met `vitra_states_path` sla je pass 1 bij een volgende run over.

Op de RTX 5080 duurde pass 1 (HaWoR/MoGe/MANO, 322 frames) ongeveer 245 seconden en de VITRA-pass ongeveer 60 seconden (5,3 predicties/s). Een tweede run slaat pass 1 over via `vitra_reuse_states`.

De resultaten komen in `outputs/Disassembly1/vitra/`:

- `vitra_hand_states.npz` – per-frame MANO-states (`frames`, `states` [N,122], `fov_x`, `valid` per hand) uit pass 1.
- `vitra_actions.csv` – per gesampled frame, per sample, per hand en per voorspelde stap: `video`, `frame`, `timestamp`, `sample`, `hand`, `step`, `transl_x/y/z` (relatieve translatie in meter), `rot_x/y/z` (relatieve rotatie in rad), `transl_norm` (stapgrootte in meter) en `pose_mean` (gemiddelde absolute gewrichtshoek in rad). Het schema staat centraal in `src/duplo/schema.py:VITRA_COLUMNS`. Filter `step == 0` voor het eerstvolgende frame; de overige stappen beschrijven de voorspelde toekomst.
- `vitra_debug.mp4` – debug-video met de invoerinstructie, frametijd, per hand de voorspelde verplaatsing met richting en de geprojecteerde baan (uit te zetten met `save_vitra_video: false`).
- `vitra_frame_*.jpg` – geannoteerde voorbeeldframes volgens `screenshot_every`.
- `vitra_summary.json` – timing, aantal actierijen, instructie, handen, chunkgrootte, sample-instellingen, statebron en FOV.

### Configuratie

De VITRA-opties staan bij elkaar in `configs/baseline.yaml`:

| Sleutel | Betekenis |
| --- | --- |
| `vitra_repo_path` | Pad naar de clone (standaard `third_party/VITRA`). |
| `vitra_config_path` | HF-repo-id of lokaal `config.json` van het checkpoint. |
| `vitra_model_path` / `vitra_statistics_path` | Optioneel lokaal `.pt`-bestand en `dataset_statistics.json`; leeg betekent de HF-repo uit de config. |
| `vitra_instruction` | Engelse taakinstructie, bijv. `"Left hand: None. Right hand: Pick up the tool..."`. |
| `vitra_hands` | `["right"]`, `["left"]` of beide. Beide handen verdubbelen de rekentijd en hebben geen extra MANO-bestand nodig (de linkerhand wordt gespiegeld). |
| `vitra_frame_stride` | Aantal videoframes per VITRA-predictie (standaard 30 ≈ 1 predictie/seconde). |
| `vitra_sample_times` | Aantal actiesamples per frame; de debug-overlay toont sample 0. |
| `vitra_ddim_steps` / `vitra_cfg_scale` | Diffusiestappen en classifier-free-guidance in de actiekop. |
| `vitra_fov_deg` | `[fov_x, fov_y]` in graden; `null` gebruikt `fov_x` uit het statebestand/per-frame state en leidt `fov_y` af uit de beeldverhouding, anders 60/60. |
| `vitra_state_path` | Statisch `.npy`-statebestand (VITRA-formaat of 122-vector); heeft voorrang op reconstructie. |
| `vitra_reconstruct_hands` | `true` = pass 1 met HaWoR + MoGe + MANO; vereist `MANO_RIGHT.pkl`. |
| `vitra_reuse_states` | Hergebruik een bestaand `vitra_hand_states.npz` in de uitvoermap (`false` = altijd een nieuwe pass 1). |
| `vitra_runtime_path` | Map met de gepinde `transformers==4.47.1` (standaard `third_party/vitra_runtime`). |
| `vitra_states_path` | Bestaand `vitra_hand_states.npz` hergebruiken; heeft voorrang op `vitra_state_path` en reconstructie. |
| `vitra_hawor_model_path` / `vitra_hawor_detector_path` | HaWoR-checkpoint en handdetector. |
| `vitra_moge_model_name` | MoGe-model voor de FOV-schatting (HF-repo of lokaal pad). |
| `vitra_mano_path` | Map met `MANO_RIGHT.pkl` en `mano_mean_params.npz`. |
| `vitra_image_size` | Korte zijde die VITRA verwacht (224). |
| `vitra_debug_repeat_frames` | `true` = elk voorspelframe herhalen tot de bronfps, zodat de debug-video normaal afspeelt (anders 1 fps). |
| `save_vitra_video` | Debug-video aan/uit. |

De statebron wordt in deze volgorde gekozen:

1. `vitra_states_path` – bestaande per-frame states uit een eerdere pass 1;
2. `vitra_state_path` – één statische state voor de hele video (bijv. `third_party/VITRA/examples/0002.npy`);
3. `vitra_reconstruct_hands: true` – pass 1 maakt per frame een state met HaWoR + MoGe + MANO;
4. anders de nul-state uit VITRA's minimale voorbeeld.

De nul-state werkt, maar de voorspelling is dan grof: de huidige handpositie en -pose ontbreken. Omdat er dan geen echte state is, geeft de stap die hand als **onbekend** aan VITRA door (state-masker uit) in plaats van als een geldige neutrale pose. De VITRA-repo bevat `examples/0001.npy` en `examples/0002.npy` met echte MANO-states; let op dat `0002.npy` alleen een rechterhand bevat, dus bij `vitra_hands: ["left", "right"]` krijgt de linkerhand daar het "onbekend"-masker (met een waarschuwing). Voor je eigen video is de reconstructie-route (pass 1) de bedoelde manier. Zo'n `.npy`/`.npz` kan pickles bevatten; gebruik alleen bestanden die je zelf of de VITRA-repo hebt gemaakt. Het MANO-model zelf is niet vrij voor commercieel gebruik; zie de licentie op https://mano.is.tue.mpg.de/.

Net als bij handpose wordt VITRA niet meegenomen in `compare_models.py`: de stap levert geen tool-detecties maar continue handacties. Gebruik de CSV en de debug-video voor de kwalitatieve beoordeling.

Wat VITRA wel en niet doet, zodat de output niet verkeerd gelezen wordt:

- **Geen actielabel of tekstbeschrijving.** VITRA voorspelt een *beweging*, geen klasse als "unscrewing". De eerste regel in de debug-video is de invoerprompt (`vitra_instruction`) en blijft dus bewust gelijk; alleen de regels per hand veranderen.
- **Eén voorspelling per gesampled frame** (`vitra_frame_stride: 30` ≈ 1 per seconde). Elke voorspelling is een chunk van 16 stappen vooruit, niet één frame. De stap is dus geen 30 fps-model: de video wordt standaard herhaald tot de bronfps (`vitra_debug_repeat_frames: true`) zodat hij normaal afspeelt, maar de inhoud ververst één keer per seconde. Voor fijnere temporal resolutie verlaag je `vitra_frame_stride` (kost evenredig rekentijd) of verhoog je `vitra_sample_times`.
- De regel per hand toont de **som van de voorspelde translatie** over de chunk (meters), de richting in cameracoördinaten (x rechts, y omlaag, z van je af), de eerste stap in millimeters en de gemiddelde gewrichtshoek in rad. De groene/oranje lijn is dezelfde translatie, geprojecteerd op het beeld.

De VITRA-rijbouw, state-parsing, padding en checkpoint-remapping zijn los te testen zonder model of GPU:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## ByteTrack-tracking

Dubbelklik op `run_tracking_test.bat` om YOLO + ByteTrack op dezelfde video uit te voeren. Voor SAM 3.1 gebruik je `run_sam31_tracking_test.bat`; de overige open-vocabulary varianten zijn direct via Python te starten (`src/run_yolo_world_tracking.py`, `src/run_grounding_dino_tracking.py` en `src/run_sam3_tracking.py`). Elke detectie krijgt een unieke track-ID die tussen frames stabiel blijft, ook wanneer een object korte tijd aan het zicht onttrokken is (occlusie). De resultaten komen in `outputs/Disassembly1/tracking/` (resp. `yolo_world_tracking/`, `grounding_dino_tracking/`, `sam3_tracking/`, `sam31_tracking/`):

- `tracking_history.csv` – per frame één rij per track met `track_id`, `class_id`, `class_name`, `confidence`, `bbox_x1`..`bbox_y2`, `center_x`/`center_y` (bbox en center in pixels), `velocity_x`/`velocity_y` (verplaatsing in pixels per videoframe), `track_age` (videoframes sinds het ontstaan van de track), `frames_since_detection` (videoframes sinds de laatste detectie) en `status`. `track_age` en `frames_since_detection` gebruiken dezelfde bronframe-index als `frame_number`. Rijen met status `occluded` hebben geen bbox of confidence: het object was tijdelijk niet zichtbaar, maar de track blijft bestaan om de occlusie te overbruggen. Rijen met status `tentative` zijn tracks die net gestart zijn en nog bevestigd moeten worden.
- `tracking_debug.mp4` – debug-video met YOLO-bboxes, klassennamen, confidence, track-ID's en trajectories. Occluded tracks worden grijs weergegeven op hun voorspelde positie. De video wordt als H.264 (yuv420p) geschreven, zodat hij in VS Code, browsers en deelplatforms afspeelt; OpenCV's standaard mp4v-formaat doet dat niet.

De trackerinstellingen staan in `configs/baseline.yaml`: `track_high_thresh` (drempel voor de eerste associatiestap), `track_low_thresh` (ondergrens waarmee YOLO-detecties worden meegenomen; lage scores worden in de tweede associatiestap gebruikt om occlusies te overbruggen), `new_track_thresh` (minimale score om een nieuwe track te starten), `track_buffer` (hoeveel frames een track zonder detectie blijft bestaan), `match_thresh` en `fuse_score` (associatie) en `track_trajectory_length` (lengte van de getekende trajectory). Associatie gebeurt met beweging (Kalman-filter) en IoU; ByteTrack heeft geen appearance-embedder en draait op de CPU.

## Draaien op de GPU

De scripts accepteren een `--device` argument met de waarden `cpu` of `gpu`:

```bat
run_yolo_test.bat --device gpu
run_handpose_test.bat --device gpu
run_tracking_test.bat --device gpu
run_rfdetr_test.bat --device gpu
run_rfdetr_seg_test.bat --device gpu
run_grounding_dino_test.bat --device gpu
run_yolo_world_test.bat --device gpu
run_yolo_seg_test.bat --device gpu
run_sam3_test.bat --device gpu
run_sam31_test.bat --device gpu
run_vitra_test.bat --device gpu
```

Of direct via Python:

```bat
.venv\Scripts\python.exe src\run_yolo_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_handpose.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_tracking.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_rfdetr_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_rfdetr_seg_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_grounding_dino_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_yolo_world_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_yolo_seg_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_sam3_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_sam31_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_vitra_test.py --config configs\baseline.yaml --device gpu
```

Zonder argument gebruiken de scripts de waarde van `device` in `configs\baseline.yaml` (daar staat `gpu`; met `--device cpu` dwing je een CPU-run af).

**YOLO:** bij `--device gpu` controleert het script of torch CUDA ondersteunt. Standaard volgt een foutmelding met het installatie-commando (`pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130`); alleen met `auto_install_torch: true` in de config wordt de CUDA-build (~2 GB) automatisch geïnstalleerd. De ByteTrack-tracker draait op de CPU en heeft geen CUDA nodig; alleen de detectie gebruikt de GPU. `yolo_half` (FP16 via `quantize=16`), `yolo_fuse` en `yolo_backend: tensorrt` (opt-in `.engine`-export met fallback op `.pt`) veranderen alleen kernels, geen drempels/gewichten. `video_encoder: auto` gebruikt NVENC indien aanwezig, anders x264, anders mp4v.

**MediaPipe:** in de huidige Windows-build van mediapipe is de GPU-delegate niet beschikbaar. `--device gpu` wordt geaccepteerd, maar het script meldt dit en valt terug op CPU.

## Acties die we gaan herkennen

- schroeven losdraaien
- schroeven vastdraaien
- gereedschap pakken
- gereedschap neerleggen
- opwippen van een onderdeel
- roteren van onderdeel

## Voortgang

Tijdens het draaien wordt de voortgang direct getoond: in een console een balk die op één regel bijwerkt, en bij vastgelegde uitvoer (logbestand of IDE-paneel) voortgangsregels bij elke 10% of elke 5 seconden.
