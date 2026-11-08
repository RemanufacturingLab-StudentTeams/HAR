# Testrapport – modelvergelijking voor toolrecognition

*Uitgevoerd op 14 oktober 2026 op `data/Disassembly1.mp4` met de pipelines in `src/` en de instellingen uit `configs/baseline.yaml`. Bijgewerkt op 17 oktober 2026: Grounding DINO opnieuw gedraaid na de labelnormalisatie, YOLO-tracking opnieuw gedraaid na de frame-index-fix en SAM 3.1 toegevoegd. Het bijbehorende cijfermateriaal staat in `outputs/Disassembly1/comparison/` (niet in git; opnieuw te genereren met `run_compare_models.bat`).*

> **Stand van zaken:** de SAM3-promptbug is gefixt en SAM3 is opnieuw gedraaid. Grounding DINO is opnieuw gedraaid met de aangescherpte labelnormalisatie, zodat `##drive`/`screwr`-fragmenten niet meer als aparte klassen in de CSV staan. SAM 3.1 is toegevoegd. De vier trackingvarianten (YOLO-World, Grounding DINO, SAM3 en SAM 3.1) zijn nog niet volledig op de video gedraaid; alleen YOLO-tracking staat in de trackingtabel.

## 1. Doel

Fase 7 van het technisch plan (zie [PLAN.md](PLAN.md)) draait om **tool recognition**: kan een pretrained model de gereedschappen in de DUPLO-disassemblyvideo's herkennen zonder eigen training? De baseline gebruikt YOLO26x met COCO-gewichten; dit onderzoek zet daar zeven alternatieven naast en beantwoordt de vraag welk model voor deze taak het beste past:

* **RF-DETR** (detectie) en **RF-DETR Seg** (segmentatie) – gesloten vocabulaire, COCO;
* **Grounding DINO** en **YOLO-World** – open-vocabulary: de gereedschapsnamen gaan als tekstprompt mee;
* **YOLO26x-seg** (segmentatie) – gesloten vocabulaire, COCO;
* **SAM3** en **SAM 3.1** – open-vocabulary segmentatie op basis van een tekstprompt.

## 2. Testopzet

| Onderdeel | Waarde |
| --- | --- |
| Video | `data/Disassembly1.mp4` (top-down opname van het demonteren van een telefoon) |
| Videoformaat | 9639 frames · 30 fps · 1280×720 · 321 s |
| `frame_stride` | 2 → **4820 verwerkte frames** per model |
| `conf_threshold` | 0,25 voor alle modellen |
| Grounding DINO `text_threshold` | 0,25 |
| Doelgereedschappen (`tool_classes`) | `screwdriver`, `wrench`, `pliers`, `cutter`, `hammer` |
| Voorbeeldframes | elke 150 verwerkte frames (`screenshot_every`) |
| Hardware | NVIDIA GeForce RTX 5080 (16 GB), Windows |
| Software | Python 3.13.15 · torch 2.14.1+cu130 · ultralytics 8.4.177 · rfdetr 1.11.2 · transformers 5.19.0 · supervision 0.30.9 · sam3 0.1.4 · triton-windows 3.8.0.post29 |

Elke modelstap doorloopt dezelfde pipeline: video inlezen (`duplo.video.VideoFrames`), per frame inferentie, resultaten wegschrijven en de rekentijd meten (`duplo.benchmark`). Per stap verschijnt een eigen submap in `outputs/Disassembly1/` met:

* `<model>_detections.csv` – één rij per detectie: `video`, `frame`, `timestamp`, `klasse`, `confidence`, `bbox_x/y/w/h` en bij segmentatie `mask_area` (maskeroppervlak in pixels);
* `<model>_debug.mp4` – H.264-video met bboxes, labels (en maskers);
* `<model>_frame_*.jpg` – voorbeeldframes voor de kwalitatieve controle;
* `<model>_summary.json` – frames, detecties, klassen en rekentijd.

## 3. Geteste modellen

| Stap | Gewichten | Soort | Omvang | Licentie |
| --- | --- | --- | --- | --- |
| `yolo` | `models/yolo26x.pt` (COCO) | detectie, gesloten vocabulaire | 59,0 M | AGPL-3.0 (Ultralytics) |
| `rfdetr` | `RFDETRLarge` (COCO) | detectie, gesloten vocabulaire | 33,9 M · 704×704 | Apache-2.0 (XL/2XL vallen onder PML) |
| `grounding_dino` | `IDEA-Research/grounding-dino-base` | detectie, open-vocabulary | 232,8 M | Apache-2.0 |
| `yolo_world` | `models/yolov8x-worldv2.pt` | detectie, open-vocabulary (CLIP-tekstprompt) | 72,9 M | AGPL-3.0 (Ultralytics) |
| `yolo_seg` | `models/yolo26x-seg.pt` (COCO) | segmentatie, gesloten vocabulaire | 70,7 M | AGPL-3.0 (Ultralytics) |
| `rfdetr_seg` | `RFDETRSeg2XLarge` (COCO) | segmentatie, gesloten vocabulaire | 38,6 M · 768×768 | Apache-2.0 |
| `sam3` | `facebook/sam3` (`sam3.pt`) | segmentatie, open-vocabulary | 859,9 M | Meta-licentie (gewichten achter HF-goedkeuring) |
| `sam31` | `models/sam3.1_multiplex.pt` | segmentatie, open-vocabulary | 3,5 GB checkpoint | Meta-licentie (gewichten achter HF-goedkeuring) |

De gewichten van RF-DETR, SAM3 en SAM 3.1 worden bij de eerste run automatisch gedownload; YOLO-gewichten komen uit de Ultralytics-releases. SAM3 en SAM 3.1 vragen daarnaast om een goedgekeurde Hugging Face-toegangsaanvraag en `hf auth login` (zie [README](../README.md)).

## 4. Werkwijze en metrieken

Er is **geen geannoteerde ground truth**, dus precision/recall/mAP zijn niet te meten. De vergelijking gebruikt daarom meetbare proxies die samen iets zeggen over bruikbaarheid voor toolrecognition:

| Metriek | Betekenis |
| --- | --- |
| Tool-detecties | aantal detecties waarvan de klasse in `tool_classes` valt |
| Dekking | aantal doelklassen met ≥ 1 detectie (van de 5) |
| Tool-frames | aandeel verwerkte frames met ≥ 1 tool-detectie (temporele aanwezigheid) |
| Gem. conf. | gemiddelde confidence van de tool-detecties (alleen binnen één modelfamilie vergelijkbaar) |
| Det./frame | alle detecties per verwerkt frame (ook niet-tools; maat voor ruis/overkill) |
| Inferentie (ms) | mediaan van de modelaanroep per frame |
| Doorvoer (fps) | verwerkte frames per seconde wandkloktijd, inclusief video-inlezen, tekenen en debug-video |
| Score | heuristische rangschikking, zie formule hieronder |

```text
score = 100 × (0,4 × dekking + 0,3 × tool-frames + 0,3 × min(1, doorvoer/15))
```

De gewichten staan in `configs/baseline.yaml` (`comparison_weights`) en zijn bewust aanpasbaar: 15 fps is de doorvoer die nodig is om met `frame_stride: 2` real-time mee te lopen met de 30fps-video. Dekking en tool-frames wegen samen zwaarder dan snelheid (70/30), omdat een model dat de tools niet vindt ook snel niet nuttig is.

## 5. Resultaten

### 5.1 Score en hoofmetrieken

| Model | Modelnaam | Toolklassen (dekking) | Tool-detecties | Tool-frames | Gem. conf. | Det./frame | Inferentie (ms) | Doorvoer (fps) | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sam31 | SAM 3.1 (Object Multiplex) | **5/5** | 9 070 | **100,0 %** | 0,81 | 1,88 | 99,4 | 9,2 | **88,4** |
| yolo_world | YOLO-World v2 X | 4/5 | 1 955 | 35,1 % | 0,38 | 0,41 | **9,2** | **72,2** | 72,5 |
| grounding_dino | Grounding DINO base | 3/5 | 7 653 | 98,2 % | 0,43 | 1,59 | 123,4 | 7,9 | 69,3 |
| sam3 | SAM3 (`facebook/sam3`) | 3/5 | 8 962 | 99,9 % | **0,82** | 1,86 | 330,6 | 2,9 | 59,8 |
| rfdetr | RF-DETR Large | 0/5 | 0 | 0 % | n.v.t. | 3,11 | 10,1 | 73,1 | 30,0 |
| rfdetr_seg | RF-DETR Seg 2XLarge | 0/5 | 0 | 0 % | n.v.t. | 2,75 | 19,7 | 36,0 | 30,0 |
| yolo | YOLO26x (baseline) | 0/5 | 0 | 0 % | n.v.t. | 2,49 | 12,0 | 65,3 | 30,0 |
| yolo_seg | YOLO26x-seg | 0/5 | 0 | 0 % | n.v.t. | 1,98 | 14,8 | 46,9 | 30,0 |

Totale wandkloktijd per volledige video: RF-DETR 65,9 s, YOLO-World 66,8 s, YOLO 73,8 s, YOLO-seg 102,8 s, RF-DETR-seg 134,1 s, SAM 3.1 522,7 s (8,7 min), Grounding DINO 607,3 s (10,1 min) en SAM3 1637,7 s (27,3 min).

#### Hardware

Getest op: RTX 5080 (16 GB VRAM), Ryzen 9 9950X (16 cores, 32 threads, 4,5 GHz), 64GB DDR5-6000 RAM.

### 5.2 Detecties per gereedschapsklasse

| Gereedschap | grounding_dino | rfdetr | rfdetr_seg | sam3 | sam31 | yolo | yolo_seg | yolo_world |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| screwdriver | 7 613 | 0 | 0 | **8 817** | 8 515 | 0 | 0 | 1 918 |
| pliers | 31 | 0 | 0 | 144 | 523 | 0 | 0 | 29 |
| hammer | 9 | 0 | 0 | 0 | 4 | 0 | 0 | 5 |
| wrench | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 3 |
| cutter | 0 | 0 | 0 | 1 | 25 | 0 | 0 | 0 |

Alleen de vier open-vocabulary modellen vinden gereedschappen; de COCO-modellen scoren 0/5. Dat is geen detectiefout maar een **vocabulairebeperking**: RF-DETR en de YOLO-modellen labelen de schroevendraaier consequent als `knife` of `scissors`. Die klassen komen in de ruwe data veel voor (RF-DETR: 698× `knife`, 167× `scissors`; YOLO-seg: 57× `scissors`), maar tellen niet mee omdat ze niet in `tool_classes` staan.

De segmentatiemodellen loggen ook `mask_area` (maskeroppervlak in pixels). Die getallen zeggen vooral *wat* er gemaskeerd wordt: YOLO-seg (gem. 49 266,5 px) en RF-DETR-seg (35 377,6 px) maskeren vrijwel uitsluitend de persoon/handen en de telefoon, terwijl SAM3 (gem. 3 114,4 px) en SAM 3.1 (3 159,7 px) de gereedschappen zelf maskeren — een klein, precies oppervlak dat past bij een schroevendraaier.

### 5.3 Meest gedetecteerde klassen (top-3)

| Model | Top-3 klassen |
| --- | --- |
| sam31 | screwdriver (8 515), pliers (523), cutter (25) |
| yolo_world | screwdriver (1 918), pliers (29), hammer (5) |
| grounding_dino | screwdriver (7 613), pliers (31), hammer (9) |
| sam3 | screwdriver (8 817), pliers (144), cutter (1) |
| rfdetr | person (8 475), cell phone (4 542), knife (698) |
| rfdetr_seg | person (5 825), cell phone (5 770), knife (696) |
| yolo | person (6 440), cell phone (2 702), book (1 335) |
| yolo_seg | person (7 509), cell phone (1 697), remote (192) |

## 6. Kwalitatieve controle

De cijfers zijn gecontroleerd tegen de opgeslagen voorbeeldframes en debug-video's:

* **Frame 3600** – YOLO-World zet een strak kader om de echte schroevendraaier (`screwdriver 0,28`); RF-DETR zet vrijwel hetzelfde kader om hetzelfde object, maar noemt het `knife` (0,27). Het vocabulaireprobleem is dus zichtbaar in één frame.
* **Frame 6000** – Grounding DINO vindt twee echte schroevendraaiers (0,50 en 0,50); SAM3 segmenteert dezelfde twee objecten met maskers op 0,96 en 0,96. De maskers liggen precies op de gereedschappen.
* **Frame 9150** – SAM3 vindt de schroevendraaier opnieuw (0,97), maar produceert ook twee kleine false positives op de hand (0,34 en 0,42). SAM 3.1 vindt hetzelfde object (0,97) en dezelfde twee twijfelgevallen (0,36 en 0,29).
* De zeldzame klassen zijn ruis: YOLO-Worlds `wrench` (3×) en `hammer` (5×) liggen op de schroevendraaier of hand, DINO's `pliers`/`hammer` (31/9) zijn kortstondig en laag-confident, en SAM3's `pliers` (144, mediaan 0,36) liggen grotendeels op de handen. SAM 3.1's extra klassen (`pliers` 523, `cutter` 25, `hammer` 4, `wrench` 3) moeten nog apart tegen de debug-video worden gecontroleerd. Het betrouwbare signaal is bij alle vier **`screwdriver`**.
* De continue `person`- en `cell phone`-detecties van de COCO-modellen grijpen de handen/arm van de monteur en de telefoon die in beeld ligt; ze zijn voor toolrecognition niet bruikbaar.

## 7. Interpretatie en advies

1. **SAM 3.1** haalt de hoogste heuristische score (88,4): 5/5 klassen, tool-detecties in 100 % van de frames en 9,2 fps. Het checkpoint is met 3,5 GB wel groot en de doorvoer blijft onder de 15 fps die voor real-time nodig is; gebruik het daarom voor offline analyse, keyframes en kwaliteitscontrole.
2. **YOLO-World v2 X** is de beste keuze voor de pipeline zoals die nu draait: het gereedschap dat in deze video echt voorkomt (de schroevendraaier) krijgt meteen de juiste naam, het vindt 4 van de 5 doelklassen, en met 72,2 fps (pipeline) is het ruim snel genoeg om real-time te draaien en om er direct ByteTrack-tracking op te zetten.
3. **SAM3** levert de hoogste kwaliteit per detectie: mediaan-confidence 0,96, maskers die de gereedschappen precies volgen en een tool-detectie in 99,9 % van de frames. Met ± 2,9 fps is het niet real-time, maar het is zeer geschikt voor keyframe-analyse, offline verwerking en als **auto-labeler** om een toolannotatie te maken voor fine-tuning of voor precision/recall-meting.
4. **Grounding DINO** zit er tussenin: 98,2 % tool-frames en redelijke dekking, maar lagere confidence, meer verspreide kaders en 7,9 fps.
5. **RF-DETR en de YOLO26-modellen (detectie en segmentatie) vallen af zonder finetuning**: ze zijn zeer snel (10–20 ms), maar missen de juiste klassen. Wil het project deze modellen gebruiken, dan is finetuning op de toolklassenlijst uit PLAN.md fase 7 nodig — met SAM3 (of Grounding DINO) als labelbron.

## 8. Beperkingen

* **Geen ground truth**: dekking en tool-frames zijn proxies; absolute precision/recall zijn niet bekend. Een geannoteerde steekproef (bijvoorbeeld 200 frames) is nodig om de modellen echt te rangschikken op kwaliteit.
* **Vocabulaire-bias**: open-vocabulary modellen krijgen de doelklassen als prompt en kunnen die daardoor "altijd" terugvinden; gesloten-vocabulaire modellen kunnen per definitie niet winnen op dekking. De vergelijking meet dus eerder haalbaarheid dan accuratesse.
* **Confidence is niet vergelijkbaar tussen families**: 0,38 bij YOLO-World is niet hetzelfde als 0,38 bij Grounding DINO.
* **Kleine aantallen zijn ruis**: klassen met < ± 50 detecties moeten als onzeker worden beschouwd (zie de kwalitatieve controle).
* **Eén video, één scène**: alles is gemeten op één top-down opname van één product; generalisatie naar andere video's is niet getest.
* **Snelheid is hardware-afhankelijk**: gemeten op een RTX 5080 zonder batching; een tragere GPU verandert de score via de doorvoerterm.
* **SAM3 draait per frame met vijf prompts**; SAM 3.1 doet per-frame beeldinferentie via Ultralytics (Object Multiplex-videotracking is daar nog niet beschikbaar) en haalt daarmee 9,2 fps.
* **Trackingvarianten**: alleen YOLO-tracking is volledig gedraaid; de vier open-vocabulary trackingvarianten hebben nog geen volledige run en staan daarom niet in de trackingtabel.

## 9. Reproduceren

```bat
run_yolo_test.bat               :: baseline (optioneel, zit al in de vergelijking)
run_rfdetr_test.bat
run_rfdetr_seg_test.bat
run_grounding_dino_test.bat
run_yolo_world_test.bat
run_yolo_seg_test.bat
run_sam3_test.bat               :: vereist HF-toegang tot facebook/sam3 + 'hf auth login'
run_sam31_test.bat              :: vereist HF-toegang tot facebook/sam3.1 of een lokaal checkpoint
run_sam31_tracking_test.bat     :: SAM 3.1-detecties volgen met ByteTrack
run_compare_models.bat          :: schrijft outputs\Disassembly1\comparison\model_comparison.{csv,md}
```

Voor YOLO + ByteTrack gebruik je `run_tracking_test.bat`; voor SAM 3.1 + ByteTrack
is er `run_sam31_tracking_test.bat`. De overige trackingvarianten kun je direct
via Python starten, bijvoorbeeld:

```bat
.venv\Scripts\python.exe src\run_yolo_world_tracking.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_grounding_dino_tracking.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_sam3_tracking.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_sam31_tracking.py --config configs\baseline.yaml --device gpu
```

Direct via Python werkt ook voor de andere stappen, bijvoorbeeld `.venv\Scripts\python.exe src\run_yolo_world_video.py --config configs\baseline.yaml --device gpu`. De modelkeuzes, prompts, drempels en scoringsgewichten staan in `configs/baseline.yaml`; `src/run_sam3_video.py` haalt het ontbrekende BPE-vocabulaire van SAM3 eenmalig uit de sam3-repo en zet het in `models/`.

Wil je een ander gereedschap of een andere prompt testen, pas dan `tool_classes` aan (de open-vocabulary modellen gebruiken die lijst automatisch) en draai de betreffende stappen opnieuw.

## 10. Vervolgstappen

1. **Steekproef annoteren** (200–500 frames) om precision/recall en een echte F1 per model te berekenen; de opgeslagen voorbeeldframes en debug-video's zijn daar een goed startpunt voor.
2. **Auto-labeling**: SAM3- of SAM 3.1-maskers gebruiken om toolannotaties te genereren en daarmee YOLO26/RF-DETR op de toolklassen te finetunen (fase 7 van het plan).
3. **Trackingvarianten volledig draaien**: de vier open-vocabulary trackingstappen op de volledige video uitvoeren en de trackingtabel in dit rapport aanvullen; YOLO-tracking is al gedraaid.
4. **SAM 3.1 videomodel** proberen voor geprompt tracken van gereedschappen over de hele video (multiplex-checkpoint, aparte venv).
5. **Uitbreiden naar meerdere video's en camerahoeken** om de conclusie robuuster te maken, en naar connectorherkenning zodra de actieherkenning er staat.
6. **Koppeling naar DUPLO-outputs**: de toolklassen en confidence per frame zijn direct bruikbaar als input voor de Disassembly Graph en de tool-taxonomie.

## 11. Bronnen

* RF-DETR (Roboflow, ICLR 2026): <https://github.com/roboflow/rf-detr>
* Grounding DINO (IDEA-Research): <https://github.com/IDEA-Research/GroundingDINO> · modelcard <https://huggingface.co/IDEA-Research/grounding-dino-base>
* YOLO-World (AILab-CVC): <https://github.com/AILab-CVC/YOLO-World> · Ultralytics-documentatie: <https://docs.ultralytics.com/models/yolo-world/>
* SAM 3 / SAM 3.1 (Meta): <https://github.com/facebookresearch/sam3> · gewichten <https://huggingface.co/facebook/sam3> en <https://huggingface.co/facebook/sam3.1>
* Ultralytics YOLO26: <https://docs.ultralytics.com/>
