# Technisch plan – DUPLO Tool Recognition & Human Action Recognition

## 1. Doel van het project

We bouwen een modulaire video-analysepipeline voor DUPLO-disassembly waarmee we onderzoeken in hoeverre disassemblyprocessen automatisch uit video kunnen worden herkend en gestructureerd. De scope is bewust beperkt tot twee hoofddoelen: **tool recognition** (het herkennen en volgen van gebruikte gereedschappen) en **human action recognition** (het herkennen van de menselijke handelingen tijdens het demonteren).

De pipeline moet uiteindelijk informatie kunnen verzamelen over:

* welke gereedschappen worden gebruikt;
* welke gereedschappen door de tijd gevolgd worden;
* waar en hoe de handen en het lichaam van de monteur bewegen;
* welke handelingen plaatsvinden;
* wanneer een handeling begint en eindigt.

Deze informatie moet later bruikbaar zijn voor onder andere **eDIM/reDIM, Disassembly Graphs, tool taxonomy en toekomstige work instructions**.

Het project is een proof of concept. Het doel is daarom niet om direct een volledig productierijp systeem te bouwen, maar om experimenteel vast te stellen wat technisch haalbaar is en waar de beperkingen liggen.

### Scope-beslissing

Tijdens het project is besloten om **geen generiek disassembly-model te trainen**. Het finetunen van YOLO op een volledige set disassemblyobjecten (componenten, connectoren, losse onderdelen) blijkt te moeilijk en te tijdrovend voor de beschikbare projectduur. In plaats daarvan richten we ons op:

1. **Tool recognition** – het herkennen van gereedschappen met pretrained modellen en, waar haalbaar, gerichte finetuning op een kleine toolklassenlijst;
2. **Human action recognition** – het herkennen van handelingen op basis van hand- en lichaamskeypoints en toolgebruik.

Andere herkenningsvormen (generieke component- en connectorherkenning, een volledig disassemblymodel) vallen expliciet buiten scope. Ze worden alleen nog als toekomstig werk genoemd, niet als doel voor dit project.

---

# 2. Overall workflow

```text
Video
  │
  ▼
Frame extraction
  │
  ├─────────────────────────┐
  ▼                         ▼
YOLO tooldetectie         MediaPipe Hands
  │                         │
  ▼                         ▼
Bounding boxes            21 landmarks/hand
  │                         │
  ▼                         │
Tool tracking               │
(ByteTrack)                 │
  │                         │
  ▼                         ▼
Tool IDs + history + hand movement
             │
             ▼
      Gecombineerde
      temporele data
             │
             ▼
      Human action recognition
             │
             ▼
  Disassembly action labels
             │
             ▼
 eDIM / Disassembly Graph /
 Tool taxonomy / work instructions
```

De pipeline wordt in fases opgebouwd. Eerst wordt onderzocht wat met bestaande pretrained modellen mogelijk is (YOLO voor tooldetectie en tracking, MediaPipe voor handinformatie). Daarna wordt de dataset uitgebreid en wordt het action-recognitionmodel ontwikkeld. Er wordt géén generiek disassemblymodel getraind; finetuning van YOLO gebeurt uitsluitend gericht op de toolklassen waar dat haalbaar is.

---

# 3. Fase 1 – Technische baseline

## 3.1 YOLO tooldetectie

Gebruik YOLO26X met pretrained weights als eerste baseline voor de detectie van gereedschappen.

Per frame worden minimaal opgeslagen:

* frame (bronframe-index)
* timestamp
* klasse (class_name)
* confidence
* bounding box: bbox_x, bbox_y, bbox_w, bbox_h

De tracking-CSV bevat daarnaast `class_id` en de trackgeschiedenis per `track_id`.

De pretrained versie wordt gebruikt om te onderzoeken welke gereedschappen al zonder training uit de beschikbare video's kunnen worden gehaald (bijvoorbeeld schroevendraaiers, tangen en andere gangbare tools).

### Belangrijke beperking

Het pretrained model is getraind op algemene objectklassen en bevat niet noodzakelijk de gereedschappen die voor DUPLO relevant zijn.

Voorbeelden van tools die mogelijk niet of onvoldoende worden gedetecteerd:

* specifieke schroevendraaiers;
* precisiegereedschap;
* minder gangbare tools uit de DUPLO-toolset.

Wanneer een tool niet door YOLO wordt gedetecteerd, kan ByteTrack deze ook niet volgen en mist de action recognition een belangrijk signaal.

De baseline wordt daarom vooral gebruikt om de **gap tussen een generiek pretrained model en de DUPLO-specifieke toolset** te bepalen.

---

# 4. Fase 2 – Tool tracking

Wanneer YOLO een gereedschap detecteert, worden de detecties doorgegeven aan een multi-object tracker.

Er wordt gebruikgemaakt van **ByteTrack**, de tracker die ook in de Ultralytics-tooling is opgenomen.

ByteTrack moet:

* dezelfde gereedschappen over meerdere frames koppelen;
* een unieke track_id geven;
* bewegingsinformatie gebruiken (Kalman-filter);
* detecties met een lage confidence in een tweede associatiestap gebruiken;
* tijdelijke occlusion zo goed mogelijk kunnen overbruggen.

Voorbeeld:

```text
Frame 100:
screwdriver → track_id 17

Frame 101:
screwdriver → track_id 17

Frame 102:
screwdriver → track_id 17

Frame 103:
screwdriver → track_id 17
```

Hierdoor ontstaat een gereedschapgeschiedenis in plaats van alleen losse frame-detecties. Die geschiedenis is later een belangrijk signaal voor de action recognition: welk gereedschap wordt wanneer gepakt, gebruikt en weggelegd.

Omdat DUPLO-disassembly veel beweging en occlusion bevat, kan ByteTrack later worden vergeleken met alternatieven zoals **BoT-SORT**.

---

# 5. Fase 3 – Tool history

Per track_id wordt een tijdreeks opgeslagen.

Bijvoorbeeld:

```text
track_id: 17
class: screwdriver

frame 100 → bbox + center + confidence
frame 101 → bbox + center + confidence
frame 102 → bbox + center + confidence
frame 103 → bbox + center + confidence
```

Waar mogelijk worden aanvullende eigenschappen berekend:

* center_x
* center_y
* displacement
* velocity
* track age
* frames sinds laatste detectie
* visibility/status
* trajectory

Hiermee kan later worden onderzocht of het gebruik van gereedschappen (oppakken, verplaatsen, neerleggen) kan helpen bij het herkennen van disassembly actions.

---

# 6. Fase 4 – Handdetectie en handtracking

MediaPipe Hands wordt onafhankelijk van YOLO/ByteTrack gebruikt.

Per frame worden opgeslagen:

* hand (0/1-detectievolgorde; MediaPipe-handedness wordt nog niet opgeslagen);
* 21 landmarks;
* landmark x/y/z;
* zichtbaar (huidige implementatie: 1.0; handconfidence is toekomstig werk).

De handdata krijgt dezelfde:

* frame_number;
* timestamp;

als de tooldetecties en trackingdata.

Hierdoor kunnen bijvoorbeeld de volgende gebeurtenissen later worden geanalyseerd:

```text
hand approaches tool
        ↓
hand touches tool
        ↓
tool starts moving
        ↓
tool moves away
```

MediaPipe levert hiermee belangrijke informatie voor human action recognition zonder dat hiervoor direct een zwaar pose-estimationmodel nodig is.

---

# 7. Fase 5 – Definitie van acties en labels

Voordat een human-action-recognitionmodel wordt getraind, wordt een beperkte actietaxonomie opgesteld.

De voorgestelde startcategorieën zijn gebaseerd op de opdracht:

| Actie                      | Mogelijke eDIM-koppeling     |
| -------------------------- | ---------------------------- |
| Unscrewing                 | Manipulation / Disconnection |
| Tool change                | Tool change                  |
| Pick up                    | Manipulation                 |
| Remove component           | Removing                     |
| Disconnect cable/connector | Disconnection                |
| Position product           | Positioning                  |

De uiteindelijke labels worden afgestemd met de opdrachtgever.

Daarnaast wordt onderzocht welke informatie nodig is om een actie volledig te beschrijven.

Bijvoorbeeld:

```text
REMOVE
component = battery
tool = screwdriver
start = 12.4s
end = 16.8s
```

Hierdoor wordt niet alleen "remove" herkend, maar ontstaat informatie die later daadwerkelijk naar DUPLO-structuren kan worden vertaald.

---

# 8. Fase 6 – Dataset en annotatie

De beschikbare DUPLO-video's worden gebruikt om een kleine gelabelde dataset op te bouwen, gericht op tools en acties.

De dataset bestaat uit meerdere informatielagen:

### Toolannotaties

Bounding boxes voor de gereedschappen die in de video's worden gebruikt en die relevant zijn voor de action recognition (bijvoorbeeld schroevendraaiers, tangen en ander precisiegereedschap).

### Actieannotaties

Acties krijgen:

* action label;
* start frame/timestamp;
* eind frame/timestamp.

### Eventueel

Relaties tussen:

* hand;
* gereedschap;
* actie.

Bijvoorbeeld:

```text
hand → screwdriver → screw
```

Dit maakt het mogelijk om later complexere acties te herkennen. Objecten die geen gereedschap zijn (componenten, kabels, connectoren) worden hierbij alleen meegenomen voor zover ze nodig zijn om een actie te beschrijven; er wordt geen generieke object- of connectordataset opgebouwd.

---

# 9. Fase 7 – Tool recognition

Het trainen van een generiek disassemblymodel met YOLO is als te moeilijk en te tijdrovend beoordeeld en valt daarom buiten scope. In plaats daarvan bestaat fase 7 uit **tool recognition**:

* het pretrained YOLO-model wordt geëvalueerd op de gereedschappen die in de DUPLO-video's voorkomen;
* waar het pretrained model tekortschiet en er voldoende geannoteerde frames zijn, wordt YOLO **uitsluitend op een kleine toolklassenlijst** gefinetuned;
* finetuning is optioneel: de pipeline moet ook met het pretrained model kunnen werken.

Naast YOLO worden in de baseline ook RF-DETR (detectie en segmentatie), Grounding DINO, YOLO-World, YOLO26-seg, SAM3 en SAM 3.1 op dezelfde video getest, zodat de toolherkenning van de modellen onderling te vergelijken is (zie de sectie *Modelvergelijking: toolrecognition* in de README). Open-vocabulary modellen kunnen de toolklassen als tekstprompt krijgen en hebben daardoor geen finetuning nodig; de gesloten-vocabulaire modellen blijven beperkt tot hun COCO-klassen. De meetresultaten en het advies staan in [Modelvergelijking.md](Modelvergelijking.md). ByteTrack-tracking draait op YOLO, YOLO-World, Grounding DINO, SAM3 en SAM 3.1 (`run_tracking.py` + vier `*_tracking.py` varianten) met dezelfde tracking-CSV; `compare_models.py` rapporteert tracking apart van de detectie-score.

De code is modulair: `src/duplo/detection.py` definieert het `Detector`-protocol (`detect(frame, conf) -> list[Detection]`) met `create_detector()`-factory, `src/duplo/schema.py` de CSV-schema's, `src/duplo/annotate.py` de debug-tekenhelpers en `src/duplo/pipeline.py` de gedeelde runtime (`setup_runtime`, `run_detection_loop`). Nieuwe detectoren hoeven alleen het protocol te implementeren.

De pretrained COCO-klassen worden niet één-op-één overgenomen. De toolklassenlijst wordt afgestemd op de gereedschappen uit de video's en de DUPLO-toolset.

Bijvoorbeeld:

```text
0 = screwdriver
1 = wrench
2 = pliers
3 = cutter
4 = other_tool
```

Alleen gereedschappen komen als klasse in het model. Een hand kan bijvoorbeeld als `person` worden gedetecteerd, maar `person` is geen relevante klasse omdat MediaPipe Hands specifiek voor handinformatie wordt gebruikt.

De pretrained YOLO-weights worden gebruikt als startpunt voor transfer learning.

Als finetuning niet haalbaar blijkt, wordt de toolherkenning gebaseerd op de pretrained klassen en wordt dit als beperking gerapporteerd.

---

# 10. Fase 8 – Human action recognition

Pas nadat voldoende temporele data en actielabels beschikbaar zijn, wordt onderzocht welke methode geschikt is voor daadwerkelijke herkenning van menselijke handelingen.

Mogelijke kandidaten zijn:

* ST-GCN op MediaPipe keypoints;
* temporal action detection;
* video classification;
* een eigen lichtgewicht temporal model;
* een combinatie van tool trajectories en hand landmarks;
* zero-shot handactie-voorspelling met VITRA (`src/run_vitra_test.py`; zie de README).

De keuze wordt gebaseerd op:

* hoeveelheid beschikbare data;
* benodigde nauwkeurigheid;
* computationele eisen;
* interpreteerbaarheid;
* benodigde temporele informatie.

Het human-action-recognitionmodel hoeft dus niet noodzakelijk end-to-end op de volledige video te werken.

Een mogelijke input is bijvoorbeeld:

```text
Hand landmarks
       +
Tool track history
       +
Tool classes
       ↓
Temporal model
       ↓
"Unscrewing" / "Tool change" / "Remove component"
```

---

# 11. Fase 9 – Output

De pipeline moet uiteindelijk een machine-readable dataset kunnen produceren.

Bijvoorbeeld:

```text
timestamp
frame_number
track_id
object_class
confidence
bbox
center
velocity
hand_landmarks
action
action_start
action_end
```

De velden `track_id`, `object_class` en `bbox` hebben in dit project betrekking op **gereedschappen**; het formaat is bewust generiek gehouden zodat een latere uitbreiding naar andere objecten mogelijk blijft.

Hiermee kan een gedetecteerde gebeurtenis bijvoorbeeld worden weergegeven als:

```text
12.4s – 16.8s

ACTION:
Unscrewing

TOOL:
Screwdriver

HAND:
Right hand

CONFIDENCE:
0.91
```

Deze informatie kan later worden vertaald naar DUPLO-outputs.

---

# 12. Relatie met DUPLO

De uiteindelijke pipeline moet niet alleen een computer vision-demo opleveren, maar informatie produceren die aansluit op de bestaande DUPLO-structuur.

### Disassembly Graph

Gedetecteerde:

* tools;
* menselijke acties;
* volgorde;

kunnen later worden vertaald naar relaties en processtappen.

### eDIM / reDIM

Herkende acties en eventueel hun duur kunnen worden gebruikt als input voor:

* identifying;
* manipulation;
* positioning;
* disconnection;
* removing;
* tool change.

### Tool taxonomy

Gedetecteerde gereedschappen en tool changes kunnen worden gekoppeld aan de bestaande toolcategorieën.

### Work instructions

De gedetecteerde sequence:

```text
pick up screwdriver
→ unscrew
→ remove component
→ disconnect cable
→ position component
```

kan later dienen als basis voor automatisch of semi-automatisch gegenereerde werkinstructies. De focus ligt hierbij op de **acties en het gebruikte gereedschap**, niet op het herkennen van alle losse onderdelen.

---

# 13. Evaluatie

De evaluatie wordt per onderdeel afgestemd op de gekozen methode.

### Tool recognition

* Precision
* Recall
* F1-score
* mAP

### Tracking

* ID consistency
* ID switches
* tracking accuracy
* eventueel standaard MOT-metrics indien relevant

### Human action recognition

Afhankelijk van de gekozen methode:

* Precision
* Recall
* F1-score
* confusion matrix
* temporal localization/segmentation quality

### Praktische evaluatie

Daarnaast wordt gekeken naar:

* processing time;
* visuele kwaliteit van de tracking;
* failure cases;
* occlusion;
* kleine objecten;
* false positives;
* false negatives.

Wanneer de dataset te klein is voor betrouwbare statistische conclusies, wordt een kwalitatieve demonstratie gecombineerd met een kritische analyse van de beperkingen.

---

# 14. Belangrijke uitgangspunten

Focus van dit project: **tool recognition** en **human action recognition**. Andere herkenningsdoelen (generieke objectherkenning, connectoren) vallen buiten scope.

De eerste versie hoeft niet direct het volledige systeem te bevatten.

De ontwikkeling verloopt stapsgewijs:

```text
FASE 1
Pretrained YOLO + MediaPipe
        ↓
Wat kunnen we zonder training?

FASE 2
Tool tracking
        ↓
Kunnen we gereedschappen door de tijd volgen?

FASE 3
Annotatie + dataset
        ↓
Welke tools en acties zijn relevant?

FASE 4
Tool recognition
        ↓
Kunnen we DUPLO-specifieke tools herkennen?

FASE 5
Human action recognition
        ↓
Kunnen we menselijke disassembly actions herkennen?

FASE 6
DUPLO mapping
        ↓
Kunnen we de resultaten koppelen aan
eDIM / Disassembly Graph / tool taxonomy?
```

Het project blijft daarmee een proof of concept en hoeft niet alle bovenstaande fases volledig uit te voeren wanneer de beschikbare data daarvoor onvoldoende is. Ook geldt: er wordt geen generiek disassemblymodel getraind; het zwaartepunt ligt op tool recognition en human action recognition, en de waarde zit in het experimenteel vaststellen van wat haalbaar is en waar de beperkingen liggen.

---

# 15. Kern van de architectuur

**YOLO26X**

> WHICH tools are present?

**ByteTrack / tracking**

> WHICH detected tool is the same tool over time?

**MediaPipe Hands**

> WHERE are the hands and how do they move?

**Tool history**

> HOW do tools move over time?

**Human action recognition**

> WHAT human action happened?

**DUPLO mapping**

> HOW does that action translate into eDIM, Disassembly Graphs and work instructions?

De code wordt modulair ontworpen zodat YOLO, tracker, MediaPipe en het latere human-action-recognitionmodel afzonderlijk vervangen en geëvalueerd kunnen worden.
