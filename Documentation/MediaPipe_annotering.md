# Handleiding: acties annoteren met MediaPipe-handlandmarks

## 1. Doel

In deze handleiding geef je videobeelden van een demontageproces actielabels.
De labels kunnen later worden gekoppeld aan de handlandmarks om een model voor
human action recognition te trainen.

Je hoeft hiervoor niet te kunnen programmeren. Je volgt de stappen en noteert
welke actie op welk moment in de video plaatsvindt.

**Belangrijk onderscheid:**

- MediaPipe detecteert handlandmarks: de posities van maximaal twee handen en
  21 landmarks per gedetecteerde hand.
- MediaPipe bepaalt niet welke demontageactie iemand uitvoert.
- De annotator bepaalt handmatig welke actie plaatsvindt en wanneer die begint
  en eindigt.
- De combinatie van handlandmarks en actie-framenummers vormt de basis voor de
  dataset.

## 2. Stap 1 — Een video toevoegen

1. Plaats de video in de map `data`.
2. Gebruik een duidelijke bestandsnaam, bijvoorbeeld `Disassembly2.mp4`.
3. Controleer of de video afspeelt en of de handen en handelingen goed
   zichtbaar zijn. Kies bij voorkeur video's waarin de handen zo min mogelijk
   buiten beeld vallen.

## 3. Stap 2 — De configuratie aanpassen

1. Open [`configs/baseline.yaml`](../configs/baseline.yaml).
2. Pas `video_path` aan, bijvoorbeeld:

   ```yaml
   video_path: "data/Disassembly2.mp4"
   ```

3. Pas ook `output_dir` aan naar een aparte map voor deze video, bijvoorbeeld:

   ```yaml
   output_dir: "outputs/Disassembly2"
   ```

4. Sla het bestand op.

Pas voor deze workflow alleen `video_path` en `output_dir` aan, tenzij het
projectteam andere instellingen heeft afgesproken. `output_dir` is belangrijk:
de code gebruikt die instelling letterlijk. De uitvoer wordt dus niet
automatisch naar een map met dezelfde naam als de video verplaatst.

## 4. Stap 3 — MediaPipe uitvoeren

1. Ga naar de hoofdmap van het project.
2. Voer op Windows het volgende script uit:

   ```bat
   run_handpose_test.bat
   ```

   Je kunt optioneel een apparaat kiezen, bijvoorbeeld
   `run_handpose_test.bat --device cpu`. Op Windows gebruikt deze MediaPipe-
   pipeline CPU, omdat de geïnstalleerde Windows-build de MediaPipe
   GPU-delegate niet bevat. Op andere platformen kan de configuratie GPU
   gebruiken en bij een niet-beschikbare GPU-delegate terugvallen op CPU.
3. Wacht tot het programma klaar is.
4. Open de map `outputs/<videonaam>/handpose/`, waarbij `<videonaam>` de naam
   van de gekozen `output_dir` is, bijvoorbeeld
   `outputs/Disassembly2/handpose/`.

De belangrijkste uitvoer is:

- `hand_landmarks.csv`: de automatisch gegenereerde MediaPipe-uitvoer;
- `handpose_summary.json`: informatie over de verwerking;
- `handpose_debug.mp4`: een debugvideo met getekende handlandmarks en een
  zichtbare framecounter, als `save_handpose_video` aanstaat;
- eventuele JPG-voorbeeldframes.

`hand_landmarks.csv` bevat deze kolommen:

| Kolom | Betekenis |
| --- | --- |
| `video` | Het videopad uit de configuratie |
| `frame` | Het oorspronkelijke framenummer |
| `hand` | Index van de gedetecteerde hand in dat frame |
| `landmark` | Landmarknummer, van 0 tot en met 20 |
| `x`, `y` | Genormaliseerde landmarkcoördinaten |
| `z` | Relatieve dieptecoördinaat van MediaPipe |
| `zichtbaar` | `1.0` voor een landmark die door MediaPipe is geregistreerd |

De standaardconfiguratie heeft `frame_stride: 2`. Daardoor wordt niet elk
videoframe verwerkt, maar elk tweede frame. Het `frame`-nummer verwijst nog
steeds naar het oorspronkelijke videoframe. De handpose-CSV bevat geen
actielabels, begin- of eindframes.

### De framecounter in de debugvideo gebruiken

In `handpose_debug.mp4` staat linksboven op elk frame `Frame: <nummer>`.
Dit is het oorspronkelijke framenummer van de video en is hetzelfde nummer
dat in de kolom `frame` van `hand_landmarks.csv` staat. Gebruik deze counter
om tijdens het bekijken de exacte `start_frame` en `end_frame` van een actie
over te nemen in het actie-CSV-bestand.

Door `frame_stride: 2` bevat de debugvideo alleen de verwerkte frames. De
framenummers in beeld kunnen daardoor bijvoorbeeld `54`, `56`, `58` zijn.
Gebruik het getoonde oorspronkelijke nummer; tel niet simpelweg de positie
van het frame in de debugvideo. Een actie die tussen twee getoonde frames
begint, moet je volgens dezelfde frameconventie met het dichtstbijzijnde
relevante oorspronkelijke framenummer annoteren.

**Controleer de uitvoer voordat je verdergaat.** Als het script stopt met een
foutmelding of `hand_landmarks.csv` niet verschijnt, annoteer deze video dan
niet verder. Vraag hulp aan een groepslid dat bekend is met de code.

## 5. Stap 4 — De video bekijken en acties annoteren

Open de originele video in een videospeler waarmee je nauwkeurig kunt pauzeren,
terugspoelen en de huidige tijd kunt aflezen. Bekijk de video eerst volledig
zodat je de volgorde van de handelingen kent.

Noteer per actie:

- **Startframe:** het framenummer waarop de actie begint;
- **Eindframe:** het framenummer waarop de actie eindigt;
- **Actielabel:** de naam van de actie;
- **Opmerking:** optioneel, bijvoorbeeld bij een moeilijk te beoordelen
  moment.

Gebruik framenummers als tijdseenheid. Het framenummer begint in deze pipeline
bij `0` en verwijst naar het oorspronkelijke videoframe. Gebruik dus niet de
verstreken tijd van de videospeler als annotatiewaarde. Als een videospeler
alleen seconden toont, reken dan om met:

```text
frame = tijd in seconden × fps
```

Rond af naar een geheel framenummer en gebruik voor alle annotaties dezelfde
fps. Controleer bij voorkeur het framenummer met de `Frame: <nummer>`-counter in
`handpose_debug.mp4`, omdat die dezelfde framenummering gebruikt als de CSV.

### Afgesproken actielabels

Gebruik uitsluitend deze labels:

| Label | Wanneer gebruik je dit? |
| --- | --- |
| `screw_unscrew` | Een schroef wordt losgedraaid. |
| `screw_tighten` | Een schroef wordt vastgedraaid. |
| `tool_pick_up` | Iemand pakt gereedschap op. |
| `tool_put_down` | Iemand legt gereedschap neer. |
| `pry_component` | Iemand maakt een onderdeel met een hefbeweging los. |
| `rotate_object` | Iemand roteert een object. |
| `AO` | Er vindt geen afgesproken doelactie plaats. |

De labels zijn Engelstalig om één consistente naamgeving in de dataset te
behouden. Deze actielabels worden momenteel niet door MediaPipe of door
`run_handpose_test.bat` voorspeld; ze worden handmatig toegevoegd.

Annotaties worden gemaakt op basis van wat daadwerkelijk zichtbaar is. Raad
niet welke actie plaatsvindt wanneer de video onvoldoende informatie geeft.
Zet een opmerking bij twijfelgevallen en bespreek die later.

## 6. Stap 5 — Het begin en einde bepalen

Gebruik voor alle video's dezelfde regels, zodat trainingsvoorbeelden
vergelijkbaar blijven.

### Voorbeelden

- **Schroef losdraaien**
  - Begin: de beweging die gericht is op het losdraaien begint.
  - Einde: het losdraaien stopt of gaat over in een andere actie.
- **Gereedschap pakken**
  - Begin: de hand beweegt gericht naar het gereedschap om het te pakken.
  - Einde: het gereedschap is opgepakt en de pakbeweging is afgerond.
- **Gereedschap neerleggen**
  - Begin: de hand beweegt het gereedschap naar de plek waar het wordt
    neergelegd.
  - Einde: het gereedschap is neergelegd en de neerlegbeweging is afgerond.
- **Onderdeel opwippen**
  - Begin: de beweging waarmee het onderdeel wordt losgemaakt begint.
  - Einde: de hefbeweging stopt of het onderdeel komt los.

### Algemene regels

- Gebruik de oorspronkelijke framenummers van de video.
- Bepaal begin- en eindframes zo nauwkeurig mogelijk.
- Dwing niet iedere beweging in een doelklasse als de actie niet duidelijk
  herkenbaar is.
- Noteer bij een overgang twee afzonderlijke acties met elk een eigen interval.
- Als de hele video moet worden gelabeld, gebruik je `AO` voor momenten zonder
  doelactie in plaats van stukken ongemarkeerd te laten.
- Zorg dat intervallen niet negatief zijn en dat de framenummers binnen de
  videolengte vallen.

## 7. Stap 6 — Overige momenten (`AO`)

Gebruik `AO` voor stukken video waarin geen afgesproken doelactie plaatsvindt,
bijvoorbeeld:

- de persoon kijkt naar het werkstuk;
- de handen rusten tijdelijk;
- de persoon wacht op de volgende stap;
- de persoon houdt een onderdeel vast zonder een doelactie uit te voeren.

Gebruik `AO` niet alleen omdat een actie moeilijk te beoordelen is. Zet in dat
geval een opmerking en vraag om beoordeling. Verander het label zodra een
doelactie begint of eindigt.

## 8. Stap 7 — De annotaties opslaan

Bewaar de handmatige actieannotaties in een afzonderlijk bestand. Wijzig
`hand_landmarks.csv` niet. Gebruik bijvoorbeeld:

`Disassembly2_actions.csv`

Een geschikt afgesproken formaat is:

```csv
video,start_frame,end_frame,label,notes
Disassembly2.mp4,72,144,tool_pick_up,
Disassembly2.mp4,144,243,screw_unscrew,
Disassembly2.mp4,243,285,AO,"Korte pauze"
Disassembly2.mp4,285,336,pry_component,
```

Dit actie-CSV-bestand wordt niet door `run_handpose_test.bat` gegenereerd en
de huidige code valideert deze kolomnamen niet. Stem het definitieve formaat af
met het projectteam voordat de datasetverwerking wordt gebouwd. Houd de
annotaties gescheiden van de automatisch gegenereerde handlandmarks.

## 9. Stap 8 — Het werk controleren

Controleer de video en het annotatiebestand nog één keer:

- alle relevante doelacties zijn geannoteerd;
- alleen afgesproken labels zijn gebruikt;
- alle begin- en eindframes zijn ingevuld;
- de frames vallen binnen de videolengte;
- er zijn geen negatieve of onbedoeld overlappende intervallen;
- acties zijn niet onnodig samengevoegd;
- `AO` is gebruikt voor echte overige momenten;
- twijfelgevallen hebben een opmerking;
- het bestand heeft de juiste videonaam;
- het oorspronkelijke `hand_landmarks.csv` is niet aangepast.

Laat bij voorkeur een ander groepslid een deel controleren. Bespreek
verschillen en leg gewijzigde interpretatieregels vast.

## 10. Relatie met de uiteindelijke dataset

Het aantal videoframes is niet hetzelfde als het aantal onafhankelijke
trainingsvoorbeelden. Opeenvolgende frames kunnen vrijwel identieke
handposities bevatten. De datasetverwerking kan de handlandmarks daarom later
in tijdvensters verwerken en elk venster koppelen aan het actielabel dat op dat
moment geldt.

Verdeel video's op videoniveau over training, validatie en test. Zo komen
sterk gelijkende frames uit dezelfde video niet in meerdere datasets terecht.

Het resultaat van deze workflow bestaat uit:

1. het automatisch gegenereerde `hand_landmarks.csv` met handgegevens en
   framenummers;
2. het handmatig gemaakte actie-CSV-bestand met begin- en eindframes en
   actielabels.

Samen vormen deze bestanden de basis voor action recognition. De koppeling
tussen beide bestanden en de uiteindelijke tijdvensterlogica moeten door de
datasetverwerking worden uitgevoerd.
