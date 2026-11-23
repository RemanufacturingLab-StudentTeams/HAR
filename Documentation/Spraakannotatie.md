# Spraakannotatie

Annoteren met spraak: wie een product demonteert, spreekt tijdens het werk korte vaste commando's in
(bijv. "start losdraaien" … "klaar"). Een spraakmodel schrijft dat uit met tijden per woord, en het
script maakt er automatisch een actie-annotatie van: begin- en eindframe, label, eDIM-categorie,
gereedschap en toelichting. De uitkomst komt in een Excel-bestand waarin de spreker elke regel controleert.

Alles draait lokaal (geen cloud) en op een gewone laptop zonder GPU.

## Waarom een aparte omgeving

De spraakmodellen gebruiken geen torch en draaien in een eigen omgeving `.venv-asr`.
Zo blijft de hoofd-`.venv` (torch / transformers voor de beeldmodellen) onaangeroerd.

## Gebruikte modellen

| Model | Pakket | Rol |
|---|---|---|
| Vosk `vosk-model-small-nl-0.22` | `vosk` | Herkent alleen de vaste commandolijst; zeer snel, precieze tijden |
| Whisper small / medium | `faster-whisper` | Schrijft alles uit (ook vakwoorden); controleert de commando's van Vosk |

Aanbevolen: de **combinatie** (`--combi small`): tijden van Vosk, controle en toelichting van Whisper small.

## Installatie en gebruik

De eerste keer maakt `run_asr_test.bat` zelf `.venv-asr` aan, installeert `requirements-asr.txt`
en downloadt het Vosk-model naar `models/` (~40 MB). Whisper-modellen worden bij de eerste run
automatisch gedownload (small ~0,5 GB, medium ~1,5 GB).

1. Zet een opname in `data/` (bijv. `data/opname_3min.m4a`; iPhone-dictafoon, wav of mp4 werkt ook).
2. Annotatie maken:
   ```
   run_asr_test.bat data\opname_3min.m4a --combi small --vanaf 88
   ```
   `--vanaf 88` laat acties vóór 1:28 weg (testdelen van de opname); weglaten als die er niet zijn.
3. Excel-bestand voor de spreker maken:
   ```
   run_asr_excel.bat outputs\opname_3min --vanaf 88
   ```
4. Unittests (geen model of audio nodig): `run_unit_tests.bat` draait ook `tests/test_asr_annotation.py`.

Andere opties: `--models vosk,small,medium` (modellen los vergelijken), `--no-hint` (Whisper zonder woordenlijst).

## Uitvoer (`outputs/<opname>/`)

| Bestand | Inhoud |
|---|---|
| `<model>_words.csv` | Elk herkend woord met begin- en eindtijd |
| `<model>_commands.csv` | Gevonden commando's, wat er gehoord werd en score |
| `<model>_actions.csv` | `video,start_frame,end_frame,label,edim,start_s,end_s,notes` (30 fps) |
| `combi_vosk+<model>_actions.csv` | Idem + `tool` en `transcript` |
| `combi_vosk+<model>_verworpen.csv` | Commando's die Whisper niet bevestigde |
| `summary.csv` | Rekentijd en aantallen per model |
| `<opname>_resultaten.xlsx` | Excel met controlekolom (ja/nee), transcript en berekening van score en tijdwinst |

## Spreekprotocol

Elke handeling begint met een commando en eindigt met **klaar**. **fout** schrapt de vorige handeling.

| Commando | eDIM | Label |
|---|---|---|
| zoek verbinding | Identity | `identify` * |
| draai om | Manipulate | `rotate_object` |
| vasthouden | Fix | `fixate` * |
| pak schroevendraaier / pak spatel / pak tang / pak sleutel / pak hamer | Tool Change | `tool_pick_up` |
| leg neer | Tool Change | `tool_put_down` |
| plaats gereedschap | Tool Position | `tool_position` * |
| start losdraaien / start vastdraaien | Disconnect | `screw_unscrew` / `screw_tighten` |
| start wrikken | Disconnect | `pry_component` |
| start loskoppelen | Disconnect | `disconnect_other` * |
| verwijder onderdeel | Removal | `component_remove` * |

\* Nieuw label ten opzichte van `MediaPipe_annotering.md`; nog af te stemmen met het team.

De commandolijst en een tekst met vakwoorden staan in `configs/asr/`.

## Resultaten (samenvatting)

Test op één opname van 3:24 (computermuis, 14 handelingen, één spreker, stille kamer), laptop i3-8130U zonder GPU:

| | Vosk | Whisper small | Whisper medium | Combinatie |
|---|---|---|---|---|
| Handelingen gevonden | 14/14 | 14/14 | 14/14 | 14/14 |
| Valse commando's bij vrije tekst | 2 | 0 | 0 | 0 |
| Gem. tijdfout eind actie | 0,02 s | 0,46 s | 1,08 s | 0,02 s |
| Vakwoorden goed | – | 14/16 | 15/16 | via small |
| Rekentijd | 10 s | 1,5 min | 14 min | 1,5 min |

De volledige uitvoer staat in `outputs/opname_3min/` (niet in git; opnieuw te maken met `run_asr_test.bat`). Het onderzoeksverslag, de figuur en de meetdata staan in `docs/experimental/spraakannotatie/`.

## Bekende beperkingen

- Vosk kent alleen woorden uit zijn woordenboek; `losdraaien` wordt daarom intern als `los draaien` gezocht.
- Whisper-woordtijden zijn minder precies (pauzes worden aan woorden geplakt); daarom komen de tijden van Vosk.
- Opnames met stemmen zijn persoonsgegevens: zet ze **niet** in git (`data/*.m4a` en `data/*.wav` staan in `.gitignore`).
