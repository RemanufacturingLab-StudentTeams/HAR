For communication protocols. 

Schemas: These should be extensively documented. A comprehensive overview of the schema should be in the schemas directory, with a detailed README.md. The idea is that a collaborator should not have to be able to understand the internal workings of a component in order to still communicate with it over a given network protocol.

## Actieclassificatie-CSV (`*_actions.csv`)

Eén rij per geclassificeerde videoclip. De kolomdefinitie staat centraal in
`src/duplo/schema.py` (`ACTION_COLUMNS`).

| Kolom | Type | Betekenis |
| --- | --- | --- |
| `video` | string | Pad naar de invoervideo. |
| `frame` | integer | Eerste videoframe van de clip (zelfde conventie als detectie). |
| `timestamp` | float | `frame / fps` in seconden. |
| `clip_start` | integer | Eerste frame van de clip (inclusief). |
| `clip_end` | integer | Laatste frame van de clip (exclusief). |
| `klasse` | string | Voorspelde actie uit `action_classes`. |
| `confidence` | float of leeg | Confidence van de voorspelde klasse (0-1); leeg bij VLM's zonder gekalibreerde score. |
| `start_s` | float of leeg | Door het model voorspelde starttijd in seconden (absoluut). |
| `end_s` | float of leeg | Door het model voorspelde eindtijd in seconden (absoluut). |
| `top_k` | string | Top-k scores als `label=score; label=score`. |
| `raw_response` | string | Ruwe VLM-uitvoer voor foutanalyse (leeg bij classifiers). |

De modellen (X-CLIP, InternVideo2, ActionCLIP, SmolVLM2, LLaVAction en
Qwen3-VL) gebruiken hetzelfde schema, zodat `compare_models.py` ze in
`action_comparison.csv` kan vergelijken. `start_s`/`end_s` zijn alleen gevuld
bij modellen die temporele grenzen voorspellen. `action_eval.py` meet die
tijden apart tegen `action_ground_truth` (IoU, R@IoU en boundary-MAE).
VITRA's `vitra_actions.csv` valt hier bewust buiten: dat zijn continue
handacties per voorspelde stap, geen clipclassificatie.

