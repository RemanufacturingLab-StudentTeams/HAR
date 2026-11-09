# Video-invoer

Zet hier je invoervideo('s) neer, bijvoorbeeld `data/Disassembly1.mp4`. Video's
staan bewust niet in git (`.gitignore`: `data/*.mp4`), zodat de repository
klein blijft.

- `configs/baseline.yaml` verwijst standaard naar `data/Disassembly1.mp4`.
- Bestaat dat pad niet en staat er precies één `.mp4` in deze map, dan gebruikt
  de pipeline automatisch die video.
- Staan er meerdere video's, zet dan `video_path` in de config of geef een
  eigen config mee met `--config`.
- Ondersteunde formaten zijn alles wat OpenCV kan lezen (mp4/H.264 wordt
  aanbevolen).

## Actie-ground-truth (optioneel)

Voor classificatie-accuracy en temporele IoU van actiemodellen kun je een
`data/action_ground_truth.csv` aanmaken (zie
`data/action_ground_truth.example.csv`) en naar dat bestand verwijzen met
`action_ground_truth` in de config:

```csv
video,action,start_s,end_s
Disassembly1.mp4,unscrewing a screw,12.50,18.20
Disassembly1.mp4,picking up a tool,18.20,21.00
Disassembly1.mp4,idle,21.00,24.50
```

De actienamen moeten overeenkomen met `action_classes` (hoofdletter-
ongevoelig). Zonder ground truth schrijft de benchmark alleen aantallen,
confidence en timing; de accuracy-kolommen blijven dan leeg.
