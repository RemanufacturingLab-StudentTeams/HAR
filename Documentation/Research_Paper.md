# From Disassembly Video to Structured Work Steps

## A Research Synthesis of the DUPLO Human Action Recognition Proof of Concept

**Manuscript type:** Workspace-based research synthesis · **Project:** DUPLO — Human Action Recognition · **Evidence cut-off:** 25 November 2026

## Abstract

The DUPLO project explores whether video and locally recorded speech can help structure manual disassembly processes for later use in Disassembly Graphs, eDIM/reDIM, tool taxonomies, and work instructions. This paper synthesizes the project brief, technical plan, implementation, experiment reports, benchmark outputs, annotation guidance, and automated tests in this workspace. The main video pipeline combines prompted tool detection, ByteTrack histories, MediaPipe hand landmarks, and clip-based action classifiers; separate experiments investigate VITRA hand-motion prediction and speech-assisted annotation.

On one 321-second, top-down phone-disassembly video, open-vocabulary models detected the screwdriver more consistently than closed-vocabulary COCO models. SAM 3.1 achieved the highest heuristic detection score (88.4), while YOLO-World reached the highest reported pipeline throughput (72.2 frames/s). These are proxy results, not accuracy estimates: the video has no independently annotated tool ground truth. Three clip classifiers processed 322 clips each, but their strongly concentrated predictions cannot establish recognition accuracy because no video action ground truth was supplied. MediaPipe produced 159,348 landmark rows, and VITRA generated continuous hand-motion predictions; neither output is itself a discrete action annotation.

A separate speech-annotation experiment on one 3-minute-24-second mouse-disassembly recording found 14 of 14 spoken actions with a Vosk–Whisper-small combination, with Vosk supplying action times and Whisper filtering false command detections and adding context. This result is promising for human-reviewed annotation, but is based on one speaker and a quiet room. Overall, the workspace demonstrates a modular, documented proof of concept with rerunnable workflows and useful feasibility findings. It does not yet demonstrate generalizable visual action recognition. The next research priority is a versioned, independently reviewed dataset spanning multiple products, sessions, camera views, and speakers, followed by task-specific evaluation on held-out videos.

**Keywords:** disassembly; human action recognition; tool detection; video understanding; hand landmarks; speech annotation; circular manufacturing

## 1. Introduction

Repair, remanufacturing, and recycling depend on understanding how products are taken apart: which tools are used, which actions occur, and in what order. DUPLO stores relevant process knowledge in structures such as Disassembly Graphs and eDIM/reDIM tables, but much of the information from disassembly workshops is currently interpreted and recorded manually. The project brief therefore asks whether computer vision and human action recognition can help make this process more automatic or semi-automatic [1].

The project deliberately narrowed its technical scope. Rather than train a general-purpose detector for components, connectors, and all disassembly objects, it focuses on (1) recognizing and following tools and (2) exploring human action recognition from video. This reflects the limited data and project time described in the technical plan [2]. The desired output is not simply a frame label: it should eventually include an action, its time interval, relevant tool information, and an interpretable mapping to DUPLO process concepts.

This synthesis addresses three questions:

1. What can the current workspace demonstrate about tool detection, hand and motion signals, and video action classification?
2. What does the separate speech-annotation work contribute to creating time-aligned action labels?
3. What evidence and system changes are needed before these experiments can support reliable DUPLO process analysis?

## 2. Materials and methods

### 2.1 Evidence included

The synthesis draws on the assignment and technical plan, the repository README and task-specific documentation, source code and tests, the checked-in benchmark reports and JSON summaries, and the experiment reports in Word and PDF format. Results are reported according to the task and dataset on which they were obtained. Planned capabilities are identified as future work rather than treated as completed results.

The video, model weights, and several generated run directories are local assets rather than portable, independently annotated research data. The principal video results are documented for `data/Disassembly1.mp4`; the speech results come from separate recordings. These datasets are not merged or treated as interchangeable.

### 2.2 Video pipeline

The shared pipeline reads video frames, applies each model, writes machine-readable CSV output, and records timing and environment summaries. Tool detectors implement a common detector interface. Detection, tracking, hand landmarks, clip classification, and VITRA use separate output schemas and task-aware benchmark summaries [3, 4].

The primary video is a 1280 × 720, 30 fps, approximately 321-second top-down recording of phone disassembly (9,639 source frames). The tool comparison processed every second frame (4,820 frames), using a confidence threshold of 0.25 and five prompted classes: `screwdriver`, `wrench`, `pliers`, `cutter`, and `hammer`. The documented comparison ran on Windows with an NVIDIA RTX 5080 (16 GB VRAM); its recorded software stack included Python 3.13 and CUDA-enabled PyTorch [5].

Eight detector or segmenter configurations were compared: YOLO26x, RF-DETR, YOLO-World, Grounding DINO, YOLO26x-seg, RF-DETR-seg, SAM3, and SAM 3.1. Because the video has no tool ground truth, the comparison reports class coverage, the fraction of processed frames with a prompted tool detection, throughput, and a weighted heuristic score. This score is a feasibility-ranking device, not precision, recall, or mAP [5].

Action classification used seven configured labels and a stride of approximately one clip per second. The selected benchmark report covers ActionCLIP, InternVideo2, and X-CLIP, each on 322 clips from the same video. Since no independent action labels were supplied, the report records prediction distributions and runtime but leaves classification and temporal-accuracy metrics unavailable [6].

### 2.3 Auxiliary signals and speech annotation

MediaPipe Hands records up to 21 landmarks for each detected hand, indexed by source frame. Its output is a pose signal; action labels and action boundaries are manually annotated separately [7]. VITRA is a distinct, optional experiment: given an image, a text instruction, and a reconstructed MANO hand state, it predicts a chunk of future continuous hand motion. It does not emit a discrete class such as “unscrewing” [3].

The speech pipeline is a separate local workflow. It combines Vosk command recognition with Whisper transcription, maps fixed commands to labels and eDIM categories, converts times to frames, and produces an Excel sheet for human review [8]. The principal speech experiment used one 3:24 recording of a computer mouse disassembly by one speaker in a quiet room. A second, separate experiment evaluated Wav2Vec 2.0 Dutch and Meta MMS on a 27.96-second synthetic-voice recording; those results are treated as a technical pilot, not as evidence from real disassembly speech [9].

## 3. Results

### 3.1 Tool recognition

The open-vocabulary models found the prompted screwdriver in the video, while the closed-vocabulary COCO models did not produce detections matching the target tool labels. The latter frequently classified the screwdriver as `knife` or `scissors`, illustrating a label-vocabulary mismatch as well as a limitation of the pretrained baseline.

| Model | Prompted classes detected | Tool frames | Pipeline throughput (fps) | Heuristic score |
|---|---:|---:|---:|---:|
| SAM 3.1 | 5/5 | 100.0% | 9.2 | 88.4 |
| YOLO-World v2 X | 4/5 | 35.1% | 72.2 | 72.5 |
| Grounding DINO base | 3/5 | 98.2% | 7.9 | 69.3 |
| SAM3 | 3/5 | 99.9% | 2.9 | 59.8 |
| RF-DETR Large | 0/5 | 0% | 73.1 | 30.0 |
| RF-DETR Seg 2XLarge | 0/5 | 0% | 36.0 | 30.0 |
| YOLO26x | 0/5 | 0% | 65.3 | 30.0 |
| YOLO26x-seg | 0/5 | 0% | 46.9 | 30.0 |

SAM 3.1 had the highest heuristic score and broadest prompted-class coverage, but its 9.2 fps throughput was below the 15 fps real-time target used by the report. YOLO-World was much faster and detected the screwdriver in 35.1% of processed frames, but it did not provide the near-continuous tool-frame coverage reported by the slower segmentation and grounding models. Grounding DINO and SAM3 also found the screwdriver in nearly all frames, but at substantially lower throughput.

Across the four open-vocabulary systems, the screwdriver was the reliable signal. Counts for secondary classes were sparse and, on visual inspection, often represented false positives on hands or other objects. The aggregate score favors the prompted classes and includes a speed term; because there is no ground truth and the open-vocabulary models receive those class names as prompts, the ranking should be interpreted as a comparison of practical feasibility on this scene—not as a measured ranking of detector accuracy [5].

### 3.2 Tracking

The YOLO + ByteTrack run processed 4,820 frames and produced 20,666 track observations at a reported pipeline throughput of 54.39 fps. A separate SAM 3.1 + ByteTrack run also processed 4,820 frames, producing 18,259 observations at 5.62 fps. Both pipelines export track IDs, boxes, centers, velocities, track ages, and visibility status [3, 10].

These are execution and output-volume results only. The workspace does not report identity ground truth, ID-switch counts, or standard multi-object tracking metrics for these runs. A complete run therefore establishes that the tracking path executes and serializes histories; it does not demonstrate identity consistency. In addition, detector vocabulary can affect whether a tracked object is a correctly named tool in the first place.

### 3.3 Hand landmarks and VITRA

On the primary video, MediaPipe processed 4,820 frames on CPU, producing 159,348 landmark rows at a reported pipeline throughput of 87.86 fps [11]. These coordinates can provide temporal hand-motion features and support manual alignment to action intervals. They do not encode action semantics, and the annotation guide explicitly cautions that MediaPipe does not determine which disassembly action is occurring [7].

The VITRA run processed 322 sampled frames using per-frame reconstructed hand states. It wrote 10,304 continuous action rows across the selected left and right hands, with a 16-step prediction chunk, and reported 4.59 pipeline fps [12]. This experiment demonstrates an additional motion-prediction interface, not action classification. No reference hand trajectories or action labels were used to measure prediction quality.

### 3.4 Video action classification

All three selected classifiers completed 322 clips. Their most frequent predicted labels differed:

| Model | Most frequent label | Share of clips | Mean confidence | Pipeline throughput (clips/s) |
|---|---|---:|---:|---:|
| ActionCLIP | Prying open a part | 91.9% | 0.531 | 43.7 |
| InternVideo2 | Prying open a part | 93.5% | 0.568 | 8.79 |
| X-CLIP | Tightening a screw | 68.3% | 0.332 | 4.39 |

ActionCLIP and InternVideo2 assigned almost all clips to “prying open a part”; X-CLIP predominantly assigned “tightening a screw.” The skewed distributions are evidence that the models ran and produced outputs, but they may indicate class collapse or prompt/label ambiguity as readily as useful recognition. Confidence values are not calibrated consistently across model families and cannot be compared directly. Because no independent action ground truth was configured, classification accuracy, temporal coverage, mIoU, and Recall@IoU are unavailable [4, 6].

A separate short-fragment pilot compared X-CLIP and SmolVLM2 on a roughly seven-second clip with a researcher’s tentative “loosening a screw” reference. X-CLIP predicted “component removal,” and SmolVLM2 predicted “product positioning.” The reference was not independently confirmed, the input frame counts differed, and neither system had an “other/unknown” option. This pilot is useful as a warning about forced-choice predictions, but it is not a controlled accuracy estimate [13].

The experiment reports also record that SmolVLM2 was tested but not retained as a useful candidate for the selected action benchmark. InternVideo2.5 and InternVideo3 were not included in that benchmark because the available test computer could not run them successfully at the time. A later model comparison therefore should not treat those exclusions as measured comparative performance [13].

### 3.5 Speech-assisted annotation

In the principal speech experiment, Vosk, Whisper small, and Whisper medium each found the 14 spoken actions in a 3:24 computer-mouse disassembly recording. Vosk supplied more precise boundaries but generated two false command detections during free speech. Whisper did not produce those false commands and transcribed technical vocabulary better, but its estimated boundaries were less precise. The combination used Vosk timings and required Whisper confirmation; it retained 14/14 actions, with no false commands in the reported test, in about 1.5 minutes of processing [8, 14].

| System | Actions found | False commands during free speech | Mean start-boundary error | Mean end-boundary error | Runtime |
|---|---:|---:|---:|---:|---:|
| Vosk | 14/14 | 2 | 0.04 s | 0.02 s | 10 s |
| Whisper small | 14/14 | 0 | 0.81 s | 0.46 s | about 1.5 min |
| Whisper medium | 14/14 | 0 | 0.10 s | 1.08 s | about 14 min |
| Vosk + Whisper small | 14/14 | 0 | Vosk timing retained | Vosk timing retained | about 1.5 min |

Whisper small correctly transcribed 14 of 16 technical terms and Whisper medium 15 of 16. The reported Vosk–Whisper combination additionally associated tools and spoken descriptions with the action records and generated a human-review spreadsheet. The report describes an Excel calculation for comparing review time with manual annotation time, but explicitly states that time savings were not measured. A separate test on synthetic speech found 5/6 exact commands for Dutch Wav2Vec 2.0 and 2/6 for Meta MMS, with no false command detections in silence or free speech; however, it used a single 27.96-second synthetic recording and showed substantial end-time error [9].

These audio results are an adjacent annotation contribution, not an audio branch of the primary video benchmark. They suggest that structured speech may help produce labels and timestamps for training or review. They do not show that the video models can infer actions from the corresponding audio, or that speech annotation generalizes to noisy workshops or multiple speakers.

## 4. System design and integration

The repository implements a modular experiment harness rather than one end-to-end trained model. Detector adapters share a common interface; a shared pipeline handles frame iteration, timing, CSV output, debug video, and run summaries. ByteTrack records trajectories separately from raw detections. MediaPipe, clip classifiers, VITRA, and ASR produce task-specific outputs so that their metrics are not collapsed into a single detection score [3, 4].

This separation is methodologically important. Detection coverage, tracking identity, hand landmarks, continuous hand-motion forecasts, clip-level classification, and speech-derived annotations answer different questions. Their output formats can eventually be joined using video identity, source frame, and timestamp, but the workspace does not yet implement a validated, unified action-event dataset with linked hand, tool, component, and action annotations.

The current action taxonomy includes operations such as unscrewing, tightening, picking up or putting down a tool, prying, rotating, and idle. Speech commands add labels such as identify, fixate, tool position, disconnect, and component removal. The annotation documentation notes that some of these labels are not yet aligned between the workflows. A single agreed label vocabulary and mapping to eDIM should therefore precede dataset expansion [2, 7, 8].

Automated tests cover action schemas and prompts, action interval evaluation, benchmark summaries, CSV/JSON ground-truth loading, ASR command parsing and cancellation, and VITRA state and output handling [15]. These tests establish expected software behavior for the tested components; they are not model-performance tests.

## 5. Limitations

1. **Small and narrow video evidence.** The detailed tool and action comparisons use one top-down recording of one phone. They do not establish generalization across products, sessions, operators, tools, lighting, or camera views.
2. **No visual ground truth.** Tool detections have not been independently labeled, and action-classification accuracy and temporal localization cannot be calculated from the primary benchmark. The heuristic detection score is not a substitute.
3. **Unvalidated tracking.** Track histories exist, but identity continuity, ID switches, and tracking accuracy have not been measured against annotated tracks.
4. **Action predictions are highly concentrated.** The dominant action for two classifiers accounts for more than 90% of clips. Without labels, it is not possible to tell whether this reflects the video, class imbalance, prompt sensitivity, or model failure.
5. **Inconsistent taxonomies.** The hand-annotation guide and speech protocol differ in labels and granularity; some speech labels are marked as needing team agreement. This complicates merging annotations or training a classifier.
6. **Speech evidence is limited.** The principal recording has one speaker in a quiet room. The Wav2Vec/MMS pilot uses synthetic speech, while some timing and speed measurements were made on different machines. None establish workshop robustness or annotation time savings.
7. **Hardware-sensitive performance.** Throughput depends on GPU, model loading, frame stride, debug rendering, video encoding, and implementation. Speeds from distinct reports or hardware should not be treated as controlled head-to-head measurements.
8. **Separate modalities and outputs.** VITRA predicts continuous future hand motion, MediaPipe supplies landmarks, and the speech workflow produces human-reviewed labels. None alone provides validated action recognition.
9. **Privacy and licensing.** Disassembly recordings and speech may contain identifiable people or voices. The project documentation advises keeping recordings local and out of version control; model and MANO weights also have model-specific access or licensing conditions [3, 8].

## 6. Recommendations

1. **Establish a shared annotation protocol.** Resolve label differences, define action start/end rules, include an “unknown/uncertain” option, and document the mapping from each label to eDIM and downstream graph entities.
2. **Build a multi-video ground-truth set.** Annotate tools, tracks, actions, and temporal boundaries across different products and sessions. Use at least two reviewers on a subset and report agreement and adjudication rules. Split training, validation, and test data by video/session, not by adjacent frames.
3. **Evaluate detectors on a fixed labeled sample.** Compare per-class precision, recall, F1, and mAP on identical frames. Report false positives and missed tools, and keep the open-vocabulary prompts and closed-vocabulary label mapping explicit.
4. **Measure tracking directly.** Annotate object identities on short sequences and report ID switches and standard multi-object tracking metrics. Compare YOLO and open-vocabulary detector + ByteTrack variants using the same frame set.
5. **Re-run action classification with ground truth.** Use the existing ground-truth interface to report accuracy, per-class precision/recall/F1, confusion matrices, temporal IoU, boundary MAE, and coverage. Evaluate an unknown class or rejection mechanism to avoid misleading forced-choice outputs.
6. **Test multimodal fusion only after independent baselines.** Align video, landmarks, tool tracks, and speech events by source timestamps. First validate each modality separately, then test whether the combination improves held-out performance.
7. **Run speech trials in realistic conditions.** Include multiple speakers, accents, microphone positions, tool noise, and spontaneous non-command speech. Preserve an untouched test set and measure both annotation quality and human review time before claiming productivity gains.
8. **Publish reproducible experiment manifests.** Record model checkpoint/revision, prompt, thresholds, stride, hardware/software environment, ground-truth version, and exact run outputs. Keep licensed weights and private recordings outside source control.

## 7. Conclusion

The workspace demonstrates a practical and modular foundation for studying automatic disassembly analysis. Prompted open-vocabulary detectors identify the screwdriver in the single examined phone-repair scene more effectively than the tested COCO-label baselines; hand landmarks and tool histories provide machine-readable temporal signals; action models and VITRA execute and generate distinct types of output; and local speech recognition can turn spoken commands into editable, time-aligned annotations.

The evidence supports feasibility, not validated recognition accuracy. In particular, the project cannot yet claim that the visual system reliably recognizes human actions or tracks tools across products. The most defensible next step is not to train a larger model, but to standardize the labels and create an independently reviewed, multi-video evaluation set. That dataset would make it possible to determine whether combining tool use, hand motion, video semantics, and speech yields dependable input for DUPLO’s Disassembly Graphs, eDIM/reDIM tables, tool taxonomy, and future work instructions.

## Data and reproducibility statement

The principal video is excluded from version control, and generated outputs may be local-only. Reproduce the documented tool comparison with the scripts described at https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Modelvergelijking.md; reproduce task-aware benchmark reports using https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Benchmarking.md. Action ground truth is optional and is illustrated at https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/data/action_ground_truth.example.csv. The speech workflow and its local-data constraints are documented at https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Spraakannotatie.md. These instructions reproduce the software workflow only; they do not supply the private videos, model weights, independent ground truth, or hardware used for the reported runs.

## References

1. Project DUPLO. *Datalab III DUPLO – Human Action Recognition: Assignment and compact deliverables*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Opdracht.md
2. Project DUPLO. *Technisch plan – DUPLO Tool Recognition & Human Action Recognition*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/PLAN.md
3. Project DUPLO. *Project README and pipeline documentation*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/README.md
4. Project DUPLO. *Benchmarken per modeltaak*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Benchmarking.md
5. Project DUPLO. *Testrapport – modelvergelijking voor toolrecognition*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Modelvergelijking.md
6. Project DUPLO. *Action Classification Benchmark*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Action%20Classification%20Benchmark.docx. The generated Markdown benchmark report is local-only and excluded from version control.
7. Project DUPLO. *Handleiding: acties annoteren met MediaPipe-handlandmarks*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/MediaPipe_annotering.md
8. Project DUPLO. *Spraakannotatie*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Spraakannotatie.md
9. Project DUPLO. *Audioproeven: Wav2Vec 2.0 Nederlands en Meta MMS*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/DUPLO_audio_proeven_verslag.docx, November 2026.
10. Project DUPLO. *Tracking run summaries*. Local-only generated outputs; the tracking implementations are at https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/src/run_tracking.py and https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/src/run_sam31_tracking.py.
11. Project DUPLO. *MediaPipe handpose run summary*. Local-only generated output; the implementation is at https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/src/run_handpose.py.
12. Project DUPLO. *VITRA run summary*. Local-only generated output; the implementation is at https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/src/run_vitra_test.py.
13. Project DUPLO. *Technische modelproeven voor DUPLO: X-CLIP en SmolVLM2*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/DUPLO_modelproeven_verslag%20(1).docx, 9 November 2026.
14. Project DUPLO. *Onderzoeksverslag annoteren met spraak*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/docs/experimental/spraakannotatie/Onderzoeksverslag_annoteren_met_spraak.docx, 15 November 2026; timing data: https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/docs/experimental/spraakannotatie/samenvatting_tijden.csv and https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/docs/experimental/spraakannotatie/tijden_per_actie.csv.
15. Project DUPLO. *Automated test suite*. https://github.com/RemanufacturingLab-StudentTeams/HAR/tree/main/tests
16. Project DUPLO. *Annoteren met spraak: Onderzoek naar spraakherkenningsmodellen voor het automatisch annoteren van demontagehandelingen*. https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/research%20audio%20modellen%20Vosk%20en%20Whisper%20.pdf, 15 November 2026. This is the PDF counterpart of the DOCX report in reference 14; the findings are one experiment, not two independent studies.

Model background and original publications referenced by the workspace include:

- Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems, 33*, 12449–12460. <https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html>
- Bain, M., Huh, J., Han, T., & Zisserman, A. (2023). WhisperX: Time-accurate speech transcription of long-form audio. *Proceedings of Interspeech 2023*, 4489–4493. <https://doi.org/10.21437/Interspeech.2023-78>
- Radford, A., et al. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine Learning*. <https://proceedings.mlr.press/v202/radford23a.html>
- Model references for RF-DETR, Grounding DINO, YOLO-World, SAM 3/SAM 3.1, and Ultralytics YOLO are listed at https://github.com/RemanufacturingLab-StudentTeams/HAR/blob/main/Documentation/Modelvergelijking.md.
