# Datalab III DUPLO – Human Action Recognition

## Organisation

| | |
| --- | --- |
| **Name of organisation** | Lectoraat Smart Sustainable Manufacturing van De Haagse Hogeschool |
| **Contact persons** | Pim de Jong, Abboy Verkuilen, Mirjam Zijderveld |
| **E-mail contact persons** | <P.deJong-3@hbs.nl>, <A.Verkuilen@hbs.nl>, <M.J.W.Zijderveld@hbs.nl> |
| **Phone contact persons** | 06 19435556, 06 24531050, 06 22050281 |

## Case description

| | |
|---|---|
| **Type of project** | Deep learning: computer vision of NLP |

### Background

The DUPLO (De-assemblage: Uitvoering – Planning – Organisatie) project investigates how disassembly processes of technical products can be better understood, modelled and optimised. This is relevant for circular applications such as repair, remanufacturing and recycling. Consumer electronics and other technical products contain valuable components and materials, but effective recovery requires insight into which parts are present, how they are connected, which tools are required and which disassembly actions are needed.

Within DUPLO, previous disassembly workshops have generated data on components, connectors, tool requirements, and disassembly sequences among others. This information is stored in a Disassembly Graphs, this is a hypergraph-based model used to store relevant data on the topology of specific products. We also use this data to create eDIM/reDIM (Ease of Disassembly Metric) tables and the development of work instructions. This is all done manually, so a logical next step is to explore whether disassembly processes can be analysed automatically. On top of object recognition, we would like to introduce human action recognition to better align eDIM logics without hyper graph approach and to be able to create complete work instructions (both for humans and for cobots). We will provide you with a pre-published paper by Zijderveld, Fraanje, and Coenen, outlining the conceptual disassembly model.

In this Key Enabling Technology project, students will explore how Human Action Recognition, object detection, tool recognition and connector recognition can contribute to automatically recognising and labelling disassembly actions. Examples include recognising actions such as unscrewing, picking up or changing tools, removing components, disconnecting cables or connectors and positioning the product. Students may use models and tools such as YOLO variants, pose estimation, object detection, video classification, temporal action detection or other relevant approaches they identify themselves.

The project is intentionally experimental: students use new technology to explore what is currently possible and where the limitations are. The aim is not to build a physical prototype, but to develop and evaluate a proof of concept for automatic or semi-automatic video analysis of disassembly processes.

### Problem Definition

The research group has disassembly knowledge and video material from the DUPLO project, but much of the information from disassembly processes still needs to be interpreted and structured manually. For further use in Disassembly Graphs, eDIM/redIM tables and future work instructions, it would be valuable if relevant actions, tools, objects and connector interactions could be recognised automatically from video footage.

## Assignment

1. **Exploration of techniques**  
   Conduct a short literature and tool review of suitable methods for Human Action Recognition, object detection, tool recognition and connector recognition.

2. **Definition of actions and labels**  
   Expand the available annotation scheme for a limited set of disassembly actions, for example:
   - unscrewing;
   - picking up or changing tools;
   - removing a component;
   - disconnecting a cable or connector;
   - positioning the product.

   Where possible, link these labels to eDIM categories such as identifying, manipulation, positioning, disconnection, removing and tool change.

3. **Dataset creation and annotation**  
   Use available DUPLO video footage and record a small number of additional videos of disassembly actions on consumer electronics. Take privacy into account: videos containing recognisable persons or faces should not be uploaded to external cloud environments without appropriate safeguards.

4. **Experiment with model or pipeline**  
   Build or configure a proof-of-concept pipeline for recognising actions, objects, tools and/or connector interactions. Students are free to choose their models and tools, provided they justify their choices.

5. **Evaluation and reflection**  
   Evaluate the approach using suitable metrics, to be determined by the students based on the chosen method and relevant literature. Examples include precision, recall, F1-score, mAP, confusion matrix, temporal segmentation quality or processing time. If the dataset is too small for robust quantitative validation, a qualitative demo and critical reflection are acceptable.

6. **Demo and poster presentation**  
   Present the result as a demo during the DUPLO symposium on 15 October. The poster must be in English. The final poster format will be provided later. Students do not need to deliver a fully mature model. The emphasis is on experimentation, exploration, documentation and demonstrating what is currently feasible with a compact proof of concept.

### Compact Deliverables

- Short technical exploration of relevant Human Action Recognition and computer vision methods.
- Annotation scheme for disassembly actions, linked where possible to eDIM categories.
- Small labelled dataset or reproducible annotation/labeling approach.
- Proof-of-concept pipeline/model for video analysis.
- Short evaluation of results, limitations and recommendations for further work within DUPLO.
- Code repository with instructions to reproduce the experiments.
- English demo poster for the DUPLO symposium.

## Intended Connection to DUPLO Outputs

The students do not need to fully deliver the products below, but their output should be usable or translatable towards:

- **Disassembly Graph input:** recognised actions, components, tools and sequences can later be converted into nodes, relationships and process steps.
- **eDIM/reDIM tables:** recognised actions and, if available, measured durations can support the completion or validation of eDIM and reDIM tables.
- **Tool taxonomy:** detected tools and tool changes can be linked to the tool categories used within the DUPLO project.
- **Future work instructions:** automatically recognised process steps can later support the creation of clearer and more evidence-based disassembly instructions.
- **Process optimisation:** recognised sequences and bottlenecks can provide input for later analysis of the best business cases for repair, recycling and remanufacturing.

## Dataset and Practical Context

The project may use existing DUPLO video recordings, supplemented by a small number of new student-made recordings. Available video material may include top-down and front-down camera angles. Audio is not required, but may be explored as an additional signal if available and appropriate.

The available videos may sometimes contain recognisable persons. Such footage should be handled carefully and should not be uploaded to external platforms without appropriate privacy measures. Students are expected to work responsibly with data, but a full ethical analysis is not required within this short project.

The product focus is diverse consumer electronics. The exact products and available video material will be specified when the project starts.

## Technical Scope

Students are free to choose suitable technology. Examples include, but are not limited to:

- YOLO variants for object, tool or action-related detection;
- pose estimation models;
- video classification models;
- temporal action detection methods;
- OpenCV-based pipelines;
- Python-based computer vision workflows;
- annotation tools such as CVAT, Label Studio or similar tools.

Open-source tools are a logical starting point but are not mandatory. Cloud tools may be used where appropriate, but students must take privacy and budget constraints into account. Real-time analysis is not required; offline video analysis is sufficient. Real-time application may become relevant in later phases of the DUPLO project.

## Expected Student Profile

This project is suitable for students of the Master Next Level Engineering who are interested in:

- artificial intelligence and computer vision;
- circular manufacturing and disassembly processes;
- experimental use of new Key Enabling Technologies;
- translating technical experiments into usable input for applied research.

## Success Criteria

The project is successful if students can demonstrate:

- a clear and compact label structure for disassembly actions;
- a reasoned selection of one or more suitable Human Action Recognition/computer vision methods;
- a working proof-of-concept pipeline on a small dataset;
- a clear connection between recognised actions and DUPLO information needs such as eDIM, Disassembly Graphs and tool taxonomy;
- an honest evaluation of what works, what does not work and what should be explored next;
- a clear English poster and demo for the DUPLO symposium.
