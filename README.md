# RSNA-Triage: From Classifier to Queueing Simulation for AI-Assisted Radiology Worklist Triage

This repository provides a **fully reproducible, end-to-end pipeline** for converting head CT classifier outputs into **operational workflow metrics** for radiology worklists.  
It implements the pipeline described in our AAAI-26 publication:

> **Translating Classifier Scores into Clinical Impact:  
> Calibrated Risk and Queueing Simulation for AI-Assisted Radiology Worklist Triage**

The codebase supports:
- **Preprocessing** of RSNA-ICH DICOMs (HU conversion + clinical windows)  
- **Training** slice-level classifiers (ResNet18, EfficientNet-B0, ViT-B/16)  
- **Exam-level aggregation** (max/mean pooling)  
- **Post-hoc calibration** (Temperature, Isotonic)  
- **Discrete-event simulation** of FIFO vs. AI-driven Priority queues  
- **Operational metrics:**  
  - Time-to-read (TTR)  
  - SLA-10 / SLA-20  
  - ΔTTR(ICH), ΔTTR(all)  
  - Workload–AUC heatmaps and staffing sensitivity  

This repository is designed to be **plug-and-play**. Edit the paths in the config files and run `bash run_pipeline.sh`.

---

## 📦 Dataset: RSNA Intracranial Hemorrhage (RSNA-ICH)

Download from Kaggle:  
https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection

Expected layout:
```bash
data/
└── raw/
└── rsna-ich/
├── stage_2_train/
├── train.csv
└── ... (DICOMs)
```

We follow standard clinical preprocessing:
- Convert pixel values to **Hounsfield Units (HU)**
- Apply three clinical window settings (brain, subdural, bone)
- Stack as pseudo-RGB and resize to **224×224**

---

## 🚀 Pipeline Overview

Run the full workflow with a single command:

```bash
bash run_pipeline.sh
```
This performs:
- Preprocessing (DICOM → HU windows → images)
- Training a slice-level classifier
- Generating exam-level predictions
- Calibration (val → calibrated test probabilities)
- Worklist simulation (FIFO vs Priority)

## 🗂️ Directory Structure
```bash
RSNA-Triage/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── sim.yaml
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── calibrate.py
│   └── simulate.py
├── run_pipeline.sh
├── README.md
└── outputs/        # populated after running the pipeline
```

## ⚙️ Running Individual Steps
1. Preprocess DICOMs
```bash
python -m src.preprocess --config configs/data.yaml
```
2. Train a classifier
```bash
python -m src.train --config configs/model.yaml
```
3. Generate predictions
```bash
python -m src.predict
```
4. Calibrate exam-level probabilities
```bash
python -m src.calibrate
```
5. Run FIFO vs Priority queue simulation
```bash
python -m src.simulate --config configs/sim.yaml \
    --pred_csv outputs/predictions_test_calibrated.csv
```

Outputs:
* outputs/predictions_*
* outputs/simulation_results.csv
* outputs/figures/

## 🔬 Reproducibility
* All splits are exam-level (no patient leakage)
* YAML configs control every step
* Seed control ensures deterministic runs
* Simulation is repeated over 100 independent sessions to estimate variance

## 📄 Citation
If this repository helps your research, please cite:
```bash
@inproceedings{
baruah2025translating,
title={Translating Classifier Scores into Clinical Impact: Calibrated Risk and Queueing Simulation for {AI}-Assisted Radiology Worklist Triage},
author={Tirthajit Baruah and Punit Rathore},
booktitle={2nd AI for Medicine and Healthcare Bridge Program at AAAI26},
year={2025},
url={https://openreview.net/forum?id=OBR8CAFu9s}
}
```

## 📬 Contact
If you have any questions, need clarification, or would like to collaborate, please don't hesitate to reach out.

This repository aims to provide a transparent, reproducible, and deployment-focused framework for evaluating AI-based triage policies in radiology workflows.
