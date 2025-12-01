# RSNA-Triage: From Classifier to Queueing Simulation for AI-Assisted Radiology Worklist Triage

This repository provides an end-to-end, reproducible pipeline for:
1. Preprocessing the public **RSNA Intracranial Hemorrhage (RSNA-ICH)** dataset  
2. Training slice-level classifiers (ResNet18, EfficientNet-B0, ViT-B/16)  
3. Calibrating exam-level predictions (temperature or isotonic)  
4. Running **discrete-event simulations** comparing FIFO vs AI-based priority  
5. Computing operational metrics: TTR, SLA10/20, ΔTTR(ICH), ΔTTR(all)  

This repository accompanies the AAAI-26 paper:
**_Translating Classifier Scores into Clinical Impact: Calibrated Risk and Queueing Simulation for AI-Assisted Radiology Worklist Triage_**

---

## 📦 Dataset

Download the RSNA-ICH dataset from Kaggle:

https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection

Expected layout:
* data/raw/rsna-ich/
* stage_2_train/
* stage_2_test/
* train.csv

---

## 🚀 Pipeline

```bash
### 1. Preprocess
python src/preprocess.py --config configs/data.yaml

### 2. Train classifier
python src/train.py --config configs/model.yaml

### 3. Calibrate predictions
python src/calibrate.py --config configs/model.yaml

### 4. Run simulation (FIFO vs Priority)
python src/simulate.py --config configs/sim.yaml
```
Outputs saved to:
* outputs/checkpoints/
* outputs/sims/
* outputs/figures/

## Citation
If you find this project useful for your research or if you use this implementation in your academic projects, please consider citing:
```
@inproceedings{
baruah2025translating,
title={Translating Classifier Scores into Clinical Impact: Calibrated Risk and Queueing Simulation for {AI}-Assisted Radiology Worklist Triage},
author={Tirthajit Baruah and Punit Rathore},
booktitle={2nd AI for Medicine and Healthcare Bridge Program at AAAI26},
year={2025},
url={https://openreview.net/forum?id=OBR8CAFu9s}
}

```
