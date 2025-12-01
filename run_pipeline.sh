#!/bin/bash
set -e

echo "============================"
echo "  RSNA-ICH TRIAGE PIPELINE"
echo "============================"

# Activate environment (optional, uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rsna_triage

###############################################
# 0. Create output directories if needed
###############################################
mkdir -p outputs
mkdir -p data/processed/rsna-ich/images

###############################################
# 1. Preprocessing: DICOM → HU windows → 224×224
###############################################
echo "[1/5] Preprocessing RSNA-ICH ..."
python -m src.preprocess --config configs/data.yaml

###############################################
# 2. Train slice-level classifier
###############################################
echo "[2/5] Training classifier ..."
python -m src.train --config configs/model.yaml

###############################################
# 3. Predict on val + test (uncalibrated logits)
###############################################
echo "[3/5] Generating val/test predictions ..."
python -m src.predict

###############################################
# 4. Calibrate on val, apply to test
###############################################
echo "[4/5] Calibrating probabilities ..."
python -m src.calibrate

###############################################
# 5. Run worklist simulation (Priority vs FIFO)
###############################################
echo "[5/5] Running queueing simulation ..."
python -m src.simulate --config configs/sim.yaml \
    --pred_csv outputs/predictions_test_calibrated.csv

echo "============================"
echo " Pipeline finished."
echo " Calibrated test predictions:"
echo "   outputs/predictions_test_calibrated.csv"
echo " Simulation results:"
echo "   outputs/simulation_results.csv"
echo "============================"
