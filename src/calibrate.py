import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import load_yaml
from models import build_model
from sklearn.isotonic import IsotonicRegression

def main():
    cfg = load_yaml("configs/model.yaml")

    df = pd.read_csv("outputs/predictions_uncalibrated.csv")
    y = df["label"].values
    p = df["logit"].values

    iso = IsotonicRegression(out_of_bounds="clip")
    p_cal = iso.fit_transform(p, y)

    out = df.copy()
    out["prob"] = p_cal
    out.to_csv("outputs/predictions_calibrated.csv", index=False)

if __name__ == "__main__":
    main()
