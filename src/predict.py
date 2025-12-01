import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

from utils import load_yaml
from models import build_model
from train import RSNADataset

def run_split(split_name, out_path):
    cfg = load_yaml("configs/model.yaml")
    base = "data/processed/rsna-ich"
    csv_path = f"{base}/{split_name}.csv"

    ds = RSNADataset(f"{base}/images", csv_path)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_model(cfg["MODEL_NAME"], cfg["NUM_CLASSES"])
    ckpt = Path(cfg["OUTPUT_DIR"]) / "checkpoints" / "model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.cuda().eval()

    ids, logits, labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            out = model(x)[:, 0]
            logits.append(out.cpu().numpy())
            labels.append(y.numpy().squeeze())
        # dataset-level IDs
    df = pd.read_csv(csv_path)
    ids = df["ID"].values

    logits = np.concatenate(logits)
    labels = np.concatenate(labels)

    out_df = pd.DataFrame({
        "ID": ids,
        "logit": logits,
        "label": labels
    })
    out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    run_split("val", "outputs/predictions_val_uncalibrated.csv")
    run_split("test", "outputs/predictions_test_uncalibrated.csv")
