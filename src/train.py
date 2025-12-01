import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from utils import load_yaml, ensure_dir, set_seed
from models import build_model
from tqdm import tqdm

class RSNADataset(Dataset):
    def __init__(self, img_dir, csv, transform=None):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(Path(self.img_dir) / f"{row['ID']}.npy")
        x = torch.tensor(x).permute(2,0,1).float()
        y = torch.tensor(row[["any"]].values, dtype=torch.float32)
        return x, y

def main():
    cfg = load_yaml("configs/model.yaml")
    set_seed(cfg["SEED"])

    base = "data/processed/rsna-ich"
    train_csv = f"{base}/train.csv"

    train_ds = RSNADataset(f"{base}/images", train_csv)
    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"],
                              shuffle=True, num_workers=2)

    model = build_model(cfg["MODEL_NAME"], cfg["NUM_CLASSES"]).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"],
                           weight_decay=cfg["WEIGHT_DECAY"])
    loss_fn = nn.BCEWithLogitsLoss()

    ckpt_dir = cfg["OUTPUT_DIR"] + "/checkpoints"
    ensure_dir(ckpt_dir)

    for epoch in range(cfg["EPOCHS"]):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['EPOCHS']}"):
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            logits = model(x)[:, 0]
            loss = loss_fn(logits, y.squeeze())
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), f"{ckpt_dir}/model.pt")


if __name__ == "__main__":
    main()

