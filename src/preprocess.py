import os
import pydicom
import pandas as pd
import numpy as np
import cv2
from utils import load_yaml, ensure_dir
from tqdm import tqdm

def window(img, WL, WW):
    low = WL - WW // 2
    high = WL + WW // 2
    img = np.clip(img, low, high)
    return (img - low) / (high - low)

def process_dicom(path, windows):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    slope = float(ds.RescaleSlope)
    intercept = float(ds.RescaleIntercept)
    hu = slope * img + intercept
    chans = [window(hu, w["WL"], w["WW"]) for w in windows]
    return np.stack(chans, axis=-1)

def main():
    cfg = load_yaml("configs/data.yaml")
    raw = cfg["RAW_DIR"]
    out = cfg["PROCESSED_DIR"]
    ensure_dir(out)

    windows = cfg["WINDOWS"]
    labels_path = os.path.join(raw, "train.csv")
    df = pd.read_csv(labels_path)

    out_images = os.path.join(out, "images")
    ensure_dir(out_images)

    # --- image preprocessing ---
    for _, row in tqdm(df.iterrows(), total=len(df)):
        exam = row["ID"]
        dcm = os.path.join(raw, "stage_2_train", exam + ".dcm")
        img = process_dicom(dcm, windows)
        img = cv2.resize(img, (cfg["IMG_SIZE"], cfg["IMG_SIZE"]))
        np.save(os.path.join(out_images, f"{exam}.npy"), img)

    # --- train/val/test split at EXAM level ---
    from sklearn.model_selection import train_test_split

    df = df.copy()
    # Binary any-ICH label (or adapt to your exact column)
    y = df["any"].values

    df_train, df_temp = train_test_split(
        df, test_size=0.3, stratify=y, random_state=cfg["SEED"]
    )
    y_temp = df_temp["any"].values
    df_val, df_test = train_test_split(
        df_temp, test_size=2/3, stratify=y_temp, random_state=cfg["SEED"]
    )

    df_train.to_csv(os.path.join(out, "train.csv"), index=False)
    df_val.to_csv(os.path.join(out, "val.csv"), index=False)
    df_test.to_csv(os.path.join(out, "test.csv"), index=False)

    # full table too (optional)
    df.to_csv(os.path.join(out, "labels.csv"), index=False)

if __name__ == "__main__":
    main()


