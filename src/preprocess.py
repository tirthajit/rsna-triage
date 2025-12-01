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
    df = pd.read_csv(os.path.join(raw, "train.csv"))

    out_images = os.path.join(out, "images")
    ensure_dir(out_images)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        exam = row["ID"]
        dcm = os.path.join(raw, "stage_2_train", exam + ".dcm")
        img = process_dicom(dcm, windows)
        img = cv2.resize(img, (cfg["IMG_SIZE"], cfg["IMG_SIZE"]))
        np.save(os.path.join(out_images, f"{exam}.npy"), img)

    df.to_csv(os.path.join(out, "labels.csv"), index=False)

if __name__ == "__main__":
    main()
