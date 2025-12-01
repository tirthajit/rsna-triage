import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def main():
    # fit on val
    df_val = pd.read_csv("outputs/predictions_val_uncalibrated.csv")
    y_val = df_val["label"].values
    z_val = df_val["logit"].values  # raw logits or probabilities

    iso = IsotonicRegression(out_of_bounds="clip")
    p_val = iso.fit_transform(z_val, y_val)

    df_val["prob_cal"] = p_val
    df_val.to_csv("outputs/predictions_val_calibrated.csv", index=False)

    # apply to test
    df_test = pd.read_csv("outputs/predictions_test_uncalibrated.csv")
    z_test = df_test["logit"].values
    p_test = iso.predict(z_test)

    df_test["prob_cal"] = p_test
    df_test.to_csv("outputs/predictions_test_calibrated.csv", index=False)

if __name__ == "__main__":
    main()

