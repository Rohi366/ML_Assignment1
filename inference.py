import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import (TRAIN_VAR1, TEST_VAR1, TRAIN_VAR2, TEST_VAR2,
                         PRED_VAR1, PRED_VAR2, OUTPUT_DIR)
from src.utils import load_data, save_predictions


def infer_var1(test_path=TEST_VAR1, output_path=PRED_VAR1):
    """Retrain optimal VAR1 model (with scaling) on full training data and predict."""
    print("Running inference for VAR1 (Power Plant)...")
    X_train, y_train, X_test, _ = load_data(TRAIN_VAR1, test_path)

    # Load optimal hyper-parameters from training run
    npz_path = os.path.join(OUTPUT_DIR, "fitted_var1.npz")
    if os.path.exists(npz_path):
        saved = np.load(npz_path)
        degree = int(saved['degree'])
        alpha = float(saved['alpha'])
        print(f"  Loaded params: degree={degree}, alpha={alpha:.4f}")
    else:
        degree, alpha = 5, 23.1013
        print(f"  Using fallback params: degree={degree}, alpha={alpha:.4f}")

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(poly.transform(X_test))

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    save_predictions(preds, output_path)
    print(f"VAR1 predictions saved to {output_path} ({len(preds)} samples)")
    return preds


def infer_var2(test_path=TEST_VAR2, output_path=PRED_VAR2):
    """Retrain optimal VAR2 model (with scaling) on full training data and predict."""
    print("Running inference for VAR2 (Thermal Reservoir)...")
    X_train, y_train, X_test, _ = load_data(TRAIN_VAR2, test_path)

    # Load optimal hyper-parameters from training run
    npz_path = os.path.join(OUTPUT_DIR, "fitted_var2.npz")
    if os.path.exists(npz_path):
        saved = np.load(npz_path)
        degree = int(saved['degree'])
        alpha = float(saved['alpha'])
        print(f"  Loaded params: degree={degree}, alpha={alpha:.4f}")
    else:
        degree, alpha = 12, 1.2328
        print(f"  Using fallback params: degree={degree}, alpha={alpha:.4f}")

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(poly.transform(X_test))

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    save_predictions(preds, output_path)
    print(f"VAR2 predictions saved to {output_path} ({len(preds)} samples)")
    return preds


def main():
    print("Generating inferences for IMT2024042...\n")
    infer_var1()
    print("-" * 40)
    infer_var2()
    print("\nInference complete.")


if __name__ == "__main__":
    main()
