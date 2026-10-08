import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.utils import (
    load_data, generate_polynomial_features, evaluate_ridge,
    evaluate_holdout, unscale_coefficients,
    save_predictions, plot_degree_vs_metric, plot_residuals, plot_actual_vs_predicted
)


def main():
    print("Running VAR2 pipeline (Thermal Reservoir)...")

    # 1. load data
    X_train, y_train, X_test, _ = load_data(TRAIN_VAR2, TEST_VAR2)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. hold-out split — reserves 20 % of training data that the model
    #    never sees during degree / alpha selection, so we can detect overfitting
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X_train, y_train, test_size=HOLDOUT_RATIO, random_state=RANDOM_SEED
    )
    print(f"Dev shape: {X_dev.shape}, Hold-out shape: {X_hold.shape}")

    # 3. scan polynomial degrees with Ridge + StandardScaler
    print("\nScanning degrees 1-20 with Ridge + StandardScaler:")
    best_deg = None
    best_mse = float('inf')
    best_alpha = None
    degs, mses, r2s, hold_mses, hold_r2s = [], [], [], [], []

    for deg in VAR2_DEGREES:
        X_dev_s, X_hold_s, poly, scaler = generate_polynomial_features(
            X_dev, deg, X_hold
        )
        alpha, mse, r2 = evaluate_ridge(X_dev_s, y_dev, RIDGE_ALPHAS,
                                         N_FOLDS, RANDOM_SEED)
        h_mse, h_r2 = evaluate_holdout(X_dev_s, y_dev, X_hold_s, y_hold, alpha)

        degs.append(deg)
        mses.append(mse)
        r2s.append(r2)
        hold_mses.append(h_mse)
        hold_r2s.append(h_r2)

        print(f"  deg {deg:2d} ({X_dev_s.shape[1]:4d} feat) -> "
              f"CV MSE: {mse:.4f}  Hold MSE: {h_mse:.4f}  "
              f"CV R2: {r2:.4f}  Hold R2: {h_r2:.4f}  alpha: {alpha:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_deg = deg
            best_alpha = alpha

    print(f"\nSelected degree {best_deg} "
          f"(CV MSE: {best_mse:.4f}, alpha: {best_alpha:.4f})")

    # overfitting check: hold-out MSE should be close to CV MSE
    idx = degs.index(best_deg)
    ratio = hold_mses[idx] / mses[idx]
    print(f"Hold-out / CV MSE ratio: {ratio:.3f}  "
          f"({'OK - good generalisation' if 0.7 <= ratio <= 1.4 else 'WARNING: potential overfit'})")

    # 4. stability across random seeds
    X_dev_s, _, _, _ = generate_polynomial_features(X_dev, best_deg)
    seed_scores = []
    for s in CV_SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        score = -cross_val_score(Ridge(alpha=best_alpha), X_dev_s, y_dev,
                                  cv=kf, scoring='neg_mean_squared_error').mean()
        seed_scores.append(score)
    print(f"5-seed stability: MSE = {np.mean(seed_scores):.4f} "
          f"+/- {np.std(seed_scores):.4f}")

    # 5. final model: retrain on ALL training data with best degree + alpha
    X_full_s, X_test_s, poly_final, scaler_final = \
        generate_polynomial_features(X_train, best_deg, X_test)

    model = Ridge(alpha=best_alpha)
    model.fit(X_full_s, y_train)

    train_preds = model.predict(X_full_s)
    print(f"Train MSE: {mean_squared_error(y_train, train_preds):.4f}, "
          f"Train R2: {r2_score(y_train, train_preds):.4f}")

    # 6. generate test predictions
    test_preds = model.predict(X_test_s)
    save_predictions(test_preds, PRED_VAR2)
    print(f"Saved predictions to {PRED_VAR2}")

    # 7. save unscaled polynomial coefficients
    intercept, coefs = unscale_coefficients(model, scaler_final)
    powers = poly_final.powers_
    npz_path = os.path.join(OUTPUT_DIR, "fitted_var2.npz")
    np.savez(npz_path, intercept=intercept, coefs=coefs, powers=powers,
             degree=best_deg, alpha=best_alpha)
    print(f"Saved polynomial coefficients to {npz_path}")

    # 8. diagnostic plots
    plot_degree_vs_metric(degs, mses, r2s, best_deg, 'VAR2',
                          os.path.join(FIGURE_DIR, 'var2_degree_metric.png'),
                          hold_mses=hold_mses, hold_r2s=hold_r2s)
    plot_residuals(y_train, train_preds, 'VAR2',
                   os.path.join(FIGURE_DIR, 'var2_residuals.png'))
    plot_actual_vs_predicted(y_train, train_preds, 'VAR2',
                             os.path.join(FIGURE_DIR, 'var2_actual_vs_pred.png'))
    print("Saved plots to outputs/figures/")
    print("Done VAR2.")

    return {
        'degree': best_deg, 'alpha': best_alpha,
        'intercept': intercept, 'coefs': coefs, 'powers': powers
    }


if __name__ == "__main__":
    main()
