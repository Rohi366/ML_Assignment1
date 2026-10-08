import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')


def load_data(train_path, test_path):
    """Load train and test CSV files, returning arrays and feature names."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('y', axis=1).values
    y_train = train_df['y'].values
    X_test = test_df.values
    feature_names = [col for col in train_df.columns if col != 'y']

    return X_train, y_train, X_test, feature_names


def generate_polynomial_features(X_train, degree, X_other=None):
    """
    Generate polynomial features and normalise them with StandardScaler.

    Scaling ensures that Ridge regularisation penalises all polynomial terms
    uniformly, regardless of their magnitude.  Without scaling, high-degree
    monomials can have very different scales, causing Ridge to under- or
    over-regularise individual terms.

    Parameters
    ----------
    X_train : array of shape (n_train, n_features)
    degree  : int, polynomial degree
    X_other : optional array to transform with the *same* poly + scaler

    Returns
    -------
    X_train_scaled, X_other_scaled, poly, scaler
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)

    X_other_scaled = None
    if X_other is not None:
        X_other_scaled = scaler.transform(poly.transform(X_other))

    return X_train_scaled, X_other_scaled, poly, scaler


def evaluate_ridge(X_scaled, y, alphas, n_folds=5, seed=42):
    """
    Two-stage alpha selection:
      1. RidgeCV with efficient LOO (cv=None) to find the best alpha
      2. 5-fold CV to obtain an unbiased MSE / R2 estimate at that alpha
    """
    rcv = RidgeCV(alphas=alphas, cv=None).fit(X_scaled, y)
    best_alpha = rcv.alpha_
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    mse = -cross_val_score(Ridge(alpha=best_alpha), X_scaled, y, cv=kf,
                           scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(Ridge(alpha=best_alpha), X_scaled, y, cv=kf,
                         scoring='r2').mean()
    return best_alpha, mse, r2


def evaluate_holdout(X_train_scaled, y_train, X_hold_scaled, y_hold, alpha):
    """Fit Ridge on dev data and score on a held-out set (overfitting check)."""
    model = Ridge(alpha=alpha).fit(X_train_scaled, y_train)
    preds = model.predict(X_hold_scaled)
    mse = mean_squared_error(y_hold, preds)
    r2 = r2_score(y_hold, preds)
    return mse, r2


def unscale_coefficients(model, scaler):
    """
    Map Ridge coefficients from the scaled feature space back to the original
    (unscaled) polynomial feature space.

    If  y = b_s + w_s @ Z_scaled   and   Z_scaled = (Z - mu) / sigma,
    then  y = (b_s - w_s . (mu/sigma))  +  (w_s / sigma) @ Z

    This allows polynomial_functions.py to evaluate predictions on raw
    monomial terms without needing the scaler at inference time.
    """
    coefs = model.coef_ / scaler.scale_
    intercept = model.intercept_ - np.dot(model.coef_,
                                          scaler.mean_ / scaler.scale_)
    return intercept, coefs


def save_predictions(predictions, output_path):
    """Save predictions to CSV with column 'y'."""
    df = pd.DataFrame({'y': predictions})
    df.to_csv(output_path, index=False)


def plot_degree_vs_metric(degrees, mses, r2s, best_deg, title, save_path,
                          hold_mses=None, hold_r2s=None):
    """Plot CV MSE and R2 vs polynomial degree, with optional hold-out overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(degrees, mses, 'o-', color='#1976D2', lw=2, ms=6, label='CV MSE')
    if hold_mses is not None:
        axes[0].plot(degrees, hold_mses, 's--', color='#E65100', lw=2, ms=6,
                     label='Hold-out MSE')
    axes[0].axvline(x=best_deg, color='red', ls='--', alpha=0.7,
                    label=f'Best deg={best_deg}')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title(f'{title} - MSE vs Degree')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(degrees, r2s, 's-', color='#388E3C', lw=2, ms=6, label='CV R2')
    if hold_r2s is not None:
        axes[1].plot(degrees, hold_r2s, 's--', color='#E65100', lw=2, ms=6,
                     label='Hold-out R2')
    axes[1].axvline(x=best_deg, color='red', ls='--', alpha=0.7,
                    label=f'Best deg={best_deg}')
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('R2 Score')
    axes[1].set_title(f'{title} - R2 vs Degree')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, title, save_path):
    """Plot residuals vs fitted values and residual histogram."""
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(y_pred, res, alpha=0.4, s=15, c='steelblue')
    axes[0].axhline(y=0, color='red', ls='--')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{title} - Residuals vs Fitted')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(res, bins=35, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0, color='red', ls='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'{title} - Residual Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, title, save_path):
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, c='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('Actual y')
    ax.set_ylabel('Predicted y')
    ax.set_title(f'{title} - Actual vs Predicted')

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    ax.text(0.05, 0.95, f'R2 = {r2:.4f}\nMSE = {mse:.4f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
