"""
Phase 1: Power Plant Steam Turbine Optimization (var1)
- 6 features (x1-x6), target: Net Power Score (y)
- Polynomial degree up to 10
- Uses 5-fold cross-validation to select optimal degree
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================
DATA_DIR = 'IMT2024042/IMT2024042'

train = pd.read_csv(f'{DATA_DIR}/IMT2024042_train_var1.csv')
test  = pd.read_csv(f'{DATA_DIR}/IMT2024042_test_var1.csv')

print("=" * 60)
print("PHASE 1: var1 — Steam Turbine Optimization")
print("=" * 60)
print(f"\nTrain shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"\nTrain statistics:\n{train.describe()}")
print(f"\nMissing values:\n{train.isnull().sum()}")

X_train = train.drop('y', axis=1).values
y_train = train['y'].values
X_test  = test.values

print(f"\nFeatures: {list(train.columns[:-1])}")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# ============================================================
# 2. Cross-Validation to Find Optimal Degree
# ============================================================
print("\n" + "=" * 60)
print("Cross-Validation: Testing degrees 1 to 10")
print("=" * 60)

max_degree = 10
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for d in range(1, max_degree + 1):
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    n_features = X_poly.shape[1]
    
    # Use Ridge regression for higher degrees to handle potential multicollinearity
    if n_features > X_train.shape[0]:
        model = Ridge(alpha=1.0)
    else:
        model = LinearRegression()
    
    mse_scores = cross_val_score(model, X_poly, y_train, cv=kf, 
                                  scoring='neg_mean_squared_error')
    r2_scores  = cross_val_score(model, X_poly, y_train, cv=kf,
                                  scoring='r2')
    
    avg_mse = -mse_scores.mean()
    std_mse = mse_scores.std()
    avg_r2  = r2_scores.mean()
    std_r2  = r2_scores.std()
    
    cv_results.append({
        'degree': d,
        'n_features': n_features,
        'cv_mse': avg_mse,
        'cv_mse_std': std_mse,
        'cv_r2': avg_r2,
        'cv_r2_std': std_r2
    })
    
    print(f"  Degree {d:2d} | Features: {n_features:5d} | "
          f"CV MSE: {avg_mse:.6f} (±{std_mse:.6f}) | "
          f"CV R²: {avg_r2:.6f} (±{std_r2:.6f})")

cv_df = pd.DataFrame(cv_results)

# Find best degree
best_idx = cv_df['cv_mse'].idxmin()
best_degree = cv_df.loc[best_idx, 'degree']
best_mse = cv_df.loc[best_idx, 'cv_mse']
best_r2 = cv_df.loc[best_idx, 'cv_r2']

print(f"\n{'*' * 60}")
print(f"  BEST DEGREE: {best_degree}")
print(f"  CV MSE: {best_mse:.6f} | CV R²: {best_r2:.6f}")
print(f"{'*' * 60}")

# ============================================================
# 3. Plot CV Results
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.errorbar(cv_df['degree'], cv_df['cv_mse'], yerr=cv_df['cv_mse_std'],
             fmt='bo-', capsize=4, label='CV MSE')
ax1.axvline(x=best_degree, color='r', linestyle='--', alpha=0.7, 
            label=f'Best degree = {best_degree}')
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('var1: CV MSE vs Polynomial Degree')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.errorbar(cv_df['degree'], cv_df['cv_r2'], yerr=cv_df['cv_r2_std'],
             fmt='go-', capsize=4, label='CV R²')
ax2.axvline(x=best_degree, color='r', linestyle='--', alpha=0.7,
            label=f'Best degree = {best_degree}')
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('R² Score')
ax2.set_title('var1: CV R² vs Polynomial Degree')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('var1_cv_results.png', dpi=150, bbox_inches='tight')
print("\nCV plot saved to var1_cv_results.png")

# ============================================================
# 4. Train Final Model with Best Degree
# ============================================================
print(f"\nTraining final model with degree {best_degree}...")

poly_final = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly = poly_final.fit_transform(X_train)
X_test_poly  = poly_final.transform(X_test)

print(f"Polynomial features: {X_train_poly.shape[1]}")

# If features > samples, use Ridge for regularization
if X_train_poly.shape[1] > X_train_poly.shape[0]:
    # Try multiple alpha values to find the best one
    print("\nFeatures > Samples: Using Ridge regression with alpha tuning...")
    best_alpha = 1.0
    best_ridge_mse = float('inf')
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X_train_poly, y_train, cv=kf,
                                  scoring='neg_mean_squared_error')
        mse = -scores.mean()
        if mse < best_ridge_mse:
            best_ridge_mse = mse
            best_alpha = alpha
    print(f"Best Ridge alpha: {best_alpha}, CV MSE: {best_ridge_mse:.6f}")
    final_model = Ridge(alpha=best_alpha)
else:
    final_model = LinearRegression()

final_model.fit(X_train_poly, y_train)

# ============================================================
# 5. Evaluate on Training Data
# ============================================================
y_pred_train = final_model.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2  = r2_score(y_train, y_pred_train)

print(f"\n--- Final Model Training Metrics ---")
print(f"  Train MSE: {train_mse:.6f}")
print(f"  Train R²:  {train_r2:.6f}")

# ============================================================
# 6. Generate Test Predictions
# ============================================================
y_pred_test = final_model.predict(X_test_poly)

pred_df = pd.DataFrame({'y': y_pred_test})
pred_df.to_csv('IMT2024042_pred_var1.csv', index=False)

print(f"\nPredictions saved to IMT2024042_pred_var1.csv")
print(f"Prediction stats: min={y_pred_test.min():.4f}, max={y_pred_test.max():.4f}, "
      f"mean={y_pred_test.mean():.4f}")
print(f"\nFirst 5 predictions: {y_pred_test[:5]}")
print("\n✅ Phase 1 (var1) COMPLETE!")
