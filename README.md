# Polynomial Regression for Geothermal Plant Modelling

**Machine Learning Assignment 1**  
**Roll Number:** IMT2024042

This repository contains the complete pipeline for designing, evaluating, and deploying polynomial regression models for two geothermal engineering problems:
1. **VAR1 (Power Plant Steam Turbine):** Predicting Net Power Score from 6 operational parameters.
2. **VAR2 (Thermal Reservoir Mapping):** Predicting Thermal Anomaly Score from 3D spatial coordinates.

## Approach
The models were built strictly using **Polynomial Regression** and **Ridge Regularisation ($L_2$)**. To ensure numerical stability and fair penalisation of high-degree polynomial features, a `StandardScaler` was applied during training. 

To prevent overfitting, a rigorous evaluation pipeline was implemented:
- **Leave-One-Out Cross-Validation (LOOCV):** Used via `RidgeCV` to dynamically find the optimal regularisation strength ($\alpha$) for every degree.
- **5-Fold Cross-Validation:** Used to estimate generalization error.
- **Hold-Out Validation:** 20% of the training data was permanently held out to act as an objective overfitting check.
- **Multi-Seed Stability:** CV was run across 5 different random seeds to ensure structural stability.

The selected optimal models are:
- **VAR1:** Degree 5 ($\alpha = 18.7382$)
- **VAR2:** Degree 12 ($\alpha = 1.2328$)

## Repository Structure

```text
ML_Assignment1/
├── IMT2024042/
│   └── IMT2024042/
│       ├── IMT2024042_train_var1.csv   # VAR1 Training Data
│       ├── IMT2024042_test_var1.csv    # VAR1 Test Data
│       ├── IMT2024042_train_var2.csv   # VAR2 Training Data
│       └── IMT2024042_test_var2.csv    # VAR2 Test Data
├── src/
│   ├── config.py                       # Global paths, seeds, and hyperparameters
│   ├── utils.py                        # Helpers for data loading, scaling, and plotting
│   ├── train_var1.py                   # Pipeline script for VAR1 degree sweep & training
│   ├── train_var2.py                   # Pipeline script for VAR2 degree sweep & training
│   └── polynomial_functions.py         # Pure NumPy polynomial predictions (unscaled weights)
├── outputs/
│   ├── figures/                        # Diagnostic plots (CV curves, residuals, etc.)
│   ├── fitted_var1.npz                 # Saved optimal unscaled weights for VAR1
│   └── fitted_var2.npz                 # Saved optimal unscaled weights for VAR2
├── run_all.py                          # Master script to run both training pipelines
├── inference.py                        # Standalone script to run predictions on test sets
├── IMT2024042_pred_var1.csv            # Final test predictions for VAR1
└── IMT2024042_pred_var2.csv            # Final test predictions for VAR2
```

## How to Reproduce

### Requirements
The pipeline relies on standard Python data science libraries.
```bash
pip install numpy pandas scikit-learn matplotlib
```

### 1. Training the Models (Degree Sweep & Evaluation)
To re-run the entire pipeline (including cross-validation, plotting, unscaling weights, and saving final test predictions), run:
```bash
python run_all.py
```
This will automatically execute both `src/train_var1.py` and `src/train_var2.py`. All plots and model weights will be saved to the `outputs/` folder.

### 2. Running Inference Only
If the models have already been trained (i.e., the `.npz` files exist in `outputs/`), you can directly run inference on the test datasets using:
```bash
python inference.py
```
This will load the hyperparameters, retrain on the full training set, and generate the final `_pred_var.csv` files. Alternatively, `src/polynomial_functions.py` can be imported to evaluate predictions on raw NumPy arrays using the pure mathematical polynomial formulation.
