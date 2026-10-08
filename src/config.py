import os
import numpy as np

# file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "IMT2024042", "IMT2024042")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

TRAIN_VAR1 = os.path.join(DATA_DIR, "IMT2024042_train_var1.csv")
TEST_VAR1 = os.path.join(DATA_DIR, "IMT2024042_test_var1.csv")
TRAIN_VAR2 = os.path.join(DATA_DIR, "IMT2024042_train_var2.csv")
TEST_VAR2 = os.path.join(DATA_DIR, "IMT2024042_test_var2.csv")

PRED_VAR1 = os.path.join(OUTPUT_DIR, "IMT2024042_pred_var1.csv")
PRED_VAR2 = os.path.join(OUTPUT_DIR, "IMT2024042_pred_var2.csv")

# random seed for reproducibility
RANDOM_SEED = 42
CV_SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
HOLDOUT_RATIO = 0.2  # fraction of training data reserved for overfitting check

# candidate degrees for each problem
VAR1_DEGREES = list(range(1, 11))
VAR2_DEGREES = list(range(1, 21))

# ridge regularization alphas
RIDGE_ALPHAS = np.logspace(-4, 5, 100)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
