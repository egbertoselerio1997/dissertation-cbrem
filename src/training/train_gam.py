# Install required libraries (uncomment if needed)
# !pip install pandas numpy scikit-learn openpyxl joblib tqdm

import pandas as pd
import numpy as np
import os
import warnings
import math
import joblib
from tqdm import tqdm

# --- Scikit-learn Imports ---
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


# --- User Input for Process Unit ---
valid_units = ['clarifier', 'cstr']
prompt_text = f"Enter process unit type ('clarifier' or 'cstr'): "
process_unit = ''
while process_unit not in valid_units:
    process_unit = input(prompt_text).lower().strip()
    if process_unit not in valid_units:
        print(f"Invalid input. Please enter one of {valid_units}.")


# --- Configuration ---
FILE_PATH = os.path.join('data', 'data.xlsx')
MACHINE_LEARNING_MODEL = 'gam'
OUTPUT_FILE_PATH = os.path.join('data', 'training_data', MACHINE_LEARNING_MODEL, 'training_stat_' + process_unit, process_unit + '_train_stat.xlsx')
MODEL_PATH = os.path.join('models', MACHINE_LEARNING_MODEL, process_unit, process_unit + '.joblib')
RANDOM_STATE = 42

# --- REVISED: Get N_SPLITS from user ---
try:
    N_SPLITS = int(input("Enter the number of folds for cross-validation (default: 10): ") or 10)
except ValueError:
    print("Invalid input. Using default value of 10.")
    N_SPLITS = 10


# --- FROM-SCRATCH GAM IMPLEMENTATION ---
class SplineBasis:
    """
    Helper class to generate a truncated power spline basis for a single feature.
    """
    def __init__(self, n_knots=10):
        self.n_knots = n_knots
        self.knots_ = None

    def fit(self, x: np.ndarray):
        """
        Finds knot locations from the data using percentiles.
        """
        if self.n_knots > 0:
            percentiles = np.linspace(0, 100, self.n_knots + 2)[1:-1]
            self.knots_ = np.percentile(x, percentiles)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms the data into the spline basis.
        Basis consists of the original feature and one function per knot.
        """
        x_reshaped = x.reshape(-1, 1)
        if self.knots_ is None or self.n_knots == 0:
            return x_reshaped # Linear feature if no knots

        # Truncated power function: max(0, x - knot)
        basis = [x_reshaped]
        for knot in self.knots_:
            basis.append(np.maximum(0, x_reshaped - knot))

        return np.hstack(basis)


class GAMRegressor(BaseEstimator, RegressorMixin):
    """
    A Generalized Additive Model (GAM) implemented from scratch using a
    backfitting algorithm and spline basis functions. Designed for a single output.
    """
    def __init__(self, n_knots=20, max_iter=100, tol=1e-5):
        self.n_knots = n_knots
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the GAM model using the backfitting algorithm.
        """
        n_samples, n_features = X.shape

        self.spline_bases_ = [SplineBasis(self.n_knots).fit(X[:, i]) for i in range(n_features)]

        self.intercept_ = np.mean(y)
        self.coeffs_ = [np.zeros(b.transform(X[:1, i]).shape[1]) for i, b in enumerate(self.spline_bases_)]
        f_values = np.zeros((n_samples, n_features))

        for it in range(self.max_iter):
            f_old_sum = f_values.sum()

            for j in range(n_features):
                partial_residuals = y - self.intercept_ - (f_values.sum(axis=1) - f_values[:, j])
                Bj = self.spline_bases_[j].transform(X[:, j])
                # Using a small regularization term (ridge) for stability
                coeffs, _, _, _ = np.linalg.lstsq(Bj, partial_residuals, rcond=None)
                self.coeffs_[j] = coeffs
                f_values[:, j] = Bj @ self.coeffs_[j]

            if np.sum(np.abs(f_values.sum() - f_old_sum)) < self.tol:
                break

        self.f_values_ = f_values
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.
        """
        y_pred = np.full(X.shape[0], self.intercept_)
        for j in range(X.shape[1]):
            Bj = self.spline_bases_[j].transform(X[:, j])
            y_pred += Bj @ self.coeffs_[j]
        return y_pred


# --- Data Loading and Preparation ---
def load_and_prepare_data(filepath: str):
    print("1. Loading and preparing data...")
    df_input = pd.read_excel(filepath, sheet_name="all_input_" + process_unit)
    df_output = pd.read_excel(filepath, sheet_name="all_output_" + process_unit)
    components_to_remove = []
    try:
        df_components = pd.read_excel(filepath, sheet_name="training_components")
        if 'components' in df_components.columns and 'considered' in df_components.columns:
            components_to_remove = df_components[df_components['considered'] == 0]['components'].tolist()
            if components_to_remove:
                print(f"   - Found 'training_components' sheet. Will remove: {components_to_remove}")
    except Exception:
        print("   - Warning: 'training_components' sheet not found.")
    if {'variable', 'default'}.issubset(df_input.columns):
        df_input_wide = df_input.pivot(index='simulation_number', columns='variable', values='default').reset_index()
        input_cols = df_input['variable'].unique().tolist()
    else:
        df_input_wide = df_input
        input_cols = [col for col in df_input.columns if col != 'simulation_number']
    data = pd.merge(df_input_wide, df_output, on='simulation_number', how='inner')
    y_cols_all = [col for col in data.columns if col.startswith('Target_')]
    if components_to_remove:
        cols_to_drop = []
        for comp in components_to_remove:
            cols_to_drop.extend([c for c in input_cols if comp in c])
            cols_to_drop.extend([c for c in y_cols_all if comp in c])
        data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        print(f"   - Removed {len(cols_to_drop)} columns corresponding to excluded components.")
    y_cols = [col for col in data.columns if col.startswith('Target_')]
    Y = data[y_cols]
    current_input_cols = [col for col in data.columns if col in input_cols]
    inf_cols = sorted([col for col in current_input_cols if col.startswith('inf_')])
    proc_cols = sorted([col for col in current_input_cols if not col.startswith('inf_')])
    x_cols_ordered = proc_cols + inf_cols
    X = data[x_cols_ordered]
    print(f"Data prepared successfully: {X.shape[1]} inputs, {Y.shape[1]} outputs.")
    return X, Y, x_cols_ordered, y_cols


# --- Evaluation ---
def calculate_statistics(y_true: pd.DataFrame, y_pred: np.ndarray, num_predictors: int):
    stats = []
    y_cols = y_true.columns
    for i, col_name in enumerate(y_cols):
        y_t = y_true[col_name].values
        y_p = y_pred[:, i]
        valid_indices = ~np.isnan(y_p)
        y_t, y_p = y_t[valid_indices], y_p[valid_indices]
        mse = mean_squared_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        adj_r2 = 1 - (1 - r2) * (len(y_t) - 1) / (len(y_t) - num_predictors - 1) if (len(y_t) - num_predictors - 1) > 0 else np.nan
        stats.append({
            'variable': col_name, 'MSE': mse, 'RMSE': math.sqrt(mse), 'MAE': mean_absolute_error(y_t, y_p),
            'R2': r2, 'Adj_R2': adj_r2
        })
    return pd.DataFrame(stats)


# --- Main Execution Workflow ---
def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: The data file was not found at '{FILE_PATH}'")
        return

    X, Y, x_cols, y_cols = load_and_prepare_data(FILE_PATH)

    if X.empty or Y.empty:
        print("Error: No data left after filtering. Check 'training_components' sheet.")
        return

    print("   - Applying natural log transformation to Y for training.")
    Y_log = np.log(Y)

    # --- N-Fold Cross-Validation ---
    print(f"\n2. Starting {N_SPLITS}-fold cross-validation...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_fold_stats = []

    for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X, Y_log), total=N_SPLITS, desc="Cross-Validation")):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_log_train, Y_log_test = Y_log.iloc[train_index], Y_log.iloc[test_index]
        Y_test = Y.iloc[test_index]

        x_scaler = StandardScaler().fit(X_train.values)
        y_scaler = StandardScaler().fit(Y_log_train.values)

        X_train_s = x_scaler.transform(X_train.values)
        X_test_s = x_scaler.transform(X_test.values)
        Y_train_s = y_scaler.transform(Y_log_train.values)

        gam_estimator = GAMRegressor(n_knots=15, max_iter=100, tol=1e-5)
        model = MultiOutputRegressor(estimator=gam_estimator, n_jobs=-1)
        model.fit(X_train_s, Y_train_s)

        Y_pred_test_s = model.predict(X_test_s)
        Y_pred_test_log = y_scaler.inverse_transform(Y_pred_test_s)
        Y_pred_test = np.exp(Y_pred_test_log)

        num_predictors = X.shape[1]
        fold_stats = calculate_statistics(Y_test, Y_pred_test, num_predictors)
        all_fold_stats.append(fold_stats)

    print("\n--- Cross-Validation Complete ---")
    cv_results_df = pd.concat(all_fold_stats)
    aggregated_stats = cv_results_df.groupby('variable').agg(['mean', 'std'])
    print("Aggregated Cross-Validation Performance (Mean +/- Std Dev):")
    print(aggregated_stats)

    # --- Final Model Training on ALL Data ---
    print("\n3. Training final model on the entire dataset...")
    final_x_scaler = StandardScaler().fit(X.values)
    final_y_scaler = StandardScaler().fit(Y_log.values)

    X_s_full = final_x_scaler.transform(X.values)
    Y_log_s_full = final_y_scaler.transform(Y_log.values)

    final_gam_estimator = GAMRegressor(n_knots=15, max_iter=100, tol=1e-5)
    final_model = MultiOutputRegressor(estimator=final_gam_estimator, n_jobs=-1)
    final_model.fit(X_s_full, Y_log_s_full)
    print("   - Final model training complete.")

    # --- Saving Final Model and Results ---
    print("\n4. Saving final model, scalers, and results...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    model_bundle = {
        'model': final_model,
        'x_scaler': final_x_scaler,
        'y_scaler': final_y_scaler
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"   - Model bundle saved to {MODEL_PATH}")

    # --- Evaluate Final Model on Full Dataset for Reporting ---
    print("\n5. Evaluating final model on full dataset for reporting...")
    Y_pred_full_s = final_model.predict(X_s_full)
    Y_pred_full_log = final_y_scaler.inverse_transform(Y_pred_full_s)
    Y_pred_full = np.exp(Y_pred_full_log)

    full_dataset_stats = calculate_statistics(Y, Y_pred_full, X.shape[1])
    print("\n--- Validation Performance (Test Set) ---")
    print("(Note: This is the performance of the final model on the full dataset)")
    print(full_dataset_stats)

    with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='xlsxwriter') as writer:
        aggregated_stats.to_excel(writer, sheet_name='cross_val_statistics')
        pd.concat([X, Y], axis=1).to_excel(writer, sheet_name='calibration_data', index=False)
        full_dataset_stats.to_excel(writer, sheet_name='calibration_statistics', index=False)
        pd.DataFrame(Y_pred_full, columns=y_cols, index=Y.index).to_excel(writer, sheet_name='calibration_prediction', index=False)
        pd.DataFrame().to_excel(writer, sheet_name='validation_data', index=False)
        pd.DataFrame().to_excel(writer, sheet_name='validation_statistics', index=False)
        pd.DataFrame().to_excel(writer, sheet_name='validation_prediction', index=False)

    print(f"\nProcess finished successfully. Results are saved in '{OUTPUT_FILE_PATH}'.")


if __name__ == '__main__':
    main()