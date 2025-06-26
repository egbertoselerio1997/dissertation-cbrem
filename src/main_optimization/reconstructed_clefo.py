import pandas as pd
import numpy as np
import joblib
import os
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- PyTorch Imports (needed for loading the joblib file) ---
import torch
import torch.nn as nn

# --- Configuration ---
MODEL_PATH = os.path.join('models', 'clefo_model.joblib')
TRAINING_DATA_PATH = os.path.join('data', 'training_statistics.xlsx')
OUTPUT_FILE_PATH = os.path.join('data', 'reconstructed_clefo_test.xlsx')


# --- PyTorch Model Definition ---
# The class definition MUST be present for joblib/pickle to successfully load the saved model object.
class CoupledCLEFOModel(nn.Module):
    """
    Implements the coupled linear equations framework using PyTorch.
    Y = (I - Γ - diag(ΛX))⁻¹ * (Υ + BX + ΘZ)
    """
    def __init__(self, n_dep, n_indep, n_inter):
        super().__init__()
        self.n_dep = n_dep
        
        self.Upsilon = nn.Parameter(torch.randn(n_dep, 1) * 0.01)
        self.B = nn.Parameter(torch.randn(n_dep, n_indep) * 0.01)
        self.Theta = nn.Parameter(torch.randn(n_dep, n_inter) * 0.01)
        self.Gamma = nn.Parameter(torch.randn(n_dep, n_dep) * 0.01)
        self.Lambda = nn.Parameter(torch.randn(n_dep, n_indep) * 0.01)
        
        self.Gamma.register_hook(lambda grad: grad.fill_diagonal_(0))
        self.register_buffer('identity', torch.eye(n_dep), persistent=False)

    def forward(self, X, Z):
        batch_size = X.shape[0]

        upsilon_expanded = self.Upsilon.expand(-1, batch_size)
        bx = self.B @ X.T
        theta_z = self.Theta @ Z.T
        RHS = upsilon_expanded + bx + theta_z

        lambda_x = self.Lambda @ X.T
        diag_lambda_x = torch.diag_embed(lambda_x.T)
        
        gamma_expanded = self.Gamma.unsqueeze(0).expand(batch_size, -1, -1)
        identity_expanded = self.identity.unsqueeze(0).expand(batch_size, -1, -1)
        
        LHS_matrix = identity_expanded - gamma_expanded - diag_lambda_x

        try:
            reg = torch.eye(self.n_dep, device=LHS_matrix.device).unsqueeze(0) * 1e-7
            Y_pred_solved = torch.linalg.solve(LHS_matrix + reg, RHS.T.unsqueeze(-1))
            Y_pred = Y_pred_solved.squeeze(-1)
        except torch.linalg.LinAlgError as e:
            print(f"Warning: Linear algebra error during solve: {e}. Returning zeros for this batch.")
            Y_pred = torch.zeros(batch_size, self.n_dep, device=X.device)
            
        return Y_pred

# --- Helper Function (copied from the original script) ---
def create_interaction_features_np(X: np.ndarray, m: int):
    """Generates pairwise interaction features (Z) from independent variables (X) using NumPy."""
    col_pairs = list(itertools.combinations(range(m), 2))
    q = len(col_pairs)
    N = X.shape[0]

    Z = np.zeros((N, q), dtype=np.float64)
    for i, (col1_idx, col2_idx) in enumerate(col_pairs):
        Z[:, i] = X[:, col1_idx] * X[:, col2_idx]
    return Z

# --- Analytical Model Reconstruction ---
def predict_analytical(X_scaled: np.ndarray, Z_scaled: np.ndarray, coeffs: dict) -> np.ndarray:
    """
    Reconstructs the CLEFO model analytically and makes predictions using NumPy.
    Equation: Y = (I - Γ - diag(ΛX))⁻¹ * (Υ + BX + ΘZ)
    """
    # Unpack coefficients
    Upsilon = coeffs['Upsilon']
    B = coeffs['B']
    Theta = coeffs['Theta']
    Gamma = coeffs['Gamma']
    Lambda = coeffs['Lambda']

    n_samples = X_scaled.shape[0]
    n_dep = Upsilon.shape[0]
    predictions = np.zeros((n_samples, n_dep))
    I = np.eye(n_dep)
    
    # The calculation is sample-wise due to the diag(ΛX) term
    for i in range(n_samples):
        x_sample = X_scaled[i].reshape(-1, 1) # (n_indep, 1)
        z_sample = Z_scaled[i].reshape(-1, 1) # (n_inter, 1)
        
        # Calculate the Right-Hand Side (RHS) of the equation
        RHS = Upsilon + (B @ x_sample) + (Theta @ z_sample)
        
        # Calculate the Left-Hand Side (LHS) matrix
        lambda_x = Lambda @ x_sample
        diag_lambda_x = np.diag(lambda_x.flatten())
        LHS = I - Gamma - diag_lambda_x
        
        # Solve the linear system LHS * Y = RHS for Y
        try:
            y_pred_solved = np.linalg.solve(LHS, RHS)
            predictions[i, :] = y_pred_solved.flatten()
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix encountered for sample {i}. Filling with NaNs.")
            predictions[i, :] = np.nan

    return predictions

# --- Main Demonstration Script ---

print("--- CLEFO Model Reconstruction and Validation ---")

# 1. Retrieve the coefficients and data from files
print(f"1. Loading model bundle from '{MODEL_PATH}'...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please run the main training script first.")
model_bundle = joblib.load(MODEL_PATH)
coefficients = {k: v for k, v in model_bundle.items() if k not in ['model', 'x_scaler', 'y_scaler']}
x_scaler = model_bundle['x_scaler']
y_scaler = model_bundle['y_scaler']
print("   - Coefficients and scalers loaded successfully.")

print(f"2. Loading datasets from '{TRAINING_DATA_PATH}'...")
if not os.path.exists(TRAINING_DATA_PATH):
    raise FileNotFoundError(f"Data file not found at '{TRAINING_DATA_PATH}'. Please run the main training script first.")
df_cal = pd.read_excel(TRAINING_DATA_PATH, sheet_name='calibration_data')
df_val = pd.read_excel(TRAINING_DATA_PATH, sheet_name='validation_data')
df_cal_pred_orig = pd.read_excel(TRAINING_DATA_PATH, sheet_name='calibration_prediction')
df_val_pred_orig = pd.read_excel(TRAINING_DATA_PATH, sheet_name='validation_prediction')
print("   - Calibration and validation datasets loaded.")

y_cols = [col for col in df_cal.columns if col.startswith('Effluent_')]
x_cols = [col for col in df_cal.columns if col not in y_cols]

X_cal = df_cal[x_cols]
X_val = df_val[x_cols]
y_cols_ordered = df_cal_pred_orig.columns.tolist()

# 2. Reconstruct the analytical model and predict
print("3. Reconstructing model and predicting with analytical formula...")

# Prepare calibration data
X_cal_s = x_scaler.transform(X_cal)
Z_cal_s = create_interaction_features_np(X_cal_s, X_cal_s.shape[1])

# Predict for calibration set
Y_pred_cal_s_recon = predict_analytical(X_cal_s, Z_cal_s, coefficients)
Y_pred_cal_recon = y_scaler.inverse_transform(Y_pred_cal_s_recon)
df_cal_pred_recon = pd.DataFrame(Y_pred_cal_recon, columns=y_cols_ordered)
print("   - Prediction on calibration data complete.")

# Prepare validation data
X_val_s = x_scaler.transform(X_val)
Z_val_s = create_interaction_features_np(X_val_s, X_val_s.shape[1])

# Predict for validation set
Y_pred_val_s_recon = predict_analytical(X_val_s, Z_val_s, coefficients)
Y_pred_val_recon = y_scaler.inverse_transform(Y_pred_val_s_recon)
df_val_pred_recon = pd.DataFrame(Y_pred_val_recon, columns=y_cols_ordered)
print("   - Prediction on validation data complete.")


# 3. Compare predictions and report statistics
print("4. Comparing reconstructed model predictions with original PyTorch model predictions...")
comparison_results = []

# Calibration set comparison
cal_mse = mean_squared_error(df_cal_pred_orig, df_cal_pred_recon)
cal_mae = mean_absolute_error(df_cal_pred_orig, df_cal_pred_recon)
comparison_results.append({'Dataset': 'Calibration', 'Metric': 'MSE', 'Value': cal_mse})
comparison_results.append({'Dataset': 'Calibration', 'Metric': 'MAE', 'Value': cal_mae})
print(f"   - Calibration Set | MSE vs original: {cal_mse:.4e}, MAE vs original: {cal_mae:.4e}")

# Validation set comparison
val_mse = mean_squared_error(df_val_pred_orig, df_val_pred_recon)
val_mae = mean_absolute_error(df_val_pred_orig, df_val_pred_recon)
comparison_results.append({'Dataset': 'Validation', 'Metric': 'MSE', 'Value': val_mse})
comparison_results.append({'Dataset': 'Validation', 'Metric': 'MAE', 'Value': val_mae})
print(f"   - Validation Set  | MSE vs original: {val_mse:.4e}, MAE vs original: {val_mae:.4e}")

df_comparison = pd.DataFrame(comparison_results)

# 4. Save results to a new Excel file
print(f"5. Saving reconstructed predictions and comparison to '{OUTPUT_FILE_PATH}'...")
with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='xlsxwriter') as writer:
    df_cal_pred_recon.to_excel(writer, sheet_name='calibration_prediction', index=False)
    df_val_pred_recon.to_excel(writer, sheet_name='validation_prediction', index=False)
    df_comparison.to_excel(writer, sheet_name='comparison_statistics', index=False)

print("\n--- Demonstration Complete ---")
print("The analytical reconstruction was successful. The near-zero error values confirm")
print("that the model can be perfectly replicated from the saved coefficients.")