import pandas as pd
import numpy as np
import itertools
import os
import warnings
from tqdm import tqdm
import math
import joblib

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# --- User Input for Process Unit ---
valid_units = ['clarifier', 'cstr']
prompt_text = f"Enter process unit type ('clarifier' or 'cstr'): "
process_unit = ''
while process_unit not in valid_units:
    process_unit = input(prompt_text).lower().strip()
    if process_unit not in valid_units:
        print(f"Invalid input. Please enter one of {valid_units}.")

# --- User Input for Solver Type ---
# 'solve': Standard exact solver (fast, can be unstable if singular)
# 'pinv': Pseudo-inverse (slower, robust to singular matrices)
# 'lstsq': Least squares (robust, finds closest solution)
valid_solvers = ['solve', 'pinv', 'lstsq']
prompt_solver = f"Enter linear solver type ('solve', 'pinv', or 'lstsq') [default: solve]: "
solver_type = input(prompt_solver).lower().strip()
if solver_type not in valid_solvers:
    print("Invalid or empty input. Defaulting to 'solve'.")
    solver_type = 'solve'

# --- User Input for Optimizer Type ---
valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adamw']
prompt_optim = f"Enter optimizer type ('adam', 'sgd', 'rmsprop', 'adamw') [default: adamw]: "
optimizer_type = input(prompt_optim).lower().strip()
if optimizer_type not in valid_optimizers:
    print("Invalid or empty input. Defaulting to 'adamw'.")
    optimizer_type = 'adamw'

# --- Configuration ---
FILE_PATH = os.path.join('data', 'data.xlsx')
MACHINE_LEARNING_MODEL = 'clefo'
OUTPUT_FILE_PATH = os.path.join('data', 'training_data', MACHINE_LEARNING_MODEL, 'training_stat_' + process_unit, process_unit + '_train_stat.xlsx')
MODEL_PATH = os.path.join('models', MACHINE_LEARNING_MODEL, process_unit, process_unit + '.joblib')
RANDOM_STATE = 42

# --- PyTorch & Model Hyperparameters ---
NUM_EPOCHS = 5000 if process_unit == 'clarifier' else 10000
LEARNING_RATE = 1e-4 # Learning rate for optimizer
BATCH_SIZE = -1 # Use -1 for full-batch training
L1_LAMBDA = 1e-3 # LASSO regularization parameter
MSE_EXPLOSION_THRESHOLD = 1000 # Threshold to detect exploding loss

# --- REVISED: Get N_SPLITS from user ---
try:
    N_SPLITS = int(input("Enter the number of folds for cross-validation (default: 10): ") or 10)
except ValueError:
    print("Invalid input. Using default value of 10.")
    N_SPLITS = 10

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch found. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch using CPU. Training will be slower.")

# --- Data Loading and Preparation ---
def load_and_prepare_data(filepath: str):
    """
    Loads and preprocesses data from the specified Excel file.
    Handles both long and wide formats for the 'all_input' sheet.
    Filters components based on the 'training_components' sheet.
    """
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
        else:
            print("   - Warning: 'training_components' sheet is missing 'components' or 'considered' column. Using all components.")
    except Exception:
        print("   - Warning: 'training_components' sheet not found. Using all available components.")

    if {'variable', 'default'}.issubset(df_input.columns):
        print("   - 'all_input' is in long format. Pivoting data...")
        df_input_wide = df_input.pivot(
            index='simulation_number', columns='variable', values='default'
        ).reset_index()
        input_cols = df_input['variable'].unique().tolist()
    else:
        print("   - 'all_input' appears to be in wide format. Skipping pivot.")
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

    print(f"Data prepared successfully.")
    print(f"  - Found {len(x_cols_ordered)} independent variables (X).")
    print(f"  - Found {len(y_cols)} dependent variables (Y).")

    return X, Y, x_cols_ordered, y_cols

# --- Feature Engineering (NumPy/Pandas based) ---
def create_interaction_features_np(X: np.ndarray, m: int):
    """Generates pairwise interaction features (Z) from independent variables (X) using NumPy."""
    col_pairs = list(itertools.combinations(range(m), 2))
    q = len(col_pairs)
    N = X.shape[0]

    Z = np.zeros((N, q), dtype=np.float64)
    for i, (col1_idx, col2_idx) in enumerate(col_pairs):
        Z[:, i] = X[:, col1_idx] * X[:, col2_idx]

    return Z

# --- PyTorch Model and Dataset Definitions ---
class WastewaterDataset(Dataset):
    """PyTorch Dataset for the wastewater model."""
    def __init__(self, X, Y, Z):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.Z = torch.tensor(Z, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.Z[idx]

class CoupledCLEFOModel(nn.Module):
    """
    Implements the coupled linear equations framework using PyTorch.
    Y = (I - Γ - diag(ΛX))⁻¹ * (Υ + BX + ΘZ)
    """
    def __init__(self, n_dep, n_indep, n_inter, solver_type='solve'):
        super().__init__()
        self.n_dep = n_dep
        self.solver_type = solver_type

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
        RHS_target = RHS.T.unsqueeze(-1)

        lambda_x = self.Lambda @ X.T
        diag_lambda_x = torch.diag_embed(lambda_x.T)

        gamma_expanded = self.Gamma.unsqueeze(0).expand(batch_size, -1, -1)
        identity_expanded = self.identity.unsqueeze(0).expand(batch_size, -1, -1)

        LHS_matrix = identity_expanded - gamma_expanded - diag_lambda_x
        # Add slight regularization to prevent singular matrix errors
        reg = torch.eye(self.n_dep, device=LHS_matrix.device).unsqueeze(0) * 1e-7
        A = LHS_matrix + reg

        # --- Solver Selection ---
        if self.solver_type == 'pinv':
            try:
                A_inv = torch.linalg.pinv(A)
                Y_pred_solved = A_inv @ RHS_target
                Y_pred = Y_pred_solved.squeeze(-1)
            except Exception:
                Y_pred = torch.full((batch_size, self.n_dep), float('nan'), device=X.device)
        
        elif self.solver_type == 'lstsq':
            try:
                solution = torch.linalg.lstsq(A, RHS_target).solution
                Y_pred = solution.squeeze(-1)
            except Exception:
                Y_pred = torch.full((batch_size, self.n_dep), float('nan'), device=X.device)

        else: # Default 'solve'
            try:
                Y_pred_solved = torch.linalg.solve(A, RHS_target)
                Y_pred = Y_pred_solved.squeeze(-1)
            except torch.linalg.LinAlgError:
                Y_pred = torch.full((batch_size, self.n_dep), float('nan'), device=X.device)

        return Y_pred

# --- Training and Prediction Functions ---
def train_model_pytorch(X_train, Y_train, Z_train):
    """
    Trains the CoupledCLEFOModel using PyTorch with LASSO regularization.
    Uses the user-selected optimizer.
    If MSE explodes, returns None to signal a restart.
    """
    n_samples, n_indep = X_train.shape
    n_dep = Y_train.shape[1]
    n_inter = Z_train.shape[1]

    model = CoupledCLEFOModel(n_dep, n_indep, n_inter, solver_type=solver_type).to(device)
    criterion = nn.MSELoss()
    
    # --- Optimizer Selection ---
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_type == 'sgd':
        # SGD often requires a higher learning rate or momentum to converge well
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE * 10, momentum=0.9) 
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    else:
        # Fallback
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = WastewaterDataset(X_train, Y_train, Z_train)
    model.train()

    if BATCH_SIZE <= 0:
        X_gpu = train_dataset.X.to(device)
        Y_gpu = train_dataset.Y.to(device)
        Z_gpu = train_dataset.Z.to(device)

        for epoch in range(NUM_EPOCHS):
            Y_pred = model(X_gpu, Z_gpu)
            l1_penalty = L1_LAMBDA * (torch.norm(model.B, 1) + torch.norm(model.Theta, 1) + torch.norm(model.Lambda, 1) + torch.norm(model.Gamma, 1))
            loss = criterion(Y_pred, Y_gpu) + l1_penalty
            
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > MSE_EXPLOSION_THRESHOLD:
                tqdm.write(f"   - Warning: MSE exploded in epoch {epoch+1} with value {loss.item():.2e}. Aborting and restarting this training run.")
                return None
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(NUM_EPOCHS):
            for X_batch, Y_batch, Z_batch in train_loader:
                X_batch, Y_batch, Z_batch = X_batch.to(device), Y_batch.to(device), Z_batch.to(device)
                Y_pred = model(X_batch, Z_batch)
                l1_penalty = L1_LAMBDA * (torch.norm(model.B, 1) + torch.norm(model.Theta, 1))
                loss = criterion(Y_pred, Y_batch) + l1_penalty
                
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > MSE_EXPLOSION_THRESHOLD:
                    tqdm.write(f"   - Warning: MSE exploded in epoch {epoch+1} with value {loss.item():.2e}. Aborting and restarting this training run.")
                    return None

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return model

def predict_pytorch(model, X_data, Z_data):
    """
    Makes predictions using the trained PyTorch model.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
        Z_tensor = torch.tensor(Z_data, dtype=torch.float32).to(device)
        Y_pred_tensor = model(X_tensor, Z_tensor)
        all_preds = Y_pred_tensor.cpu().numpy()
    return all_preds

# --- Evaluation ---
def calculate_statistics(y_true: pd.DataFrame, y_pred: np.ndarray, num_predictors: int):
    """Calculates performance metrics for the model predictions."""
    stats = []
    y_cols = y_true.columns

    for i, col_name in enumerate(y_cols):
        y_t = y_true[col_name].values
        y_p = y_pred[:, i]

        valid_indices = ~np.isnan(y_p)
        if not np.any(valid_indices):
            stats.append({'variable': col_name, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'Adj_R2': np.nan})
            continue

        y_t, y_p = y_t[valid_indices], y_p[valid_indices]
        
        if len(y_t) < 2: # Not enough data to calculate metrics
            stats.append({'variable': col_name, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'Adj_R2': np.nan})
            continue

        mse = mean_squared_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        
        n_samples = len(y_t)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - num_predictors - 1) if (n_samples - num_predictors - 1) > 0 else np.nan

        stats.append({
            'variable': col_name, 'MSE': mse, 'RMSE': math.sqrt(mse) if mse is not np.nan else np.nan, 'MAE': mean_absolute_error(y_t, y_p),
            'R2': r2, 'Adj_R2': adj_r2
        })

    return pd.DataFrame(stats)

# --- Main Execution Workflow ---
def main():
    """Main function to orchestrate the entire workflow."""
    if not os.path.exists(FILE_PATH):
        print(f"Error: The data file was not found at '{FILE_PATH}'")
        return

    X, Y, x_cols, y_cols = load_and_prepare_data(FILE_PATH)

    if X.empty or Y.empty:
        print("Error: No data left after filtering components. Check 'training_components' sheet.")
        return

    print("   - Applying natural log transformation to Y for training.")
    Y_log = np.log(Y)

    # --- N-Fold Cross-Validation ---
    print(f"\n2. Starting {N_SPLITS}-fold cross-validation...")
    print(f"   - Solver: {solver_type}")
    print(f"   - Optimizer: {optimizer_type}")
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_fold_stats = []
    m = X.shape[1]

    for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X, Y_log), total=N_SPLITS, desc="Cross-Validation")):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_log_train = Y_log.iloc[train_index]
        Y_test = Y.iloc[test_index]

        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(Y_log_train)

        X_train_s = x_scaler.transform(X_train)
        X_test_s = x_scaler.transform(X_test)
        Y_log_train_s = y_scaler.transform(Y_log_train)

        Z_train_s = create_interaction_features_np(X_train_s, m)
        Z_test_s = create_interaction_features_np(X_test_s, m)

        model_fold = None
        while model_fold is None:
            model_fold = train_model_pytorch(X_train_s, Y_log_train_s, Z_train_s)

        Y_pred_s = predict_pytorch(model_fold, X_test_s, Z_test_s)
        Y_pred_log = y_scaler.inverse_transform(Y_pred_s)
        Y_pred_test = np.exp(Y_pred_log)

        n_dep, n_indep, n_inter = Y.shape[1], X.shape[1], Z_train_s.shape[1]
        num_predictors = 1 + n_indep + n_inter + (n_dep - 1) + n_indep
        fold_stats = calculate_statistics(Y_test, Y_pred_test, num_predictors)
        all_fold_stats.append(fold_stats)

    print("\n--- Cross-Validation Complete ---")
    cv_results_df = pd.concat(all_fold_stats)
    aggregated_stats = cv_results_df.groupby('variable').agg(['mean', 'std'])
    print("Aggregated Cross-Validation Performance (Mean +/- Std Dev):")
    print(aggregated_stats)

    # --- Final Model Training on ALL Data ---
    print(f"\n3. Training final model on the entire dataset ({solver_type}, {optimizer_type})...")
    final_x_scaler = StandardScaler().fit(X)
    final_y_scaler = StandardScaler().fit(Y_log)

    X_s_full = final_x_scaler.transform(X)
    Y_log_s_full = final_y_scaler.transform(Y_log)
    Z_s_full = create_interaction_features_np(X_s_full, m)

    final_model = None
    while final_model is None:
        tqdm.write("   - Attempting to train the final model...")
        final_model = train_model_pytorch(X_s_full, Y_log_s_full, Z_s_full)

    print("   - Final model training complete.")

    # --- Saving Final Model and Results ---
    print("\n4. Saving final model, scalers, and results...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    final_model.to('cpu')
    model_bundle = {
        'model': final_model, 'x_scaler': final_x_scaler, 'y_scaler': final_y_scaler,
        'Upsilon': final_model.Upsilon.detach().numpy(), 'B': final_model.B.detach().numpy(),
        'Theta': final_model.Theta.detach().numpy(), 'Gamma': final_model.Gamma.detach().numpy(),
        'Lambda': final_model.Lambda.detach().numpy(),
        'solver_type': solver_type,
        'optimizer_type': optimizer_type
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"   - Model bundle saved to {MODEL_PATH}")

    # --- Evaluate Final Model on Full Dataset for Reporting ---
    print("\n5. Evaluating final model on full dataset for reporting...")
    Y_pred_full_s = predict_pytorch(final_model, X_s_full, Z_s_full)
    Y_pred_full_log = final_y_scaler.inverse_transform(Y_pred_full_s)
    Y_pred_full = np.exp(Y_pred_full_log)

    n_dep_full, n_indep_full, n_inter_full = Y.shape[1], X.shape[1], Z_s_full.shape[1]
    num_predictors_full = 1 + n_indep_full + n_inter_full + (n_dep_full - 1) + n_indep_full
    full_dataset_stats = calculate_statistics(Y, Y_pred_full, num_predictors_full)
    print("\n--- Validation Performance (Test Set) ---")
    print("(Note: This is the performance of the final model on the full dataset)")
    print(full_dataset_stats)

    with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='xlsxwriter') as writer:
        aggregated_stats.to_excel(writer, sheet_name='cross_val_statistics')
        pd.concat([X, Y], axis=1).to_excel(writer, sheet_name='calibration_data', index=False)
        full_dataset_stats.to_excel(writer, sheet_name='calibration_statistics', index=False)
        pd.DataFrame(Y_pred_full, columns=y_cols, index=Y.index).to_excel(writer, sheet_name='calibration_prediction', index=False)
        pd.DataFrame(np.hstack([X_s_full, Y_log_s_full]), columns=x_cols+y_cols).to_excel(writer, sheet_name='scaled_log_cal_data', index=False)
        
        pd.DataFrame().to_excel(writer, sheet_name='validation_data', index=False)
        pd.DataFrame().to_excel(writer, sheet_name='validation_statistics', index=False)
        pd.DataFrame().to_excel(writer, sheet_name='validation_prediction', index=False)
        pd.DataFrame().to_excel(writer, sheet_name='scaled_log_val_data', index=False)

    print(f"\nProcess finished successfully. Results are saved in '{OUTPUT_FILE_PATH}'.")

if __name__ == '__main__':
    main()