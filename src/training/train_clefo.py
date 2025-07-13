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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
process_unit = str(input("Enter process unit type ('clarifier' or 'cstr'): ")).lower().strip()

# --- Configuration ---
FILE_PATH = os.path.join('data', 'data.xlsx')
OUTPUT_FILE_PATH = os.path.join('data', 'training_stat_' + process_unit, process_unit + '_train_stat.xlsx')
MODEL_PATH = os.path.join('models', process_unit, process_unit + '.joblib')
TEST_SIZE = 0.1
RANDOM_STATE = 42

# --- PyTorch & Model Hyperparameters ---
NUM_EPOCHS = 2500
LEARNING_RATE = 0.0005
BATCH_SIZE = -1 # Use -1 for full-batch training
LOG_INTERVAL = 100 # This is now only for potential future use, as tqdm will show live loss

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch found. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch using CPU. Training will be slower.")

# --- Data Loading and Preparation (with wide/long format fix) ---
def load_and_prepare_data(filepath: str):
    """
    Loads and preprocesses data from the specified Excel file.
    Handles both long and wide formats for the 'all_input' sheet.
    """
    print("1. Loading and preparing data...")
    df_input = pd.read_excel(filepath, sheet_name="all_input_" + process_unit)
    df_output = pd.read_excel(filepath, sheet_name="all_output_" + process_unit)

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
    
    y_cols = [col for col in data.columns if col.startswith('Target_')]
    Y = data[y_cols]

    inf_cols = sorted([col for col in input_cols if col.startswith('inf_')])
    proc_cols = sorted([col for col in input_cols if not col.startswith('inf_')])
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

# --- Training and Prediction Functions ---
def train_model_pytorch(X_train, Y_train, Z_train):
    """
    Trains the CoupledCLEFOModel using PyTorch.
    Loss is reported directly in the tqdm progress bar.
    """
    print("3. Training CLEFO model with PyTorch...")
    
    n_samples, n_indep = X_train.shape
    n_dep = Y_train.shape[1]
    n_inter = Z_train.shape[1]

    model = CoupledCLEFOModel(n_dep, n_indep, n_inter).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = WastewaterDataset(X_train, Y_train, Z_train)
    model.train()

    if BATCH_SIZE <= 0:
        print("   - Using full-batch training. Moving all data to GPU at once.")
        X_gpu = train_dataset.X.to(device)
        Y_gpu = train_dataset.Y.to(device)
        Z_gpu = train_dataset.Z.to(device)

        with tqdm(range(NUM_EPOCHS), desc="Training (Full-Batch)") as pbar:
            for epoch in pbar:
                Y_pred = model(X_gpu, Z_gpu)
                loss = criterion(Y_pred, Y_gpu)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(MSE=f'{loss.item():.6f}')
    else:
        print(f"   - Using mini-batch training with batch size: {BATCH_SIZE}")
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        with tqdm(range(NUM_EPOCHS), desc="Training (Mini-Batch)") as pbar:
            for epoch in pbar:
                epoch_loss = 0
                for X_batch, Y_batch, Z_batch in train_loader:
                    X_batch, Y_batch, Z_batch = X_batch.to(device), Y_batch.to(device), Z_batch.to(device)
                    
                    Y_pred = model(X_batch, Z_batch)
                    loss = criterion(Y_pred, Y_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                pbar.set_postfix(Avg_MSE=f'{avg_loss:.6f}')

    print("Model training complete.")
    return model

def predict_pytorch(model, X_data, Z_data):
    """
    Makes predictions using the trained PyTorch model.
    Optimized to handle full-batch prediction by moving data to GPU once.
    """
    print("4. Making predictions...")
    model.eval()
    with torch.no_grad():
        dataset = WastewaterDataset(X_data, np.zeros((X_data.shape[0], model.n_dep)), Z_data)
        
        effective_batch_size = BATCH_SIZE if BATCH_SIZE > 0 else len(dataset)
        
        if effective_batch_size >= len(dataset):
            X_gpu = dataset.X.to(device)
            Z_gpu = dataset.Z.to(device)
            Y_pred_gpu = model(X_gpu, Z_gpu)
            all_preds = Y_pred_gpu.cpu().numpy()
        else:
            loader = DataLoader(dataset=dataset, batch_size=effective_batch_size, shuffle=False)
            all_preds_list = []
            for X_batch, _, Z_batch in loader:
                X_batch, Z_batch = X_batch.to(device), Z_batch.to(device)
                Y_pred_batch = model(X_batch, Z_batch)
                all_preds_list.append(Y_pred_batch.cpu().numpy())
            all_preds = np.vstack(all_preds_list)
            
    print("Prediction complete.")
    return all_preds

# --- Evaluation ---
def calculate_statistics(y_true: pd.DataFrame, y_pred: np.ndarray, num_predictors: int):
    """Calculates performance metrics for the model predictions."""
    stats = []
    N = len(y_true)
    y_cols = y_true.columns
    
    for i, col_name in enumerate(y_cols):
        y_t = y_true[col_name].values
        y_p = y_pred[:, i]
        
        valid_indices = ~np.isnan(y_p)
        if not np.any(valid_indices):
            stats.append({'variable': col_name, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan, 'Adj_R2': np.nan})
            continue

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
    """Main function to orchestrate the entire workflow."""
    if not os.path.exists(FILE_PATH):
        print(f"Error: The data file was not found at '{FILE_PATH}'")
        return
        
    X, Y, x_cols, y_cols = load_and_prepare_data(FILE_PATH)
    
    print("2. Splitting and scaling data...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    X_train_s = x_scaler.transform(X_train)
    Y_train_s = y_scaler.transform(Y_train)
    
    m = X_train_s.shape[1]
    Z_train_s = create_interaction_features_np(X_train_s, m)

    trained_model = train_model_pytorch(X_train_s, Y_train_s, Z_train_s)
    
    os.makedirs('models', exist_ok=True)
    print("Saving model and scalers...")
    trained_model.to('cpu')
    model_bundle = {
        'model': trained_model,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'Upsilon': trained_model.Upsilon.detach().numpy(),
        'B': trained_model.B.detach().numpy(),
        'Theta': trained_model.Theta.detach().numpy(),
        'Gamma': trained_model.Gamma.detach().numpy(),
        'Lambda': trained_model.Lambda.detach().numpy()
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Model bundle saved to {MODEL_PATH}")

    print("5. Evaluating model performance using the saved joblib file...")
    loaded_bundle = joblib.load(MODEL_PATH)
    loaded_model = loaded_bundle['model'].to(device)
    loaded_x_scaler = loaded_bundle['x_scaler']
    loaded_y_scaler = loaded_bundle['y_scaler']
    
    X_train_s_loaded = loaded_x_scaler.transform(X_train)
    Z_train_s_loaded = create_interaction_features_np(X_train_s_loaded, m)
    Y_pred_train_s = predict_pytorch(loaded_model, X_train_s_loaded, Z_train_s_loaded)
    Y_pred_train = loaded_y_scaler.inverse_transform(Y_pred_train_s)

    X_test_s_loaded = loaded_x_scaler.transform(X_test)
    Z_test_s_loaded = create_interaction_features_np(X_test_s_loaded, m)
    Y_pred_test_s = predict_pytorch(loaded_model, X_test_s_loaded, Z_test_s_loaded)
    Y_pred_test = loaded_y_scaler.inverse_transform(Y_pred_test_s)
    
    n_dep = Y.shape[1]
    n_indep = X.shape[1]
    n_inter = Z_train_s.shape[1]
    num_predictors = 1 + n_indep + n_inter + (n_dep - 1) + n_indep

    print("   - Calculating and saving statistics...")
    cal_stats = calculate_statistics(Y_train, Y_pred_train, num_predictors)
    val_stats = calculate_statistics(Y_test, Y_pred_test, num_predictors)

    with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='xlsxwriter') as writer:
        pd.concat([X_train, Y_train], axis=1).to_excel(writer, sheet_name='calibration_data', index=False)
        pd.concat([X_test, Y_test], axis=1).to_excel(writer, sheet_name='validation_data', index=False)
        
        pd.DataFrame(np.hstack([X_train_s, Y_train_s]), columns=x_cols+y_cols).to_excel(writer, sheet_name='scaled_calibration_data', index=False)
        X_test_s = x_scaler.transform(X_test)
        pd.DataFrame(np.hstack([X_test_s, y_scaler.transform(Y_test)]), columns=x_cols+y_cols).to_excel(writer, sheet_name='scaled_validation_data', index=False)

        cal_stats.to_excel(writer, sheet_name='calibration_statistics', index=False)
        val_stats.to_excel(writer, sheet_name='validation_statistics', index=False)

        pd.DataFrame(Y_pred_train, columns=y_cols).to_excel(writer, sheet_name='calibration_prediction', index=False)
        pd.DataFrame(Y_pred_test, columns=y_cols).to_excel(writer, sheet_name='validation_prediction', index=False)

    print(f"\nProcess finished successfully. Results are saved in '{OUTPUT_FILE_PATH}'.")

if __name__ == '__main__':
    main()