import pandas as pd
import numpy as np
import itertools
import os
import sys
from pathlib import Path
import warnings
from tqdm import tqdm
import math
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import re
import optuna

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
from naming import CONCENTRATION_SUFFIX, canonical_base_name, normalize_stream_column, rename_concentration_columns

# --- Configuration & User Input ---
valid_units = ['clarifier', 'cstr']
process_unit = input(f"Enter process unit type {valid_units}: ").lower().strip()
if process_unit not in valid_units: process_unit = 'cstr'

try:
    N_SPLITS = int(input("Enter K-Fold splits (default: 5): ") or 5)
except:
    N_SPLITS = 5

# --- Configuration ---
solver_type = 'solve' # 'solve', 'pinv', 'lstsq'
optimizer_type = 'adamw' # 'adam', 'sgd', 'rmsprop', 'adamw'

# --- Paths ---
CONFIG_DIR = os.path.join('data', 'config')
FILE_PATH = os.path.join(CONFIG_DIR, 'simulation_training_config.xlsx')
MACHINE_LEARNING_MODEL = 'clefo'
BASE_OUTPUT_DIR = os.path.join('data', 'results', 'training', MACHINE_LEARNING_MODEL, process_unit)
OUTPUT_FILE_PATH = os.path.join(BASE_OUTPUT_DIR, 'train_stat.xlsx')
MODEL_PATH = os.path.join(BASE_OUTPUT_DIR, f'{process_unit}.joblib')
IMG_DIR = os.path.join(BASE_OUTPUT_DIR, 'images')

# Hyperparameter Paths
HYPERPARAM_DIR = os.path.join('data', 'results', 'training', MACHINE_LEARNING_MODEL, 'hyperparameters')
HYPERPARAM_FILE = os.path.join(HYPERPARAM_DIR, 'hyperparameters.xlsx')

# Ensure base directories exist immediately
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(HYPERPARAM_DIR, exist_ok=True)

# --- Data Loading ---
def load_and_prepare_data(filepath: str):
    print("1. Loading data...")
    df_input = pd.read_excel(filepath, sheet_name="all_input_" + process_unit)
    df_output = pd.read_excel(filepath, sheet_name="all_output_" + process_unit)
    df_output = rename_concentration_columns(df_output)

    # Handle Long vs Wide format
    if {'variable', 'default'}.issubset(df_input.columns):
        df_input['variable'] = df_input['variable'].apply(normalize_stream_column)
        df_input_wide = df_input.pivot(index='simulation_number', columns='variable', values='default').reset_index()
        input_cols = df_input['variable'].unique().tolist()
    else:
        df_input_wide = rename_concentration_columns(df_input)
        input_cols = [col for col in df_input_wide.columns if col != 'simulation_number']

    data = pd.merge(df_input_wide, df_output, on='simulation_number', how='inner')
    
    # Filter components if config exists
    try:
        df_comps = pd.read_excel(filepath, sheet_name="training_components")
        if 'considered' in df_comps.columns:
            remove = df_comps[df_comps['considered'] == 0]['components'].tolist()
            removal_tokens = [canonical_base_name(r) or r for r in remove]
            cols_to_drop = [c for c in data.columns if any(token in c for token in removal_tokens)]
            data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    except:
        pass

    y_cols = [col for col in data.columns if col.startswith(('effluent_', 'wastage_'))]
    current_inputs = [col for col in data.columns if col in input_cols]
    inf_cols = sorted([c for c in current_inputs if c.startswith('influent_')])
    proc_cols = sorted([c for c in current_inputs if not c.startswith('influent_')])
    x_cols_ordered = proc_cols + inf_cols

    # Return simulation_number as well for mapping back later
    return data[x_cols_ordered], data[y_cols], x_cols_ordered, y_cols, data['simulation_number']

# --- Feature Engineering ---
def create_interaction_features_np(X: np.ndarray, m: int):
    col_pairs = list(itertools.combinations(range(m), 2))
    Z = np.zeros((X.shape[0], len(col_pairs)), dtype=np.float64)
    for i, (c1, c2) in enumerate(col_pairs):
        Z[:, i] = X[:, c1] * X[:, c2]
    return Z, col_pairs

# --- PyTorch Model ---
class WastewaterDataset(Dataset):
    def __init__(self, X, Y, Z):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.Z = torch.tensor(Z, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx], self.Z[idx]

class CoupledCLEFOModel(nn.Module):
    def __init__(self, n_dep, n_indep, n_inter, solver_type='solve'):
        super().__init__()
        self.n_dep = n_dep
        self.solver_type = solver_type
        # Initialize small random weights
        self.Upsilon = nn.Parameter(torch.randn(n_dep, 1) * 0.01)
        self.B = nn.Parameter(torch.randn(n_dep, n_indep) * 0.01)
        self.Theta = nn.Parameter(torch.randn(n_dep, n_inter) * 0.01)
        self.Gamma = nn.Parameter(torch.randn(n_dep, n_dep) * 0.01)
        self.Lambda = nn.Parameter(torch.randn(n_dep, n_indep) * 0.01)
        
        # Zero diagonal for Gamma (no self-coupling)
        self.Gamma.register_hook(lambda grad: grad.fill_diagonal_(0))
        self.register_buffer('identity', torch.eye(n_dep))

    def forward(self, X, Z):
        batch_size = X.shape[0]
        RHS = self.Upsilon.expand(-1, batch_size) + (self.B @ X.T) + (self.Theta @ Z.T)
        RHS = RHS.T.unsqueeze(-1)
        
        LHS = self.identity.unsqueeze(0).expand(batch_size, -1, -1) - \
              self.Gamma.unsqueeze(0).expand(batch_size, -1, -1) - \
              torch.diag_embed((self.Lambda @ X.T).T)
        
        # Regularization for stability
        A = LHS + (torch.eye(self.n_dep, device=X.device).unsqueeze(0) * 1e-7)

        if self.solver_type == 'pinv':
            Y_pred = (torch.linalg.pinv(A) @ RHS).squeeze(-1)
        elif self.solver_type == 'lstsq':
            Y_pred = torch.linalg.lstsq(A, RHS).solution.squeeze(-1)
        else:
            try:
                Y_pred = torch.linalg.solve(A, RHS).squeeze(-1)
            except:
                Y_pred = torch.full((batch_size, self.n_dep), float('nan'), device=X.device)
        return Y_pred

# --- Training Logic ---
def train_model_pytorch(X_train, Y_train, Z_train, learning_rate, batch_size, l1_lambda, epochs, verbose=False):
    # Determine device based on batch_size
    if batch_size != -1:
        train_device = torch.device("cpu")
    else:
        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Training on device: {train_device}")

    model = CoupledCLEFOModel(Y_train.shape[1], X_train.shape[1], Z_train.shape[1], solver_type).to(train_device)
    criterion = nn.MSELoss()
    
    if optimizer_type == 'adam': opt = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd': opt = optim.SGD(model.parameters(), lr=learning_rate*10, momentum=0.9)
    elif optimizer_type == 'rmsprop': opt = optim.RMSprop(model.parameters(), lr=learning_rate)
    else: opt = optim.AdamW(model.parameters(), lr=learning_rate)

    dataset = WastewaterDataset(X_train, Y_train, Z_train)
    
    if batch_size <= 0:
        X_g, Y_g, Z_g = dataset.X.to(train_device), dataset.Y.to(train_device), dataset.Z.to(train_device)
        iterator = range(epochs)
        if verbose: iterator = tqdm(iterator, desc="Training")
        
        for _ in iterator:
            Y_pred = model(X_g, Z_g)
            reg = l1_lambda * (torch.norm(model.B, 1) + torch.norm(model.Theta, 1) + 
                               torch.norm(model.Lambda, 1) + torch.norm(model.Gamma, 1))
            loss = criterion(Y_pred, Y_g) + reg
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterator = range(epochs)
        if verbose: iterator = tqdm(iterator, desc="Training")
        
        for _ in iterator:
            for X_b, Y_b, Z_b in loader:
                X_b, Y_b, Z_b = X_b.to(train_device), Y_b.to(train_device), Z_b.to(train_device)
                loss = criterion(model(X_b, Z_b), Y_b) + \
                       l1_lambda * (torch.norm(model.B, 1) + torch.norm(model.Theta, 1))
                if torch.isnan(loss): return None
                opt.zero_grad()
                loss.backward()
                opt.step()
    return model

def predict_pytorch(model, X, Z):
    # Determine device from model parameters
    pred_device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(pred_device), 
                     torch.tensor(Z, dtype=torch.float32).to(pred_device)).cpu().numpy()

# --- Hyperparameter Optimization ---
def get_hyperparameters(X_s, Y_s, Z_s):
    """
    Fetches hyperparameters from Excel. If missing, runs Optuna optimization,
    saves the results, and returns them.
    """
    params = {}
    file_exists = os.path.exists(HYPERPARAM_FILE)
    
    if file_exists:
        try:
            df_params = pd.read_excel(HYPERPARAM_FILE)
            # Check if required columns exist and have values
            required_cols = ['learning_rate', 'batch_size', 'l1_lambda', 'num_epochs']
            if all(col in df_params.columns for col in required_cols) and not df_params.empty:
                params['learning_rate'] = float(df_params['learning_rate'].iloc[0])
                params['batch_size'] = int(df_params['batch_size'].iloc[0])
                params['l1_lambda'] = float(df_params['l1_lambda'].iloc[0])
                params['num_epochs'] = int(df_params['num_epochs'].iloc[0])
                print(f"Loaded hyperparameters from {HYPERPARAM_FILE}")
                return params
        except Exception as e:
            print(f"Error reading hyperparameter file: {e}. Proceeding to optimization.")

    print("Hyperparameters not found or invalid. Starting Optuna optimization...")
    
    # Split data for optimization (Train/Val split)
    X_train, X_val, Y_train, Y_val, Z_train, Z_val = train_test_split(
        X_s, Y_s, Z_s, test_size=0.2, random_state=42
    )

    def objective(trial):
        # Suggest parameters
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        bs = trial.suggest_categorical('batch_size', [16, 32, 64, 128, -1])
        l1 = trial.suggest_float('l1_lambda', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('num_epochs', 100, 1000, step=100)
        
        # Train model
        model = train_model_pytorch(X_train, Y_train, Z_train, lr, bs, l1, epochs, verbose=False)
        
        if model is None: # Handle NaN loss
            return float('inf')
            
        # Evaluate
        Y_pred_val = predict_pytorch(model, X_val, Z_val)
        mse = mean_squared_error(Y_val, Y_pred_val)
        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    best_params = study.best_params
    print("Optimization finished. Best parameters:")
    print(best_params)
    
    # Save to Excel
    df_best = pd.DataFrame([best_params])
    df_best.to_excel(HYPERPARAM_FILE, index=False)
    print(f"Saved optimized hyperparameters to {HYPERPARAM_FILE}")
    
    return best_params

# --- Analysis Functions ---

def analyze_sparsity(model, threshold=1e-3):
    """Calculates sparsity of the trained model."""
    params = {
        'B': model.B.detach().cpu().numpy(),
        'Theta': model.Theta.detach().cpu().numpy(),
        'Gamma': model.Gamma.detach().cpu().numpy(),
        'Lambda': model.Lambda.detach().cpu().numpy()
    }
    
    sparsity_data = []
    for name, matrix in params.items():
        total = matrix.size
        non_zero = np.sum(np.abs(matrix) > threshold)
        sparsity_data.append({
            'Matrix': name,
            'Total Terms': total,
            'Retained Terms': non_zero,
            'Sparsity Ratio (%)': round((non_zero / total) * 100, 2)
        })
    return pd.DataFrame(sparsity_data)

def calculate_feature_importance(model, x_cols, col_pairs):
    """Calculates feature importance based on summed absolute coefficients."""
    B = np.abs(model.B.detach().cpu().numpy()) # (n_dep, n_indep)
    Lambda = np.abs(model.Lambda.detach().cpu().numpy()) # (n_dep, n_indep)
    Theta = np.abs(model.Theta.detach().cpu().numpy()) # (n_dep, n_inter)
    
    importance = np.zeros(len(x_cols))
    
    # Sum B and Lambda contributions directly
    importance += np.sum(B, axis=0)
    importance += np.sum(Lambda, axis=0)
    
    # Distribute Theta contributions to constituent inputs
    theta_sum = np.sum(Theta, axis=0) # Sum across outputs -> (n_inter,)
    for idx, (i, j) in enumerate(col_pairs):
        importance[i] += theta_sum[idx]
        importance[j] += theta_sum[idx]
        
    # Normalize
    importance = 100 * importance / np.sum(importance)
    
    df_imp = pd.DataFrame({'Feature': x_cols, 'Importance': importance})
    return df_imp.sort_values('Importance', ascending=False)

def analyze_regime_generalization(X, Y_true, Y_pred, x_cols, y_cols):
    """
    Stratifies data by KLa (Oxygen Transfer) and calculates MAPE for each regime.
    Regimes: Oxygen Limited (<50), Transition (50-150), High Aeration (>150).
    """
    print("   - Running Regime Generalization Analysis...")
    
    # Identify KLa columns (case insensitive)
    kla_cols = [c for c in x_cols if 'kla' in c.lower()]
    
    if not kla_cols:
        print("     [Warning] No 'KLa' columns found. Skipping regime analysis.")
        return pd.DataFrame()

    # Calculate mean KLa for stratification
    kla_values = X[kla_cols].mean(axis=1)
    
    regimes = {
        'Oxygen Limited (<50)': kla_values < 50,
        'Transition (50-150)': (kla_values >= 50) & (kla_values <= 150),
        'High Aeration (>150)': kla_values > 150
    }
    
    results = []
    
    for regime_name, mask in regimes.items():
        if not mask.any():
            continue
            
        y_t_reg = Y_true[mask]
        y_p_reg = Y_pred[mask]
        
        for i, col in enumerate(y_cols):
            # Calculate MAPE (Mean Absolute Percentage Error)
            # Add epsilon to avoid division by zero
            y_true_safe = y_t_reg.iloc[:, i].replace(0, 1e-6)
            mape = np.mean(np.abs((y_true_safe - y_p_reg[:, i]) / y_true_safe)) * 100
            
            results.append({
                'Regime': regime_name,
                'Variable': col,
                'MAPE (%)': mape,
                'Sample_Count': mask.sum()
            })
            
    df_res = pd.DataFrame(results)
    
    if not df_res.empty:
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_res, x='Regime', y='MAPE (%)', hue='Variable')
        plt.title('Regime Generalization Analysis (MAPE)')
        plt.ylabel('Mean Absolute Percentage Error (%)')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'regime_analysis.png'), dpi=300)
        plt.close()
        
    return df_res

# --- Visualization Functions ---

def sanitize_filename(name):
    """Replaces invalid characters in filenames with underscores."""
    return re.sub(r'[\\/*?:"<>| ]', '_', name)

def plot_parity(Y_true, Y_pred, y_cols, save_path, title_prefix=""):
    """Generates a parity plot for all variables."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_vars = len(y_cols)
    cols = 3
    rows = math.ceil(n_vars / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    # Handle pandas Series/DataFrame vs numpy array inputs
    if isinstance(Y_true, (pd.DataFrame, pd.Series)):
        Y_true_np = Y_true.values
    else:
        Y_true_np = Y_true

    for i, col in enumerate(y_cols):
        ax = axes[i]
        y_t = Y_true_np[:, i]
        y_p = Y_pred[:, i]
        
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        
        ax.scatter(y_t, y_p, alpha=0.5, s=10)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title(f"{title_prefix} {col}")
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        ax.grid(True)
        
    for j in range(i+1, len(axes)): axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_heatmaps(model, x_cols, y_cols, col_pairs):
    """Generates heatmaps for Theta (Interactions) and Gamma (Coupling)."""
    os.makedirs(IMG_DIR, exist_ok=True)

    # 1. Gamma Matrix (Coupling)
    Gamma = model.Gamma.detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(Gamma, xticklabels=y_cols, yticklabels=y_cols, cmap='coolwarm', center=0)
    plt.title("Stoichiometric Coupling Matrix (Gamma)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'heatmap_gamma.png'), dpi=300)
    plt.close()
    
    # 2. Theta Tensor (Iterate over ALL output variables)
    Theta = model.Theta.detach().cpu().numpy() # (n_dep, n_inter)
    n_in = len(x_cols)
    
    print("   - Generating Theta heatmaps for all target variables...")
    for target_idx, target_name in enumerate(y_cols):
        safe_name = sanitize_filename(target_name)
        
        interaction_matrix = np.zeros((n_in, n_in))
        theta_slice = Theta[target_idx, :]
        
        for idx, (i, j) in enumerate(col_pairs):
            val = theta_slice[idx]
            interaction_matrix[i, j] = val
            interaction_matrix[j, i] = val # Symmetric
            
        plt.figure(figsize=(12, 10))
        sns.heatmap(interaction_matrix, xticklabels=x_cols, yticklabels=x_cols, cmap='coolwarm', center=0)
        plt.title(f"Interaction Tensor (Theta) for {target_name}")
        plt.tight_layout()
        
        save_path = os.path.join(IMG_DIR, f'heatmap_theta_{safe_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

# --- Main Workflow ---
def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    # 1. Load Data (Updated to return simulation numbers)
    X, Y, x_cols, y_cols, sim_nums = load_and_prepare_data(FILE_PATH)
    Y_log = np.log1p(Y)
    m = X.shape[1]

    # Prepare Scaled Data for Optimization
    sc_x_temp = StandardScaler().fit(X)
    sc_y_temp = StandardScaler().fit(Y_log)
    X_s_temp = sc_x_temp.transform(X)
    Y_s_temp = sc_y_temp.transform(Y_log)
    Z_s_temp, _ = create_interaction_features_np(X_s_temp, m)

    # 2. Get Hyperparameters (Load or Optimize)
    print("\n2. Fetching Hyperparameters...")
    hp = get_hyperparameters(X_s_temp, Y_s_temp, Z_s_temp)
    
    LEARNING_RATE = hp['learning_rate']
    BATCH_SIZE = int(hp['batch_size'])
    L1_LAMBDA = hp['l1_lambda']
    NUM_EPOCHS = int(hp['num_epochs'])

    # 3. K-Fold Cross-Validation
    print(f"\n3. Starting {N_SPLITS}-Fold Cross-Validation...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Store detailed results for Excel
    kfold_detailed_results = []
    
    for fold, (tr_idx, te_idx) in enumerate(tqdm(kf.split(X, Y_log), total=N_SPLITS)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        Y_tr_log, Y_te = Y_log.iloc[tr_idx], Y.iloc[te_idx]
        
        # Physical values for training parity plot
        Y_tr_real = np.expm1(Y_tr_log)

        sc_x = StandardScaler().fit(X_tr)
        sc_y = StandardScaler().fit(Y_tr_log)
        
        X_tr_s = sc_x.transform(X_tr)
        X_te_s = sc_x.transform(X_te)
        Y_tr_s = sc_y.transform(Y_tr_log)
        Z_tr_s, _ = create_interaction_features_np(X_tr_s, m)
        Z_te_s, _ = create_interaction_features_np(X_te_s, m)
        
        model = None
        while model is None:
            model = train_model_pytorch(X_tr_s, Y_tr_s, Z_tr_s, LEARNING_RATE, BATCH_SIZE, L1_LAMBDA, NUM_EPOCHS)
            
        # --- Predictions ---
        # Train Set
        Y_pred_tr_s = predict_pytorch(model, X_tr_s, Z_tr_s)
        Y_pred_tr = np.expm1(sc_y.inverse_transform(Y_pred_tr_s))
        
        # Test Set
        Y_pred_te_s = predict_pytorch(model, X_te_s, Z_te_s)
        Y_pred_te = np.expm1(sc_y.inverse_transform(Y_pred_te_s))
        
        # --- Metrics Calculation & Storage ---
        for i, col in enumerate(y_cols):
            # Train Metrics
            mse_tr = mean_squared_error(Y_tr_real.iloc[:, i], Y_pred_tr[:, i])
            r2_tr = r2_score(Y_tr_real.iloc[:, i], Y_pred_tr[:, i])
            kfold_detailed_results.append({
                'Fold': fold + 1, 'Type': 'Train', 'Variable': col, 
                'MSE': mse_tr, 'RMSE': np.sqrt(mse_tr), 'R2': r2_tr
            })
            
            # Test Metrics
            mse_te = mean_squared_error(Y_te.iloc[:, i], Y_pred_te[:, i])
            r2_te = r2_score(Y_te.iloc[:, i], Y_pred_te[:, i])
            kfold_detailed_results.append({
                'Fold': fold + 1, 'Type': 'Test', 'Variable': col, 
                'MSE': mse_te, 'RMSE': np.sqrt(mse_te), 'R2': r2_te
            })

        # --- Parity Plots for this Fold ---
        fold_dir = os.path.join(IMG_DIR, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        plot_parity(Y_tr_real, Y_pred_tr, y_cols, 
                    save_path=os.path.join(fold_dir, 'parity_train.png'), 
                    title_prefix=f"Fold {fold+1} Train")
        
        plot_parity(Y_te, Y_pred_te, y_cols, 
                    save_path=os.path.join(fold_dir, 'parity_test.png'), 
                    title_prefix=f"Fold {fold+1} Test")

    # Create DataFrame for Excel
    kfold_df = pd.DataFrame(kfold_detailed_results)
    
    # Calculate Summary for Console Output (Test set only)
    cv_summary = kfold_df[kfold_df['Type'] == 'Test'].groupby('Variable').agg({'R2': ['mean', 'std'], 'RMSE': ['mean', 'std']})
    print("\nCross-Validation Results (Test Set):")
    print(cv_summary['R2'])

    # 4. Final Model Training
    print("\n4. Training Final Model...")
    sc_x_final = StandardScaler().fit(X)
    sc_y_final = StandardScaler().fit(Y_log)
    X_s = sc_x_final.transform(X)
    Y_s = sc_y_final.transform(Y_log)
    Z_s, col_pairs = create_interaction_features_np(X_s, m)
    
    start_time = time.time()
    final_model = None
    while final_model is None:
        final_model = train_model_pytorch(X_s, Y_s, Z_s, LEARNING_RATE, BATCH_SIZE, L1_LAMBDA, NUM_EPOCHS, verbose=True)
    train_time = time.time() - start_time
    print(f"   - Training Time: {train_time:.2f} seconds")

    # 5. Generate Predictions & Plots (Final Model)
    Y_pred_s_full = predict_pytorch(final_model, X_s, Z_s)
    Y_pred_full = np.expm1(sc_y_final.inverse_transform(Y_pred_s_full))
    
    print("   - Generating Final Parity Plot...")
    plot_parity(Y, Y_pred_full, y_cols, 
                save_path=os.path.join(IMG_DIR, 'parity_plot_final_model.png'),
                title_prefix="Final Model")
    
    print("   - Generating Heatmaps...")
    plot_heatmaps(final_model, x_cols, y_cols, col_pairs)

    # 6. Advanced Analysis
    print("   - Analyzing Sparsity...")
    sparsity_df = analyze_sparsity(final_model)
    print(sparsity_df)
    
    print("   - Calculating Feature Importance...")
    imp_df = calculate_feature_importance(final_model, x_cols, col_pairs)
    
    # 7. Regime Generalization Analysis
    regime_df = analyze_regime_generalization(X, Y, Y_pred_full, x_cols, y_cols)
    
    # 8. Save Results
    print(f"\n5. Saving results to {OUTPUT_FILE_PATH}...")
    
    # Prepare stats for full model
    full_stats = []
    for i, col in enumerate(y_cols):
        mse = mean_squared_error(Y.iloc[:, i], Y_pred_full[:, i])
        full_stats.append({'Variable': col, 'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2_score(Y.iloc[:, i], Y_pred_full[:, i])})
    full_stats_df = pd.DataFrame(full_stats)

    # --- NEW: Generate data_current_mape.xlsx with MAPE ---
    print("   - Creating data_current_mape.xlsx with MAPE column...")
    
    # Calculate MAPE per datapoint (row-wise across all variables)
    Y_vals = Y.values
    # Avoid division by zero
    Y_vals_safe = np.where(Y_vals == 0, 1e-6, Y_vals)
    ape_matrix = np.abs((Y_vals_safe - Y_pred_full) / Y_vals_safe) * 100
    mape_per_sim = np.mean(ape_matrix, axis=1)
    
    # Create mapping DataFrame
    mape_df = pd.DataFrame({
        'simulation_number': sim_nums.values,
        'MAPE': mape_per_sim
    })
    
    # Define path: results --> training --> clefo --> data_current_mape.xlsx
    mape_file_dir = os.path.join('data', 'results', 'training', MACHINE_LEARNING_MODEL)
    copy_file_path = os.path.join(mape_file_dir, 'data_current_mape.xlsx')
    
    # Define target sheets
    target_sheets = ['all_input_cstr', 'all_output_cstr', 'all_input_clarifier', 'all_output_clarifier', 'all_input_flows']
    
    os.makedirs(os.path.dirname(copy_file_path), exist_ok=True)
    
    with pd.ExcelWriter(copy_file_path, engine='xlsxwriter') as writer:
        for sheet_name in target_sheets:
            try:
                # Read original sheet
                df_orig = pd.read_excel(FILE_PATH, sheet_name=sheet_name)
                
                if 'simulation_number' in df_orig.columns:
                    # Merge MAPE based on simulation number
                    # We use how='left' to keep all original rows. 
                    # Rows not in the training set (filtered out) will get NaN MAPE.
                    df_orig = pd.merge(df_orig, mape_df, on='simulation_number', how='left')
                else:
                    # Fallback if simulation_number missing
                    df_orig['MAPE'] = np.nan
                
                df_orig.to_excel(writer, sheet_name=sheet_name, index=False)
            except ValueError:
                print(f"     [Warning] Sheet '{sheet_name}' not found in source file.")
            except Exception as e:
                print(f"     [Error] Failed to process sheet '{sheet_name}': {e}")

    # Save standard training stats
    with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='xlsxwriter') as writer:
        cv_summary.to_excel(writer, sheet_name='CV_Summary')
        kfold_df.to_excel(writer, sheet_name='KFold_Detailed_Results', index=False)
        full_stats_df.to_excel(writer, sheet_name='Final_Model_Stats')
        sparsity_df.to_excel(writer, sheet_name='Sparsity_Analysis')
        imp_df.to_excel(writer, sheet_name='Feature_Importance')
        if not regime_df.empty:
            regime_df.to_excel(writer, sheet_name='Regime_Analysis', index=False)
        pd.DataFrame({'Training_Time_Sec': [train_time]}).to_excel(writer, sheet_name='Comp_Cost')
        
        # Data dumps
        pd.concat([X, Y], axis=1).to_excel(writer, sheet_name='Calibration_Data', index=False)
        pd.DataFrame(Y_pred_full, columns=y_cols).to_excel(writer, sheet_name='Calibration_Preds', index=False)

    # Save Model Object AND Coefficients explicitly for Optimization Code
    save_dict = {
        'model': final_model, 
        'x_scaler': sc_x_final, 
        'y_scaler': sc_y_final,
        'col_pairs': col_pairs, 
        'x_cols': x_cols, 
        'y_cols': y_cols,
        # Explicitly save weights as numpy arrays for the optimization code
        'Upsilon': final_model.Upsilon.detach().cpu().numpy(),
        'B': final_model.B.detach().cpu().numpy(),
        'Theta': final_model.Theta.detach().cpu().numpy(),
        'Gamma': final_model.Gamma.detach().cpu().numpy(),
        'Lambda': final_model.Lambda.detach().cpu().numpy()
    }
    joblib.dump(save_dict, MODEL_PATH)

    print("Done.")

if __name__ == '__main__':
    main()
