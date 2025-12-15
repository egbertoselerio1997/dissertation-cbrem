import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
import math
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
import optuna

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Scikit-learn Imports ---
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
from naming import canonical_base_name, normalize_stream_column, rename_concentration_columns

# --- Configuration & User Input ---
valid_units = ['clarifier', 'cstr']
process_unit = input(f"Enter process unit type {valid_units}: ").lower().strip()
if process_unit not in valid_units: process_unit = 'cstr'

try:
    N_SPLITS = int(input("Enter the number of folds for cross-validation (default: 10): ") or 10)
except ValueError:
    N_SPLITS = 10

# Paths
CONFIG_DIR = os.path.join('data', 'config')
FILE_PATH = os.path.join(CONFIG_DIR, 'data.xlsx')
MACHINE_LEARNING_MODEL = 'ann'
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

# Device Setup: prefer CUDA, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

    return data[x_cols_ordered], data[y_cols], x_cols_ordered, y_cols

# --- PyTorch Model ---
class WastewaterDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

class ANNModel(nn.Module):
    def __init__(self, n_indep, n_dep, h1=64, h2=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_indep, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_dep)
        )
    def forward(self, X):
        return self.layers(X)

# --- Training Logic ---
def train_ann_model(X_train, Y_train, n_epochs, batch_size, learning_rate, h1, h2, verbose=False):
    n_samples, n_indep = X_train.shape
    n_dep = Y_train.shape[1]

    model = ANNModel(n_indep, n_dep, h1, h2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = WastewaterDataset(X_train, Y_train)
    
    if batch_size <= 0:
        X_g, Y_g = dataset.X.to(device), dataset.Y.to(device)
        iterator = range(n_epochs)
        if verbose: iterator = tqdm(iterator, desc="Training ANN")
        
        for _ in iterator:
            Y_pred = model(X_g)
            loss = criterion(Y_pred, Y_g)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterator = range(n_epochs)
        if verbose: iterator = tqdm(iterator, desc="Training ANN")
        
        for _ in iterator:
            for X_b, Y_b in loader:
                X_b, Y_b = X_b.to(device), Y_b.to(device)
                loss = criterion(model(X_b), Y_b)
                if torch.isnan(loss): return None # Handle explosion
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return model

def predict_ann(model, X_data):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
        return model(X_tensor).cpu().numpy()

# --- Hyperparameter Optimization ---
def get_hyperparameters(X_s, Y_s):
    """
    Fetches hyperparameters from Excel. If missing, runs Optuna optimization,
    saves the results, and returns them.
    """
    params = {}
    file_exists = os.path.exists(HYPERPARAM_FILE)
    
    if file_exists:
        try:
            df_params = pd.read_excel(HYPERPARAM_FILE)
            required_cols = ['learning_rate', 'batch_size', 'num_epochs', 'hidden_dim1', 'hidden_dim2']
            if all(col in df_params.columns for col in required_cols) and not df_params.empty:
                params['learning_rate'] = float(df_params['learning_rate'].iloc[0])
                params['batch_size'] = int(df_params['batch_size'].iloc[0])
                params['num_epochs'] = int(df_params['num_epochs'].iloc[0])
                params['hidden_dim1'] = int(df_params['hidden_dim1'].iloc[0])
                params['hidden_dim2'] = int(df_params['hidden_dim2'].iloc[0])
                print(f"Loaded hyperparameters from {HYPERPARAM_FILE}")
                return params
        except Exception as e:
            print(f"Error reading hyperparameter file: {e}. Proceeding to optimization.")

    print("Hyperparameters not found or invalid. Starting Optuna optimization...")
    
    # Split data for optimization
    X_train, X_val, Y_train, Y_val = train_test_split(X_s, Y_s, test_size=0.2, random_state=42)

    def objective(trial):
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        bs = trial.suggest_categorical('batch_size', [16, 32, 64, 128, -1])
        epochs = trial.suggest_int('num_epochs', 500, 3000, step=500)
        h1 = trial.suggest_int('hidden_dim1', 32, 128, step=16)
        h2 = trial.suggest_int('hidden_dim2', 16, 64, step=16)
        
        model = train_ann_model(X_train, Y_train, epochs, bs, lr, h1, h2, verbose=False)
        
        if model is None: return float('inf')
        
        Y_pred_val = predict_ann(model, X_val)
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

def calculate_permutation_importance(model, X, Y, feature_names):
    """
    Calculates feature importance for ANN using Permutation Importance.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        baseline_pred = model(X_tensor)
        baseline_loss = criterion(baseline_pred, Y_tensor).item()
    
    importances = []
    
    for i, col in enumerate(feature_names):
        X_permuted = X_tensor.clone()
        idx = torch.randperm(X_permuted.size(0))
        X_permuted[:, i] = X_permuted[idx, i]
        
        with torch.no_grad():
            perm_pred = model(X_permuted)
            perm_loss = criterion(perm_pred, Y_tensor).item()
        
        importances.append(perm_loss - baseline_loss)
        
    importances = np.array(importances)
    importances = np.maximum(importances, 0)
    if np.sum(importances) > 0:
        importances = 100 * importances / np.sum(importances)
    
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
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

def plot_parity(Y_true, Y_pred, y_cols, save_path, title_prefix=""):
    """Generates a parity plot for all variables."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_vars = len(y_cols)
    cols = 3
    rows = math.ceil(n_vars / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
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

# --- Main Workflow ---
def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    # 1. Load Data
    X, Y, x_cols, y_cols = load_and_prepare_data(FILE_PATH)
    Y_log = np.log(Y)

    # Prepare Scaled Data for Optimization
    sc_x_temp = StandardScaler().fit(X)
    sc_y_temp = StandardScaler().fit(Y_log)
    X_s_temp = sc_x_temp.transform(X)
    Y_s_temp = sc_y_temp.transform(Y_log)

    # 2. Get Hyperparameters (Load or Optimize)
    print("\n2. Fetching Hyperparameters...")
    hp = get_hyperparameters(X_s_temp, Y_s_temp)
    
    LEARNING_RATE = hp['learning_rate']
    BATCH_SIZE = int(hp['batch_size'])
    NUM_EPOCHS = int(hp['num_epochs'])
    HIDDEN_DIM1 = int(hp['hidden_dim1'])
    HIDDEN_DIM2 = int(hp['hidden_dim2'])

    # 3. K-Fold Cross-Validation
    print(f"\n3. Starting {N_SPLITS}-Fold Cross-Validation...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    kfold_detailed_results = []

    for fold, (tr_idx, te_idx) in enumerate(tqdm(kf.split(X, Y_log), total=N_SPLITS)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        Y_tr_log, Y_te = Y_log.iloc[tr_idx], Y.iloc[te_idx]
        
        # Physical values for training parity plot
        Y_tr_real = np.exp(Y_tr_log)

        sc_x = StandardScaler().fit(X_tr)
        sc_y = StandardScaler().fit(Y_tr_log)
        
        X_tr_s = sc_x.transform(X_tr)
        X_te_s = sc_x.transform(X_te)
        Y_tr_s = sc_y.transform(Y_tr_log)
        
        model = None
        while model is None:
            model = train_ann_model(X_tr_s, Y_tr_s, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, HIDDEN_DIM1, HIDDEN_DIM2)
        
        # --- Predictions ---
        # Train Set
        Y_pred_tr_s = predict_ann(model, X_tr_s)
        Y_pred_tr = np.exp(sc_y.inverse_transform(Y_pred_tr_s))
        
        # Test Set
        Y_pred_te_s = predict_ann(model, X_te_s)
        Y_pred_te = np.exp(sc_y.inverse_transform(Y_pred_te_s))
        
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
    
    start_time = time.time()
    final_model = None
    while final_model is None:
        final_model = train_ann_model(X_s, Y_s, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, HIDDEN_DIM1, HIDDEN_DIM2, verbose=True)
    train_time = time.time() - start_time
    print(f"   - Training Time: {train_time:.2f} seconds")

    # 5. Generate Predictions & Plots (Final Model)
    Y_pred_s_full = predict_ann(final_model, X_s)
    Y_pred_full = np.exp(sc_y_final.inverse_transform(Y_pred_s_full))
    
    print("   - Generating Final Parity Plot...")
    plot_parity(Y, Y_pred_full, y_cols, 
                save_path=os.path.join(IMG_DIR, 'parity_plot_final_model.png'),
                title_prefix="Final Model")
    
    print("   - Calculating Feature Importance (Permutation)...")
    imp_df = calculate_permutation_importance(final_model, X_s, Y_s, x_cols)

    # 6. Regime Generalization Analysis
    regime_df = analyze_regime_generalization(X, Y, Y_pred_full, x_cols, y_cols)

    # 7. Save Results
    print(f"\n5. Saving results to {OUTPUT_FILE_PATH}...")
    
    # Prepare stats for full model
    full_stats = []
    for i, col in enumerate(y_cols):
        mse = mean_squared_error(Y.iloc[:, i], Y_pred_full[:, i])
        full_stats.append({'Variable': col, 'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2_score(Y.iloc[:, i], Y_pred_full[:, i])})
    full_stats_df = pd.DataFrame(full_stats)

    with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='xlsxwriter') as writer:
        cv_summary.to_excel(writer, sheet_name='CV_Summary')
        kfold_df.to_excel(writer, sheet_name='KFold_Detailed_Results', index=False)
        full_stats_df.to_excel(writer, sheet_name='Final_Model_Stats')
        imp_df.to_excel(writer, sheet_name='Feature_Importance')
        if not regime_df.empty:
            regime_df.to_excel(writer, sheet_name='Regime_Analysis', index=False)
        pd.DataFrame({'Training_Time_Sec': [train_time]}).to_excel(writer, sheet_name='Comp_Cost')
        
        # Data dumps
        pd.concat([X, Y], axis=1).to_excel(writer, sheet_name='Calibration_Data', index=False)
        pd.DataFrame(Y_pred_full, columns=y_cols).to_excel(writer, sheet_name='Calibration_Preds', index=False)

    # Save Model Object
    joblib.dump({
        'model': final_model, 'x_scaler': sc_x_final, 'y_scaler': sc_y_final,
        'x_cols': x_cols, 'y_cols': y_cols,
        'hyperparameters': hp
    }, MODEL_PATH)

    print("Done.")

if __name__ == '__main__':
    main()
