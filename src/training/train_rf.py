import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
import math
import joblib
import time
import re
from tqdm import tqdm
import optuna

# --- Matplotlib Backend Fix ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# --- Scikit-learn Imports ---
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
from naming import canonical_base_name, normalize_stream_column, rename_concentration_columns

# --- User Input for Process Unit ---
valid_units = ['clarifier', 'cstr']
process_unit = input(f"Enter process unit type {valid_units}: ").lower().strip()
if process_unit not in valid_units: process_unit = 'cstr'

# --- Get N_SPLITS from user ---
try:
    N_SPLITS = int(input("Enter the number of folds for cross-validation (default: 10): ") or 10)
except ValueError:
    N_SPLITS = 10

# Paths
CONFIG_DIR = os.path.join('data', 'config')
FILE_PATH = os.path.join(CONFIG_DIR, 'simulation_training_config.xlsx')
MACHINE_LEARNING_MODEL = 'random_forest'
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

# Hyperparameters
RANDOM_STATE = 42

# --- Data Loading ---
def load_and_prepare_data(filepath: str):
    print("1. Loading data...")
    df_input = pd.read_excel(filepath, sheet_name="all_input_" + process_unit)
    df_output = pd.read_excel(filepath, sheet_name="all_output_" + process_unit)
    df_output = rename_concentration_columns(df_output)

    if {'variable', 'default'}.issubset(df_input.columns):
        df_input['variable'] = df_input['variable'].apply(normalize_stream_column)
        df_input_wide = df_input.pivot(index='simulation_number', columns='variable', values='default').reset_index()
        input_cols = df_input['variable'].unique().tolist()
    else:
        df_input_wide = rename_concentration_columns(df_input)
        input_cols = [col for col in df_input_wide.columns if col != 'simulation_number']

    data = pd.merge(df_input_wide, df_output, on='simulation_number', how='inner')
    
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
            required_cols = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
            if all(col in df_params.columns for col in required_cols) and not df_params.empty:
                params['n_estimators'] = int(df_params['n_estimators'].iloc[0])
                params['max_depth'] = int(df_params['max_depth'].iloc[0])
                params['min_samples_split'] = int(df_params['min_samples_split'].iloc[0])
                params['min_samples_leaf'] = int(df_params['min_samples_leaf'].iloc[0])
                print(f"Loaded hyperparameters from {HYPERPARAM_FILE}")
                return params
        except Exception as e:
            print(f"Error reading hyperparameter file: {e}. Proceeding to optimization.")

    print("Hyperparameters not found or invalid. Starting Optuna optimization...")
    
    # Split data for optimization
    X_train, X_val, Y_train, Y_val = train_test_split(X_s, Y_s, test_size=0.2, random_state=RANDOM_STATE)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, Y_train)
        
        preds = model.predict(X_val)
        mse = mean_squared_error(Y_val, preds)
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

# --- Model Training ---
def train_random_forest_model(X_train: np.ndarray, Y_train: np.ndarray, params: dict):
    rf_model = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        bootstrap=True,
        ccp_alpha=0.0,
        criterion='squared_error',
        max_features=1.0, # Default to using all features or fraction
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_weight_fraction_leaf=0.0,
        n_jobs=-1,
        oob_score=False,
        random_state=RANDOM_STATE,
        verbose=0,
        warm_start=False
    )
    rf_model.fit(X_train, Y_train)
    return rf_model

# --- Analysis Functions ---

def calculate_rf_importance(model, feature_names):
    """Extracts Gini importance from Random Forest."""
    importances = model.feature_importances_
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

    X, Y, x_cols, y_cols = load_and_prepare_data(FILE_PATH)
    Y_log = np.log(Y + 1)

    # Prepare Scaled Data for Optimization
    sc_x_temp = StandardScaler().fit(X)
    sc_y_temp = StandardScaler().fit(Y_log)
    X_s_temp = sc_x_temp.transform(X)
    Y_s_temp = sc_y_temp.transform(Y_log)

    # 2. Get Hyperparameters (Load or Optimize)
    print("\n2. Fetching Hyperparameters...")
    hp = get_hyperparameters(X_s_temp, Y_s_temp)

    # 3. K-Fold Cross-Validation
    print(f"\n3. Starting {N_SPLITS}-Fold Cross-Validation...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    kfold_detailed_results = []

    for fold, (tr_idx, te_idx) in enumerate(tqdm(kf.split(X, Y_log), total=N_SPLITS)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        Y_tr_log, Y_te = Y_log.iloc[tr_idx], Y.iloc[te_idx]
        
        Y_tr_real = np.exp(Y_tr_log) - 1

        sc_x = StandardScaler().fit(X_tr)
        sc_y = StandardScaler().fit(Y_tr_log)
        
        X_tr_s = sc_x.transform(X_tr)
        X_te_s = sc_x.transform(X_te)
        Y_tr_s = sc_y.transform(Y_tr_log)
        
        model = train_random_forest_model(X_tr_s, Y_tr_s, hp)
        
        # Predictions
        Y_pred_tr_s = model.predict(X_tr_s)
        Y_pred_tr = np.exp(sc_y.inverse_transform(Y_pred_tr_s)) - 1
        
        Y_pred_te_s = model.predict(X_te_s)
        Y_pred_te = np.exp(sc_y.inverse_transform(Y_pred_te_s)) - 1
        
        # Metrics
        for i, col in enumerate(y_cols):
            mse_tr = mean_squared_error(Y_tr_real.iloc[:, i], Y_pred_tr[:, i])
            r2_tr = r2_score(Y_tr_real.iloc[:, i], Y_pred_tr[:, i])
            kfold_detailed_results.append({
                'Fold': fold + 1, 'Type': 'Train', 'Variable': col, 
                'MSE': mse_tr, 'RMSE': np.sqrt(mse_tr), 'R2': r2_tr
            })
            
            mse_te = mean_squared_error(Y_te.iloc[:, i], Y_pred_te[:, i])
            r2_te = r2_score(Y_te.iloc[:, i], Y_pred_te[:, i])
            kfold_detailed_results.append({
                'Fold': fold + 1, 'Type': 'Test', 'Variable': col, 
                'MSE': mse_te, 'RMSE': np.sqrt(mse_te), 'R2': r2_te
            })

        # Plots
        fold_dir = os.path.join(IMG_DIR, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        plot_parity(Y_tr_real, Y_pred_tr, y_cols, os.path.join(fold_dir, 'parity_train.png'), f"Fold {fold+1} Train")
        plot_parity(Y_te, Y_pred_te, y_cols, os.path.join(fold_dir, 'parity_test.png'), f"Fold {fold+1} Test")

    kfold_df = pd.DataFrame(kfold_detailed_results)
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
    final_model = train_random_forest_model(X_s, Y_s, hp)
    train_time = time.time() - start_time
    print(f"   - Training Time: {train_time:.2f} seconds")

    # 5. Generate Predictions & Plots
    Y_pred_s_full = final_model.predict(X_s)
    Y_pred_full = np.exp(sc_y_final.inverse_transform(Y_pred_s_full)) - 1
    
    print("   - Generating Final Parity Plot...")
    plot_parity(Y, Y_pred_full, y_cols, os.path.join(IMG_DIR, 'parity_plot_final_model.png'), "Final Model")
    
    print("   - Calculating Feature Importance...")
    imp_df = calculate_rf_importance(final_model, x_cols)

    # 6. Regime Generalization Analysis
    regime_df = analyze_regime_generalization(X, Y, Y_pred_full, x_cols, y_cols)

    # 7. Save Results
    print(f"\n5. Saving results to {OUTPUT_FILE_PATH}...")
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
        
        pd.concat([X, Y], axis=1).to_excel(writer, sheet_name='Calibration_Data', index=False)
        pd.DataFrame(Y_pred_full, columns=y_cols).to_excel(writer, sheet_name='Calibration_Preds', index=False)

    joblib.dump({
        'model': final_model, 'x_scaler': sc_x_final, 'y_scaler': sc_y_final,
        'x_cols': x_cols, 'y_cols': y_cols,
        'hyperparameters': hp
    }, MODEL_PATH)

    print("Done.")

if __name__ == '__main__':
    main()
