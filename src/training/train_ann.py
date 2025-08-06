# Install required libraries (uncomment if needed)
# !pip install pandas numpy scikit-learn torch openpyxl joblib tqdm

import pandas as pd
import numpy as np
import os
import warnings
import math
import joblib
from tqdm import tqdm

# --- PyTorch Imports for ANN ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Scikit-learn Imports ---
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

# --- Configuration ---
FILE_PATH = os.path.join('data', 'data.xlsx')
MACHINE_LEARNING_MODEL = 'ann'
OUTPUT_FILE_PATH = os.path.join('data', 'training_data', MACHINE_LEARNING_MODEL, 'training_stat_' + process_unit, process_unit + '_train_stat.xlsx')
MODEL_PATH = os.path.join('models', MACHINE_LEARNING_MODEL, process_unit, process_unit + '.joblib')
RANDOM_STATE = 42

# --- PyTorch & Model Hyperparameters ---
NUM_EPOCHS = 3000
LEARNING_RATE = 0.001
BATCH_SIZE = -1 # Set to > 0 for mini-batch, <= 0 for full-batch

# --- REVISED: Get N_SPLITS from user ---
try:
    N_SPLITS = int(input("Enter the number of folds for cross-validation (default: 10): ") or 10)
except ValueError:
    print("Invalid input. Using default value of 5.")
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

# --- PyTorch Dataset and Model Definitions ---
class WastewaterDataset(Dataset):
    """Simple PyTorch Dataset for the ANN model."""
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class ANNModel(nn.Module):
    """
    A standard Multi-Layer Perceptron (MLP) for regression.
    The architecture is: Input -> Linear -> ReLU -> Linear -> ReLU -> Output
    """
    def __init__(self, n_indep, n_dep):
        super().__init__()
        self.n_dep = n_dep
        self.layers = nn.Sequential(
            nn.Linear(n_indep, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_dep)
        )

    def forward(self, X):
        return self.layers(X)

# --- Training and Prediction Functions ---
def train_ann_model(X_train, Y_train, n_epochs, batch_size, learning_rate):
    """Trains the ANNModel using PyTorch."""
    n_samples, n_indep = X_train.shape
    n_dep = Y_train.shape[1]

    model = ANNModel(n_indep, n_dep).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = WastewaterDataset(X_train, Y_train)
    model.train()

    # Use a generic progress bar description
    pbar_desc = "Training ANN"
    if batch_size <= 0:
        X_gpu = train_dataset.X.to(device)
        Y_gpu = train_dataset.Y.to(device)

        for epoch in range(n_epochs):
            Y_pred = model(X_gpu)
            loss = criterion(Y_pred, Y_gpu)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(n_epochs):
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model

def predict_ann(model, X_data):
    """Makes predictions using the trained ANN model."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
        Y_pred_tensor = model(X_tensor)
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

    if X.empty or Y.empty:
        print("Error: No data left after filtering components. Check 'training_components' sheet.")
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

        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(Y_log_train)

        X_train_s = x_scaler.transform(X_train)
        X_test_s = x_scaler.transform(X_test)
        Y_log_train_s = y_scaler.transform(Y_log_train)

        model_fold = train_ann_model(X_train_s, Y_log_train_s, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)

        Y_pred_test_s = predict_ann(model_fold, X_test_s)
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
    final_x_scaler = StandardScaler().fit(X)
    final_y_scaler = StandardScaler().fit(Y_log)

    X_s_full = final_x_scaler.transform(X)
    Y_log_s_full = final_y_scaler.transform(Y_log)

    final_model = train_ann_model(X_s_full, Y_log_s_full, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
    print("   - Final model training complete.")

    # --- Saving Final Model and Results ---
    print("\n4. Saving final model, scalers, and results...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    final_model.to('cpu')
    model_bundle = {
        'model': final_model,
        'x_scaler': final_x_scaler,
        'y_scaler': final_y_scaler
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"   - Model bundle saved to {MODEL_PATH}")

    # --- Evaluate Final Model on Full Dataset for Reporting ---
    print("\n5. Evaluating final model on full dataset for reporting...")
    Y_pred_full_s = predict_ann(final_model, X_s_full)
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
        pd.DataFrame(Y_pred_full, columns=y_cols, index=Y.index).to_excel(writer, sheet_name='calibration_prediction')
        pd.DataFrame().to_excel(writer, sheet_name='validation_data', index=False)
        pd.DataFrame().to_excel(writer, sheet_name='validation_statistics', index=False)
        pd.DataFrame().to_excel(writer, sheet_name='validation_prediction', index=False)

    print(f"\nProcess finished successfully. Results are saved in '{OUTPUT_FILE_PATH}'.")

if __name__ == '__main__':
    main()