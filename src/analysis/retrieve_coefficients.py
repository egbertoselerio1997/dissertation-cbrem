import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
import itertools
import warnings
import re

# --- PyTorch Class Definition (Required for loading the joblib file) ---
# This class structure must be defined for joblib to successfully unpickle the
# custom model object stored within the .joblib file.
try:
    import torch
    import torch.nn as nn

    class CoupledCLEFOModel(nn.Module):
        """
        A placeholder class that matches the structure of the trained model.
        Required by joblib to deserialize the model file correctly.
        """
        def __init__(self, n_dep, n_indep, n_inter):
            super().__init__()
            self.n_dep = None
            self.Upsilon = None
            self.B = None
            self.Theta = None
            self.Gamma = None
            self.Lambda = None

        def forward(self, X, Z):
            # The forward pass is not needed for coefficient analysis.
            return None

except ImportError:
    print("Warning: PyTorch not found. Using a dummy placeholder class for model unpickling.")
    class CoupledCLEFOModel:
        """A dummy placeholder class if PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            pass

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
from naming import canonical_base_name, normalize_stream_column, rename_concentration_columns

# Suppress openpyxl warnings for a cleaner output.
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def get_feature_and_target_names(filepath: str, process_unit: str):
    """
    Loads data from the specified Excel file to extract feature and target column names.
    This function mimics the data loading and filtering logic of the original script
    to ensure the correct set of columns is retrieved.

    Args:
        filepath (str): The path to the main data Excel file.
        process_unit (str): The process unit type ('clarifier', 'cstr', or 'aao').

    Returns:
        tuple: A tuple containing the list of feature names and target variable names.
               Returns (None, None) if data loading fails or data is empty.
    """
    print("1. Retrieving feature and target names...")
    try:
        df_input = pd.read_excel(filepath, sheet_name=f"all_input_{process_unit}")
        df_output = pd.read_excel(filepath, sheet_name=f"all_output_{process_unit}")
        df_output = rename_concentration_columns(df_output)
    except FileNotFoundError:
        print(f"Error: The data file was not found at '{filepath}'")
        return None, None
    except ValueError as e:
        print(f"Error: A sheet for '{process_unit}' could not be found in '{filepath}'. {e}")
        return None, None

    components_to_remove = []
    try:
        df_components = pd.read_excel(filepath, sheet_name="training_components")
        if 'components' in df_components.columns and 'considered' in df_components.columns:
            components_to_remove = df_components[df_components['considered'] == 0]['components'].tolist()
    except Exception:
        pass

    if {'variable', 'default'}.issubset(df_input.columns):
        df_input['variable'] = df_input['variable'].apply(normalize_stream_column)
        input_cols = df_input['variable'].unique().tolist()
        df_input_wide = df_input.pivot(index='simulation_number', columns='variable', values='default').reset_index()
    else:
        df_input_wide = rename_concentration_columns(df_input)
        input_cols = [col for col in df_input_wide.columns if col != 'simulation_number']

    data = pd.merge(df_input_wide, df_output, on='simulation_number', how='inner')
    
    y_cols_all = [col for col in data.columns if col.startswith(('effluent_', 'wastage_'))]
    
    if components_to_remove:
        cols_to_drop = []
        for comp in components_to_remove:
            base_token = canonical_base_name(comp) or comp
            cols_to_drop.extend([c for c in input_cols if base_token in c])
            cols_to_drop.extend([c for c in y_cols_all if base_token in c or comp in c])
        data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    y_cols = [col for col in data.columns if col.startswith(('effluent_', 'wastage_'))]
    current_input_cols = [col for col in data.columns if col in input_cols]
    inf_cols = sorted([col for col in current_input_cols if col.startswith('influent_')])
    proc_cols = sorted([col for col in current_input_cols if not col.startswith('influent_')])
    x_cols_ordered = proc_cols + inf_cols

    if not x_cols_ordered or not y_cols:
        print("Error: No features or targets found after filtering.")
        return None, None
        
    print(f"   - Found {len(x_cols_ordered)} features and {len(y_cols)} target variables.")
    return x_cols_ordered, y_cols

def analyze_model_coefficients():
    """
    Main function to load a trained model, extract its coefficients, rank them
    by absolute value, and export them to a multi-sheet Excel file.
    """
    valid_units = ['clarifier', 'cstr', 'aao']
    prompt_text = f"Enter the process unit for the model to analyze ('clarifier', 'cstr', or 'aao'): "
    process_unit = ''
    while process_unit not in valid_units:
        process_unit = input(prompt_text).lower().strip()
        if process_unit not in valid_units:
            print(f"Invalid input. Please enter one of {valid_units}.")

    MODEL_PATH = os.path.join('data', 'results', 'training', 'clefo', process_unit, f'{process_unit}.joblib')
    DATA_FILE_PATH = os.path.join('data', 'config', 'simulation_training_config.xlsx')
    
    # The output Excel file will be saved in the same directory as the model.
    model_directory = os.path.dirname(MODEL_PATH)
    OUTPUT_EXCEL_PATH = os.path.join(model_directory, 'coefficients.xlsx')

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        return

    x_cols, y_cols = get_feature_and_target_names(DATA_FILE_PATH, process_unit)
    if not x_cols or not y_cols:
        return

    print(f"\n2. Loading model bundle from '{MODEL_PATH}'...")
    try:
        model_bundle = joblib.load(MODEL_PATH)
        print("   - Model bundle loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load the joblib file. {e}")
        return

    print("3. Extracting coefficient matrices from the bundle...")
    B = model_bundle.get('B')
    Theta = model_bundle.get('Theta')

    if B is None or Theta is None:
        print("Error: Coefficient matrices 'B' or 'Theta' not found in the model bundle.")
        return

    independent_feature_names = x_cols
    interaction_feature_names = [f"{pair[0]} * {pair[1]}" for pair in itertools.combinations(x_cols, 2)]
    
    all_feature_names = independent_feature_names + interaction_feature_names
    all_coeffs = np.hstack((B, Theta))
    print(f"   - Extracted {all_coeffs.shape[1]} total coefficients for each of the {all_coeffs.shape[0]} targets.")

    print(f"\n4. Sorting features and writing to Excel file at '{OUTPUT_EXCEL_PATH}'...")
    
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL_PATH, engine='xlsxwriter') as writer:
            for i, target_name in enumerate(y_cols):
                target_coeffs = all_coeffs[i, :]

                # Create a DataFrame with Feature, the actual Coefficient, and its Absolute Value
                df_coeffs = pd.DataFrame({
                    'Feature': all_feature_names,
                    'Coefficient': target_coeffs,
                    'Absolute Coefficient': np.abs(target_coeffs)
                })

                # Sort the DataFrame by the 'Absolute Coefficient' in descending order
                df_sorted = df_coeffs.sort_values(by='Absolute Coefficient', ascending=False).reset_index(drop=True)
                
                # Sanitize the target name to create a valid Excel sheet name
                sanitized_name = re.sub(r'[\\/*?:[\]]', '', target_name)
                sheet_name = sanitized_name[:31]
                
                # Write the sorted DataFrame to a worksheet named after the target variable
                df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print("\nProcess finished successfully!")
        print(f"The ranked coefficients have been saved to '{OUTPUT_EXCEL_PATH}'")

    except Exception as e:
        print(f"An error occurred while writing the Excel file: {e}")


if __name__ == '__main__':
    analyze_model_coefficients()
