import pandas as pd
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
# Map folder names to display names if necessary
# Keys match the MACHINE_LEARNING_MODEL variable in previous scripts
MODELS = [
    'clefo',          # CBRE
    'pls',
    'ann',
    'gam',
    'glm',
    'knn',
    'lightgbm',
    'random_forest',
    'svr',
    'xgboost'
]

UNITS = ['cstr', 'clarifier']

# The specific sheets we want to aggregate
TARGET_SHEETS = [
    'CV_Summary',
    'Final_Model_Stats',
    'Feature_Importance',
    'Comp_Cost',
    'Regime_Analysis'
]

# Paths
BASE_DIR = os.path.join('data', 'training_data')
OUTPUT_FILE = os.path.join(BASE_DIR, 'training_result_compilation.xlsx')

def flatten_multiindex_columns(df):
    """
    Flattens MultiIndex columns (e.g., R2 -> mean, std) into single strings (R2_mean, R2_std).
    Used specifically for CV_Summary.
    """
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            # Skip the index column if it was reset
            if col[0] == 'Variable' or col[0] == 'Unnamed: 0_level_0':
                new_cols.append('Variable')
            else:
                # Join non-empty levels
                new_cols.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
        df.columns = new_cols
    return df

def main():
    print("Starting Training Result Aggregation...")
    print(f"Looking for data in: {BASE_DIR}")

    # Dictionary to hold lists of dataframes for each sheet type
    aggregated_data = {sheet: [] for sheet in TARGET_SHEETS}

    found_count = 0
    missing_count = 0

    for unit in UNITS:
        for model in MODELS:
            # Construct the expected path based on the previous scripts' logic
            # Path: data/training_data/{MODEL}/training_stat_{UNIT}/{UNIT}_train_stat.xlsx
            file_path = os.path.join(BASE_DIR, model, f'training_stat_{unit}', f'{unit}_train_stat.xlsx')

            if os.path.exists(file_path):
                print(f"Processing: {model.upper()} - {unit.upper()}")
                found_count += 1
                
                try:
                    # Load the Excel file
                    xls = pd.ExcelFile(file_path)
                    
                    for sheet_name in TARGET_SHEETS:
                        if sheet_name in xls.sheet_names:
                            # Special handling for CV_Summary which has MultiIndex headers
                            if sheet_name == 'CV_Summary':
                                df = pd.read_excel(xls, sheet_name=sheet_name, header=[0, 1])
                                df = flatten_multiindex_columns(df)
                                # Clean up potential index artifacts
                                if 'Variable' in df.columns:
                                    pass # Good
                                elif df.index.name == 'Variable':
                                    df = df.reset_index()
                                else:
                                    # Sometimes read_excel treats the first column as index implicitly
                                    # Check if first column looks like variable names
                                    df = df.rename(columns={df.columns[0]: 'Variable'})
                            else:
                                df = pd.read_excel(xls, sheet_name=sheet_name)

                            # Add Metadata Columns
                            # Insert at the beginning
                            df.insert(0, 'Unit', unit.upper())
                            df.insert(0, 'Model', 'CBRE' if model == 'clefo' else model.upper())

                            aggregated_data[sheet_name].append(df)
                        else:
                            # Regime Analysis might not exist for all models/datasets if KLa is missing
                            if sheet_name != 'Regime_Analysis': 
                                print(f"  [Warning] Sheet '{sheet_name}' missing in {file_path}")

                except Exception as e:
                    print(f"  [Error] Failed to read {file_path}: {e}")
            else:
                # print(f"  [Missing] File not found: {file_path}")
                missing_count += 1

    print("-" * 30)
    print(f"Aggregation Complete. Found {found_count} files. Missing {missing_count} files.")

    # --- Save to Excel ---
    if found_count > 0:
        print(f"Saving compiled results to: {OUTPUT_FILE}")
        with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
            for sheet_name, dfs in aggregated_data.items():
                if dfs:
                    # Concatenate all dataframes for this sheet type
                    combined_df = pd.concat(dfs, ignore_index=True)
                    
                    # Formatting: Move Model/Unit to front if concat messed order (sanity check)
                    cols = list(combined_df.columns)
                    if 'Model' in cols and 'Unit' in cols:
                        cols.remove('Model')
                        cols.remove('Unit')
                        cols = ['Model', 'Unit'] + cols
                        combined_df = combined_df[cols]

                    combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  - Sheet '{sheet_name}' saved with {len(combined_df)} rows.")
                else:
                    print(f"  - Sheet '{sheet_name}' is empty (no data found).")
        print("Done.")
    else:
        print("No data found to aggregate.")

if __name__ == "__main__":
    main()