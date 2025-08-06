import pandas as pd
import os

# --- Configuration ---
excel_file_path = 'data/data.xlsx'
output_directory = os.path.dirname(excel_file_path)

# --- Ensure you have the required libraries ---
# In your terminal, run:
# pip install pandas openpyxl pyarrow

def convert_excel_to_parquet(file_path, out_dir):
    """
    Reads specific sheets from an Excel file and saves them as Parquet files.
    """
    sheets_to_convert = {
        'input_config': 'input_config.parquet',
        'initial_conditions': 'initial_conditions.parquet'
    }

    print(f"Reading from Excel file: {file_path}")

    try:
        with pd.ExcelFile(file_path) as xls:
            for sheet_name, parquet_filename in sheets_to_convert.items():
                if sheet_name in xls.sheet_names:
                    print(f"Converting sheet: '{sheet_name}'...")
                    df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
                    
                    # For input_config, we need to reset the index to make the parameter names a column
                    if sheet_name == 'input_config':
                        df = df.reset_index().rename(columns={'index': 'parameter'})

                    output_path = os.path.join(out_dir, parquet_filename)
                    df.to_parquet(output_path)
                    print(f"Successfully saved '{parquet_filename}' to '{out_dir}'")
                else:
                    print(f"Warning: Sheet '{sheet_name}' not found in the Excel file.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    convert_excel_to_parquet(excel_file_path, output_directory)