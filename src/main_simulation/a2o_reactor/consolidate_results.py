import pandas as pd
import numpy as np
import os

def create_consolidated_sheets():
    """
    Reads simulation results from 'results_output' and 'results_input' sheets
    in an Excel file, processes the data, and creates four new sheets:
    'consolidated_input', 'consolidated_output', 'consolidated_input_c1',
    and 'consolidated_output_c1'.
    """
    # --- 1. SETUP: Define file paths and column names ---
    filepath = os.path.join('data', 'data.xlsx')
    
    # Check if the source file exists
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        print("Please run the main simulation code first to generate the data file.")
        return

    # --- Column Definitions for A2/O Reactors ---
    consolidated_input_cols = [
        'simulation_number', 'flow_rate', 'inf_S_I', 'inf_X_I', 'inf_S_F', 'inf_S_A', 
        'inf_X_S', 'inf_S_NH4', 'inf_S_N2', 'inf_S_NO3', 'inf_S_PO4', 'inf_X_PP', 
        'inf_X_PHA', 'inf_X_H', 'inf_X_AUT', 'inf_X_PAO', 'inf_S_ALK', 'V', 'KLa'
    ]

    # This list is used for the output of both A2/O reactors and the C1 clarifier
    consolidated_output_cols = [
        'Target_Effluent_S_O2 (mg/L)', 'Target_Effluent_S_N2 (mg/L)', 
        'Target_Effluent_S_NH4 (mg/L)', 'Target_Effluent_S_NO3 (mg/L)', 
        'Target_Effluent_S_PO4 (mg/L)', 'Target_Effluent_S_F (mg/L)', 
        'Target_Effluent_S_A (mg/L)', 'Target_Effluent_S_I (mg/L)', 
        'Target_Effluent_S_ALK (mg/L)', 'Target_Effluent_X_I (mg/L)', 
        'Target_Effluent_X_S (mg/L)', 'Target_Effluent_X_H (mg/L)', 
        'Target_Effluent_X_PAO (mg/L)', 'Target_Effluent_X_PP (mg/L)', 
        'Target_Effluent_X_PHA (mg/L)', 'Target_Effluent_X_AUT (mg/L)', 
        'Target_Effluent_X_MeOH (mg/L)', 'Target_Effluent_X_MeP (mg/L)', 
        'Target_Effluent_H2O (mg/L)', 'Target_Effluent_COD (mg/L)', 
        'Target_Effluent_BOD (mg/L)', 'Target_Effluent_TN (mg/L)', 
        'Target_Effluent_TKN (mg/L)', 'Target_Effluent_TP (mg/L)', 
        'Target_Effluent_TSS (mg/L)', 'Target_Effluent_VSS (mg/L)'
    ]

    # --- REVISED: Column Definitions for C1 Clarifier ---
    consolidated_input_c1_cols = [
        'simulation_number', 'flow_rate', 'inf_S_I', 'inf_X_I', 'inf_S_F', 'inf_S_A', 
        'inf_X_S', 'inf_S_NH4', 'inf_S_N2', 'inf_S_NO3', 'inf_S_PO4', 'inf_X_PP', 
        'inf_X_PHA', 'inf_X_H', 'inf_X_AUT', 'inf_X_PAO', 'inf_S_ALK', 
        'C1_surface_area', 'C1_height', 'Q_was', 'Q_ext', 'O3_split_internal'
    ]
    # The output columns for C1 are the same as for the reactors, so we reuse `consolidated_output_cols`.

    # --- 2. LOAD DATA: Read the results from the Excel file ---
    print(f"Loading data from '{filepath}'...")
    try:
        df_input = pd.read_excel(filepath, sheet_name='results_input')
        df_output = pd.read_excel(filepath, sheet_name='results_output')
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Please ensure the file exists in the 'data' directory.")
        return
    except ValueError as e:
        if "sheet" in str(e).lower():
            print(f"Error: A required sheet ('results_input' or 'results_output') is missing from '{filepath}'.")
            print("Please ensure the simulation has been run and the sheets were created correctly.")
        else:
            print(f"An error occurred while reading the Excel file: {e}")
        return

    # Merge input and output dataframes for easier access during processing
    df_merged = pd.merge(df_input, df_output, on='simulation_number', how='inner')
    print(f"Found {len(df_merged)} simulations to process.")

    # --- 3. PROCESS DATA (A2/O Reactors): Transform and consolidate reactor results ---
    
    # Define mappings from generic unit to specific data sources
    units_to_process = ['A2', 'O1', 'O2', 'O3']
    input_source_prefix = {'A2': 'A1_eff', 'O1': 'A2_eff', 'O2': 'O1_eff', 'O3': 'O2_eff'}
    output_source_prefix = {'A2': 'A2_eff', 'O1': 'O1_eff', 'O2': 'O2_eff', 'O3': 'O3_to_C1'}
    volume_param = {'A2': 'V_an', 'O1': 'V_ae', 'O2': 'V_ae', 'O3': 'V_ae'}
    kla_param = {'A2': None, 'O1': 'KLa_aer1', 'O2': 'KLa_aer1', 'O3': 'KLa_aer2'}

    consolidated_input_rows = []
    consolidated_output_rows = []

    print("Consolidating data for units A2, O1, O2, and O3...")
    for _, sim_row in df_merged.iterrows():
        for unit in units_to_process:
            # Build the input row for the current unit
            input_row = {}
            input_row['flow_rate'] = sim_row.get('flow_rate')
            input_row['V'] = sim_row.get(volume_param[unit])
            kla_key = kla_param[unit]
            input_row['KLa'] = sim_row.get(kla_key) if kla_key else 0 # Use 0 for anaerobic KLa
            for col_name in consolidated_input_cols:
                if col_name.startswith('inf_'):
                    component = col_name.replace('inf_', '')
                    source_col = f"Target_{input_source_prefix[unit]}_{component} (mg/L)"
                    input_row[col_name] = sim_row.get(source_col)
            consolidated_input_rows.append(pd.Series(input_row).fillna(0).to_dict())

            # Build the output row for the current unit
            output_row = {}
            for col_name in consolidated_output_cols:
                component = col_name.replace('Target_Effluent_', '').replace(' (mg/L)', '')
                source_col = f"Target_{output_source_prefix[unit]}_{component} (mg/L)"
                output_row[col_name] = sim_row.get(source_col)
            consolidated_output_rows.append(pd.Series(output_row).fillna(0).to_dict())

    # --- 4. REVISED: PROCESS DATA (C1 Clarifier): Transform and consolidate clarifier results ---
    
    consolidated_input_c1_rows = []
    consolidated_output_c1_rows = []

    print("Consolidating data for unit C1...")
    for _, sim_row in df_merged.iterrows():
        # --- Build the input row for C1 ---
        input_c1_row = {}
        # Fetch direct parameters from the input results
        input_c1_row['flow_rate'] = sim_row.get('flow_rate')
        input_c1_row['C1_surface_area'] = sim_row.get('C1_surface_area')
        input_c1_row['C1_height'] = sim_row.get('C1_height')
        input_c1_row['Q_was'] = sim_row.get('Q_was')
        input_c1_row['Q_ext'] = sim_row.get('Q_ext')
        input_c1_row['O3_split_internal'] = sim_row.get('O3_split_internal')

        # Fetch influent concentrations from the effluent of O3 (which is the input to C1)
        for col_name in consolidated_input_c1_cols:
            if col_name.startswith('inf_'):
                component = col_name.replace('inf_', '')
                # The input to C1 is the stream named 'O3_to_C1'
                source_col = f"Target_O3_to_C1_{component} (mg/L)"
                input_c1_row[col_name] = sim_row.get(source_col)
        consolidated_input_c1_rows.append(pd.Series(input_c1_row).fillna(0).to_dict())

        # --- Build the output row for C1 ---
        output_c1_row = {}
        # Fetch effluent concentrations from the final effluent stream (Target_Effluent)
        for col_name in consolidated_output_cols:
            # The source column name in df_merged is identical to the desired column name
            output_c1_row[col_name] = sim_row.get(col_name)
        consolidated_output_c1_rows.append(pd.Series(output_c1_row).fillna(0).to_dict())

    # --- 5. CREATE DATAFRAMES: Convert all lists of rows into DataFrames ---
    if not consolidated_input_rows:
        print("No reactor data was processed. The 'consolidated_input' and 'consolidated_output' sheets will not be created.")
    else:
        # Create DataFrames for A2/O reactors
        df_consolidated_input = pd.DataFrame(consolidated_input_rows)
        df_consolidated_output = pd.DataFrame(consolidated_output_rows)
        # Add the new continuous simulation number
        new_sim_numbers = range(1, len(df_consolidated_input) + 1)
        df_consolidated_input['simulation_number'] = new_sim_numbers
        df_consolidated_output['simulation_number'] = new_sim_numbers
        # Reorder columns to match the specified format
        df_consolidated_input = df_consolidated_input[consolidated_input_cols]
        output_cols_with_sim = ['simulation_number'] + consolidated_output_cols
        df_consolidated_output = df_consolidated_output.reindex(columns=output_cols_with_sim)

    if not consolidated_input_c1_rows:
        print("No C1 data was processed. The 'consolidated_input_c1' and 'consolidated_output_c1' sheets will not be created.")
    else:
        # Create DataFrames for C1 clarifier
        df_consolidated_input_c1 = pd.DataFrame(consolidated_input_c1_rows)
        df_consolidated_output_c1 = pd.DataFrame(consolidated_output_c1_rows)
        # Add the new continuous simulation number (1 per original simulation)
        new_sim_numbers_c1 = range(1, len(df_consolidated_input_c1) + 1)
        df_consolidated_input_c1['simulation_number'] = new_sim_numbers_c1
        df_consolidated_output_c1['simulation_number'] = new_sim_numbers_c1
        # Reorder columns to match the specified format
        df_consolidated_input_c1 = df_consolidated_input_c1[consolidated_input_c1_cols]
        df_consolidated_output_c1 = df_consolidated_output_c1.reindex(columns=output_cols_with_sim)

    # --- 6. SAVE DATA: Write all four new DataFrames to the Excel file ---
    print(f"Saving consolidated sheets to '{filepath}'...")
    try:
        # Use ExcelWriter in append mode to add/replace sheets without touching others
        with pd.ExcelWriter(filepath, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            if 'df_consolidated_input' in locals():
                df_consolidated_input.to_excel(writer, sheet_name='consolidated_input', index=False)
                df_consolidated_output.to_excel(writer, sheet_name='consolidated_output', index=False)
                print("Successfully created/updated 'consolidated_input' and 'consolidated_output' sheets.")
            
            if 'df_consolidated_input_c1' in locals():
                df_consolidated_input_c1.to_excel(writer, sheet_name='consolidated_input_c1', index=False)
                df_consolidated_output_c1.to_excel(writer, sheet_name='consolidated_output_c1', index=False)
                print("Successfully created/updated 'consolidated_input_c1' and 'consolidated_output_c1' sheets.")
            
        print("File saving process complete.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == '__main__':
    # Create a dummy data folder and file for testing if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(os.path.join('data', 'data.xlsx')):
        print("Creating a dummy 'data.xlsx' file for demonstration purposes.")
        # Create dummy dataframes that mimic the structure of the real simulation output
        sim_count = 5
        # Dummy Input Data
        input_data = {
            'simulation_number': range(1, sim_count + 1),
            'flow_rate': [1000] * sim_count, 'V_an': [1500] * sim_count, 'V_ae': [3000] * sim_count,
            'KLa_aer1': [240] * sim_count, 'KLa_aer2': [120] * sim_count, 'C1_surface_area': [500] * sim_count,
            'C1_height': [4] * sim_count, 'Q_was': [50] * sim_count, 'Q_ext': [500] * sim_count,
            'O3_split_internal': [0.6] * sim_count
        }
        df_dummy_input = pd.DataFrame(input_data)
        
        # Dummy Output Data
        output_data = {'simulation_number': range(1, sim_count + 1)}
        # Generate dummy columns for all process units and final effluent
        units = ['A1_eff', 'A2_eff', 'O1_eff', 'O2_eff', 'O3_to_C1', 'Effluent']
        components = ['S_O2','S_N2','S_NH4','S_NO3','S_PO4','S_F','S_A','S_I','S_ALK','X_I','X_S','X_H','X_PAO','X_PP','X_PHA','X_AUT','X_MeOH','X_MeP','H2O','COD','BOD','TN','TKN','TP','TSS','VSS']
        for unit in units:
            for comp in components:
                # Use different random values for each unit to simulate changes
                col_name = f"Target_{unit}_{comp} (mg/L)".replace('_Effluent', '')
                output_data[col_name] = np.random.rand(sim_count) * 10
        df_dummy_output = pd.DataFrame(output_data)

        with pd.ExcelWriter(os.path.join('data', 'data.xlsx'), engine='openpyxl') as writer:
            df_dummy_input.to_excel(writer, sheet_name='results_input', index=False)
            df_dummy_output.to_excel(writer, sheet_name='results_output', index=False)
    
    create_consolidated_sheets()