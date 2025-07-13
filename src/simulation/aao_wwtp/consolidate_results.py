import pandas as pd
import numpy as np
import os

def create_all_sheets():
    """
    Reads simulation results from 'results_output' and 'results_input' sheets
    in an Excel file, processes the data, and creates four new sheets:
    'all_input_cstr', 'all_output_cstr', 'all_input_clarifier',
    and 'all_output_clarifier'.
    """
    # --- 1. SETUP: Define file paths and column names ---
    filepath = os.path.join('data', 'data.xlsx')
    
    # Check if the source file exists
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        print("Please run the main simulation code first to generate the data file.")
        return

    # --- Column Definitions for A2/O Reactors ---
    all_input_cstr_cols = [
        'simulation_number', 'flow_rate', 'inf_S_I', 'inf_X_I', 'inf_S_F', 'inf_S_A', 
        'inf_X_S', 'inf_S_NH4', 'inf_S_N2', 'inf_S_NO3', 'inf_S_PO4', 'inf_X_PP', 
        'inf_X_PHA', 'inf_X_H', 'inf_X_AUT', 'inf_X_PAO', 'inf_S_ALK', 'V', 'KLa'
    ]

    all_output_cstr_cols = [
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
    all_input_clarifier_cols = [
        'simulation_number', 'flow_rate', 'inf_S_I', 'inf_X_I', 'inf_S_F', 'inf_S_A', 
        'inf_X_S', 'inf_S_NH4', 'inf_S_N2', 'inf_S_NO3', 'inf_S_PO4', 'inf_X_PP', 
        'inf_X_PHA', 'inf_X_H', 'inf_X_AUT', 'inf_X_PAO', 'inf_S_ALK', 
        'C1_surface_area', 'C1_height', 'Q_was', 'Q_ext', 'O3_split_internal'
    ]
    
    all_output_clarifier_cols = [
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
        'Target_Effluent_TSS (mg/L)', 'Target_Effluent_VSS (mg/L)',
        'Target_Wastage_S_O2 (mg/L)', 'Target_Wastage_S_N2 (mg/L)',
        'Target_Wastage_S_NH4 (mg/L)', 'Target_Wastage_S_NO3 (mg/L)',
        'Target_Wastage_S_PO4 (mg/L)', 'Target_Wastage_S_F (mg/L)',
        'Target_Wastage_S_A (mg/L)', 'Target_Wastage_S_I (mg/L)',
        'Target_Wastage_S_ALK (mg/L)', 'Target_Wastage_X_I (mg/L)',
        'Target_Wastage_X_S (mg/L)', 'Target_Wastage_X_H (mg/L)',
        'Target_Wastage_X_PAO (mg/L)', 'Target_Wastage_X_PP (mg/L)',
        'Target_Wastage_X_PHA (mg/L)', 'Target_Wastage_X_AUT (mg/L)',
        'Target_Wastage_X_MeOH (mg/L)', 'Target_Wastage_X_MeP (mg/L)',
        'Target_Wastage_H2O (mg/L)', 'Target_Wastage_COD (mg/L)',
        'Target_Wastage_BOD (mg/L)', 'Target_Wastage_TN (mg/L)',
        'Target_Wastage_TKN (mg/L)', 'Target_Wastage_TP (mg/L)',
        'Target_Wastage_TSS (mg/L)', 'Target_Wastage_VSS (mg/L)'
    ]

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
    
    units_to_process = ['A2', 'O1', 'O2', 'O3']
    input_source_prefix = {'A2': 'A1_eff', 'O1': 'A2_eff', 'O2': 'O1_eff', 'O3': 'O2_eff'}
    output_source_prefix = {'A2': 'A2_eff', 'O1': 'O1_eff', 'O2': 'O2_eff', 'O3': 'O3_to_C1'}
    volume_param = {'A2': 'V_A2', 'O1': 'V_O1', 'O2': 'V_O2', 'O3': 'V_O3'}
    kla_param = {'A2': 'KLa_A2', 'O1': 'KLa_O1', 'O2': 'KLa_O2', 'O3': 'KLa_O3'}

    all_input_cstr_rows = []
    all_output_cstr_rows = []

    print("Consolidating data for units A2, O1, O2, and O3...")
    for _, sim_row in df_merged.iterrows():
        for unit in units_to_process:
            input_cstr_row = {'flow_rate': sim_row.get('flow_rate'), 'V': sim_row.get(volume_param[unit]),
                              'KLa': sim_row.get(kla_param[unit]) if kla_param[unit] else 0,
                              'Q_ext': sim_row.get('Q_ext'), 'O3_split_internal': sim_row.get('O3_split_internal')}
            for col_name in all_input_cstr_cols:
                if col_name.startswith('inf_'):
                    component = col_name.replace('inf_', '')
                    source_col = f"Target_{input_source_prefix[unit]}_{component} (mg/L)"
                    input_cstr_row[col_name] = sim_row.get(source_col)
            all_input_cstr_rows.append(pd.Series(input_cstr_row).fillna(0).to_dict())

            output_cstr_row = {}
            for col_name in all_output_cstr_cols:
                component = col_name.replace('Target_Effluent_', '').replace(' (mg/L)', '')
                source_col = f"Target_{output_source_prefix[unit]}_{component} (mg/L)"
                output_cstr_row[col_name] = sim_row.get(source_col)
            all_output_cstr_rows.append(pd.Series(output_cstr_row).fillna(0).to_dict())

    # --- 4. REVISED: PROCESS DATA (C1 Clarifier): Transform and consolidate clarifier results ---
    
    all_input_clarifier_rows = []
    all_output_clarifier_rows = []

    print("Consolidating data for unit C1...")
    for _, sim_row in df_merged.iterrows():
        input_clarifier_row = {
            'flow_rate': sim_row.get('flow_rate'), 'C1_surface_area': sim_row.get('C1_surface_area'),
            'C1_height': sim_row.get('C1_height'), 'Q_was': sim_row.get('Q_was'),
            'Q_ext': sim_row.get('Q_ext'), 'O3_split_internal': sim_row.get('O3_split_internal')}
        for col_name in all_input_clarifier_cols:
            if col_name.startswith('inf_'):
                component = col_name.replace('inf_', '')
                source_col = f"Target_O3_to_C1_{component} (mg/L)"
                input_clarifier_row[col_name] = sim_row.get(source_col)
        all_input_clarifier_rows.append(pd.Series(input_clarifier_row).fillna(0).to_dict())

        # --- Build the output row for C1 using the extended clarifier column list ---
        output_clarifier_row = {}
        # Fetch concentrations from the final effluent and wastage streams
        for col_name in all_output_clarifier_cols:
            # The source column name in df_merged is identical to the desired column name
            output_clarifier_row[col_name] = sim_row.get(col_name)
        all_output_clarifier_rows.append(pd.Series(output_clarifier_row).fillna(0).to_dict())

    # --- 5. CREATE DATAFRAMES: Convert all lists of rows into DataFrames ---
    if not all_input_cstr_rows:
        print("No reactor data was processed. The 'all_input_cstr' and 'all_output_cstr' sheets will not be created.")
    else:
        df_all_input = pd.DataFrame(all_input_cstr_rows)
        df_all_output = pd.DataFrame(all_output_cstr_rows)
        new_sim_numbers = range(1, len(df_all_input) + 1)
        df_all_input['simulation_number'] = new_sim_numbers
        df_all_output['simulation_number'] = new_sim_numbers
        df_all_input = df_all_input[all_input_cstr_cols]
        output_cstr_cols_with_sim = ['simulation_number'] + all_output_cstr_cols
        df_all_output = df_all_output.reindex(columns=output_cstr_cols_with_sim)

    if not all_input_clarifier_rows:
        print("No C1 data was processed. The 'all_input_clarifier' and 'all_output_clarifier' sheets will not be created.")
    else:
        df_all_input_c1 = pd.DataFrame(all_input_clarifier_rows)
        df_all_output_c1 = pd.DataFrame(all_output_clarifier_rows)
        new_sim_numbers_c1 = range(1, len(df_all_input_c1) + 1)
        df_all_input_c1['simulation_number'] = new_sim_numbers_c1
        df_all_output_c1['simulation_number'] = new_sim_numbers_c1
        
        # --- Reorder columns using the correct lists for the clarifier ---
        df_all_input_c1 = df_all_input_c1[all_input_clarifier_cols]
        output_clarifier_cols_with_sim = ['simulation_number'] + all_output_clarifier_cols
        df_all_output_c1 = df_all_output_c1.reindex(columns=output_clarifier_cols_with_sim)

    # --- 6. SAVE DATA: Write all four new DataFrames to the Excel file ---
    print(f"Saving consolidated sheets to '{filepath}'...")
    try:
        with pd.ExcelWriter(filepath, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            if 'df_all_input' in locals():
                df_all_input.to_excel(writer, sheet_name='all_input_cstr', index=False)
                df_all_output.to_excel(writer, sheet_name='all_output_cstr', index=False)
                print("Successfully created/updated 'all_input_cstr' and 'all_output_cstr' sheets.")
            
            if 'df_all_input_c1' in locals():
                df_all_input_c1.to_excel(writer, sheet_name='all_input_clarifier', index=False)
                df_all_output_c1.to_excel(writer, sheet_name='all_output_clarifier', index=False)
                print("Successfully created/updated 'all_input_clarifier' and 'all_output_clarifier' sheets.")
            
        print("File saving process complete.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(os.path.join('data', 'data.xlsx')):
        print("Creating a dummy 'data.xlsx' file for demonstration purposes.")
        sim_count = 5
        input_data = {
            'simulation_number': range(1, sim_count + 1), 'flow_rate': [18446] * sim_count,
            'V_A1': [1200] * sim_count, 'V_A2': [1500] * sim_count, 'V_O1': [1500] * sim_count,
            'V_O2': [1500] * sim_count, 'V_O3': [1500] * sim_count,
            'KLa_A1': [300] * sim_count, 'KLa_A2': [300] * sim_count, 'KLa_O1': [300] * sim_count,
            'KLa_O2': [300] * sim_count, 'KLa_O3': [300] * sim_count,
            'C1_surface_area': [1800] * sim_count, 'C1_height': [5] * sim_count,
            'Q_was': [450] * sim_count, 'Q_ext': [25000] * sim_count, 'O3_split_internal': [0.7] * sim_count
        }
        df_dummy_input = pd.DataFrame(input_data)
        
        output_data = {'simulation_number': range(1, sim_count + 1)}
        # Generate dummy columns for all process units, effluent, and wastage
        units = ['A1_eff', 'A2_eff', 'O1_eff', 'O2_eff', 'O3_to_C1', 'Effluent', 'Wastage']
        components = ['S_O2','S_N2','S_NH4','S_NO3','S_PO4','S_F','S_A','S_I','S_ALK','X_I','X_S','X_H','X_PAO','X_PP','X_PHA','X_AUT','X_MeOH','X_MeP','H2O','COD','BOD','TN','TKN','TP','TSS','VSS']
        for unit in units:
            for comp in components:
                col_name = f"Target_{unit}_{comp} (mg/L)".replace('_Effluent', '')
                output_data[col_name] = np.random.rand(sim_count) * 10
        df_dummy_output = pd.DataFrame(output_data)

        with pd.ExcelWriter(os.path.join('data', 'data.xlsx'), engine='openpyxl') as writer:
            df_dummy_input.to_excel(writer, sheet_name='results_input', index=False)
            df_dummy_output.to_excel(writer, sheet_name='results_output', index=False)
    
    create_all_sheets()