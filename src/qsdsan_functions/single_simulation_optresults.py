# File: run_validation_simulation.py
# Description: This script runs a single QSDsan simulation using the optimal
#              parameters found by the optimization script. It serves to validate
#              the surrogate model's predictions against the detailed simulator.

import os
import qsdsan as qs
import pandas as pd
import numpy as np
import warnings

# Suppress the specific pkg_resources UserWarning from QSDsan dependencies
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

def load_optimal_inputs(filepath: str):
    """
    Loads optimal decision variables and default influent quality from the
    specified Excel file.

    Args:
        filepath (str): The path to the data.xlsx file.

    Returns:
        A tuple containing:
        - dict: A dictionary of all inputs required for the simulation.
        - pd.DataFrame: A DataFrame of the surrogate model's predicted effluent quality.
    """
    print(f"1. Loading optimal inputs from '{filepath}'...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"The results file was not found at '{filepath}'. "
            "Please run the optimization script first to generate it."
        )

    try:
        xls = pd.read_excel(filepath, sheet_name=None)
        
        # Load decision variables
        df_dec_vars = xls['optimal_decision_variables']
        decision_inputs = df_dec_vars.set_index('Variable')['Optimal Value'].to_dict()
        print(f"   - Loaded optimal decision variables: {decision_inputs}")
        
        # Load influent quality
        df_influent = xls['default_influent_quality']
        influent_inputs = df_influent.set_index('Variable')['Value (mg/L)'].to_dict()
        print(f"   - Loaded default influent concentrations.")
        
        # Load predicted effluent for comparison
        df_predicted_effluent = xls['optimal_predicted_effluent']
        print(f"   - Loaded predicted effluent quality for validation.")

        # Combine all inputs into a single dictionary
        all_inputs = {**decision_inputs, **influent_inputs}
        
        return all_inputs, df_predicted_effluent

    except KeyError as e:
        raise KeyError(
            f"A required sheet {e} was not found in '{filepath}'. "
            "Ensure the optimization script ran successfully and created all necessary sheets."
        )

def run_validation_simulation(inputs: dict):
    """
    Runs a single, detailed QSDsan simulation using the provided input parameters.

    Args:
        inputs (dict): A dictionary containing all necessary inputs like
                       flow_rate, HRT, DO_setpoint, and influent concentrations.

    Returns:
        dict: A dictionary of the simulated effluent concentrations.
    """
    print("\n2. Setting up and running the QSDsan validation simulation...")
    
    simulated_effluent = {}
    try:
        # 1. Setup QSDsan environment
        cmps = qs.processes.create_asm2d_cmps()
        qs.set_thermo(cmps)
        
        # 2. Prepare influent waste stream
        influent_concentrations = {
            key.replace('inf_', ''): value
            for key, value in inputs.items() if key.startswith('inf_')
        }
        
        ws = qs.WasteStream('influent_ws_validation')
        ws.set_flow_by_concentration(
            flow_tot=inputs['flow_rate'],
            concentrations=influent_concentrations,
            units=('m3/d', 'mg/L')
        )
        
        # 3. Setup the reactor (CSTR) with ASM2d model
        asm2d_model = qs.processes.ASM2d()
        V_max = ws.F_vol * (inputs['HRT'] / 24) # HRT is in hours

        reactor = qs.sanunits.CSTR(
            ID='R_validation',
            ins=ws.copy(),
            V_max=V_max,
            aeration=inputs['DO_setpoint'],
            suspended_growth_model=asm2d_model,
            DO_ID='S_O2'
        )
        
        # 4. Create and run the system
        sys = qs.System('sys_validation', path=(reactor,))
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")

        # 5. Extract results
        effluent_stream = reactor.outs[0]
        component_ids = reactor.components.IDs
        effluent_concs_array = effluent_stream.conc
        
        for idx, comp_id in enumerate(component_ids):
            simulated_effluent[f'Effluent_{comp_id}'] = effluent_concs_array[idx]
        
    except Exception as e:
        print(f"\nAN ERROR OCCURRED during the QSDsan simulation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up QSDsan objects to prevent conflicts if run in a loop
        qs.main_flowsheet.clear()
        
    return simulated_effluent

def display_comparison_report(predicted_df: pd.DataFrame, simulated_dict: dict):
    """
    Compares the surrogate model's predictions with the detailed simulation
    results and prints a report.
    """
    print("\n3. Comparing Surrogate Model Predictions vs. Detailed QSDsan Simulation:")
    print("=" * 80)
    
    comparison_data = []
    
    for _, row in predicted_df.iterrows():
        comp_name_full = row['Component']
        # The component name from the prediction might be like "Effluent_S_I" or just "S_I"
        # We need to handle both cases to match the simulated dict keys.
        sim_key = f"Effluent_{comp_name_full}" if not comp_name_full.startswith('Effluent_') else comp_name_full
        
        if sim_key in simulated_dict:
            predicted_val = row['Predicted Value (mg/L)']
            simulated_val = simulated_dict[sim_key]
            difference = simulated_val - predicted_val
            # Avoid division by zero for relative difference
            relative_diff = (difference / predicted_val * 100) if abs(predicted_val) > 1e-6 else np.nan
            
            comparison_data.append({
                'Component': comp_name_full.replace('Effluent_', ''),
                'Predicted (Surrogate) mg/L': predicted_val,
                'Simulated (QSDsan) mg/L': simulated_val,
                'Absolute Difference': difference,
                'Relative Difference (%)': relative_diff
            })

    if not comparison_data:
        print("Could not generate comparison report. No matching components found.")
        return

    df_report = pd.DataFrame(comparison_data)
    
    # Format the report for better readability
    pd.options.display.float_format = '{:,.4f}'.format
    print(df_report.to_string(index=False))
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = df_report['Relative Difference (%)'].abs().mean()
    print("-" * 80)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("=" * 80)
    
    if mape < 5:
        print("\nConclusion: Excellent agreement. The surrogate model is a highly accurate representation.")
    elif mape < 15:
        print("\nConclusion: Good agreement. The surrogate model is a reliable representation.")
    else:
        print("\nConclusion: Moderate to poor agreement. The surrogate model shows some deviation.")


def main():
    """Main function to orchestrate the validation workflow."""
    data_filepath = os.path.join('data', 'data.xlsx')
    
    try:
        # Load the optimal parameters from the file generated by the optimization script
        optimal_inputs, predicted_effluent_df = load_optimal_inputs(data_filepath)
        
        # Run the detailed QSDsan simulation with these parameters
        simulated_effluent_concs = run_validation_simulation(optimal_inputs)
        
        # If the simulation was successful, compare results
        if simulated_effluent_concs:
            display_comparison_report(predicted_effluent_df, simulated_effluent_concs)
            
    except (FileNotFoundError, KeyError) as e:
        print(f"\nVALIDATION FAILED: {e}")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()