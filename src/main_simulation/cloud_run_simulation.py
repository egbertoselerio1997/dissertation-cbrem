import os
import argparse  # ## REVISION: Import argparse for command-line arguments
import warnings
import pandas as pd
import numpy as np
import qsdsan as qs
from joblib import Parallel, delayed
from scipy.stats import norm
from google.cloud import storage # ## REVISION: Import Google Cloud Storage library

# Suppress the specific pkg_resources UserWarning - This is fine to keep.
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

# ## REVISION: This function is perfect and needs no changes. It's the core worker.
def run_single_simulation(i, input_df, vary_inputs, probability_val=None):
    """
    Runs a single simulation instance. This function is designed to be parallelized.
    """
    try:
        # Each parallel worker needs its own QSDsan environment setup
        cmps = qs.processes.create_asm2d_cmps()
        qs.set_thermo(cmps)
        
        current_inputs = {}
        for var in input_df.index:
            is_randomizable = input_df.loc[var, 'randomizable'] == 1
            if vary_inputs and is_randomizable:
                if probability_val is None:
                    raise ValueError("Probability value must be provided for random variation.")
                
                mean_val = input_df.loc[var, 'default']
                std_dev = input_df.loc[var, 'std. dev.']
                z = norm.ppf(1 - (1 - probability_val) / 2)
                low = max(0, mean_val - z * std_dev)
                high = mean_val + z * std_dev
                current_inputs[var] = np.random.uniform(low, high)
            else:
                current_inputs[var] = input_df.loc[var, 'default']

        influent_concentrations = {
            key.replace('inf_', ''): value
            for key, value in current_inputs.items() if key.startswith('inf_')
        }
        ws = qs.WasteStream(f'influent_ws_{i}')
        ws.set_flow_by_concentration(flow_tot=current_inputs['flow_rate'], concentrations=influent_concentrations, units=('m3/d', 'mg/L'))

        asm2d_model = qs.processes.ASM2d()
        V_max = ws.F_vol * (current_inputs['HRT'] / 24)

        reactor = qs.sanunits.CSTR(
            ID=f'R_{i}', ins=ws.copy(), V_max=V_max,
            aeration=current_inputs['DO_setpoint'],
            suspended_growth_model=asm2d_model, DO_ID='S_O2'
        )
        
        sys = qs.System(f'sys_{i}', path=(reactor,))
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        
        outputs = reactor.results(with_units=False)
        effluent_stream = reactor.outs[0]
        component_ids = reactor.components.IDs
        effluent_concs = effluent_stream.conc
        effluent_data = {f'Effluent_{comp_id} (mg/L)': conc for idx, (comp_id, conc) in enumerate(zip(component_ids, effluent_concs))}
        
        run_data = {
            'simulation_number': i + 1,
            **current_inputs,
            **outputs,
            **effluent_data
        }
        
        return run_data
    except Exception as e:
        # Log the error for this specific run, but don't stop the whole process
        print(f"Warning: Simulation {i+1} failed with error: {e}")
        return None

# ## REVISION: Renamed `run_simulation` to `orchestrate_simulations` and made it return the results DataFrame.
# It no longer handles file loading or saving.
def orchestrate_simulations(input_df):
    """
    Orchestrates the running of all simulation instances and returns the results.
    """
    num_simulations = 10  # You can still configure this here
    vary_inputs = True
    probability_val = 0.99

    print("\nStarting simulations...")
    # The parallel execution logic is unchanged and correct.
    all_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_simulation)(i, input_df, vary_inputs, probability_val)
        for i in range(num_simulations)
    )

    final_successful_results = [r for r in all_results if r is not None]
    if not final_successful_results:
        print("\nAll simulations failed. No results to process.")
        return None

    print(f"\n{len(final_successful_results)}/{num_simulations} simulations completed successfully.")
    
    # Return the final DataFrame
    return pd.DataFrame(final_successful_results)

# ## REVISION: New helper function to handle uploading files to Google Cloud Storage.
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the specified GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"Successfully uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise

# ## REVISION: Replaced the old `run_simulation()` with a `main()` function.
# This is the new entry point that controls the entire workflow.
def main():
    """
    Main controller for the script. Handles I/O and calls the simulation logic.
    """
    # 1. Set up argument parsing to receive bucket/output names from the startup script
    parser = argparse.ArgumentParser(description="Run bioreactor simulation and upload results to GCS.")
    parser.add_argument('--bucket', required=True, help='GCS bucket name for results.')
    parser.add_argument('--output', required=True, help='Output filename for results in GCS.')
    args = parser.parse_args()

    bucket_name = args.bucket.replace("gs://", "")
    
    # 2. Define file paths on the VM. The startup script places data.xlsx in the root.
    local_data_path = 'data.xlsx'
    # We will save results to a temporary file before uploading.
    local_results_path = 'temp_results.xlsx'

    # 3. Load the input data
    print(f"Loading input variables from {local_data_path}...")
    try:
        input_df = pd.read_excel(local_data_path, sheet_name='input_config', index_col=0)
    except Exception as e:
        print(f"Fatal Error: Could not load or parse '{local_data_path}'. Error: {e}")
        return # Exit if we can't even load the data

    # 4. Run the simulation logic
    results_df = orchestrate_simulations(input_df)

    # 5. Process and save the results if any were successful
    if results_df is None:
        print("No successful results to save. Exiting.")
        return

    print(f"Saving results to temporary file: {local_results_path}...")
    try:
        # Replicate the multi-sheet saving logic from your original script
        input_cols = ['simulation_number'] + input_df.index.to_list()
        input_results_df = results_df[[col for col in input_cols if col in results_df.columns]]
        
        output_cols = ['simulation_number'] + [col for col in results_df.columns if col not in input_df.index]
        output_results_df = results_df[output_cols]

        with pd.ExcelWriter(local_results_path, engine='openpyxl') as writer:
            input_results_df.to_excel(writer, sheet_name='results_input', index=False)
            output_results_df.to_excel(writer, sheet_name='results_output', index=False)
            
            if len(results_df) > 1:
                print("Calculating statistics...")
                stats_df = output_results_df.drop(columns=['simulation_number']).describe().T
                stats_df.to_excel(writer, sheet_name='results_statistics')
        
        # 6. Upload the final, multi-sheet Excel file to the bucket
        upload_to_gcs(
            bucket_name=bucket_name,
            source_file_name=local_results_path,
            destination_blob_name=args.output
        )

    except Exception as e:
        print(f"An error occurred while saving or uploading the results: {e}")

# ## REVISION: The standard entry point now calls the new main() function.
if __name__ == '__main__':
    main()