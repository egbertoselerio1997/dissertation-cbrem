import os
import qsdsan as qs
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
import gc

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_single_simulation(sim_id, input_df, vary_inputs, init_cond_df):
    """
    Runs a single simulation instance of the full A2/O activated sludge system.
    This function is designed to be parallelized.
    The 'sim_id' parameter is now used as the unique simulation number.
    """
    try:
        # --- 1. Get the inputs for the current simulation run ---
        current_inputs = {}
        for var in input_df.index:
            is_randomizable = input_df.loc[var, 'randomizable'] == 1
            if vary_inputs and is_randomizable:
                low = input_df.loc[var, 'min']
                high = input_df.loc[var, 'max']
                current_inputs[var] = np.random.uniform(low, high)
            else:
                current_inputs[var] = input_df.loc[var, 'min']

        # --- 2. Build the wastewater treatment system using the current inputs ---
        cmps = qs.processes.create_asm2d_cmps()
        qs.set_thermo(cmps)
        asm2d_model = qs.processes.ASM2d()
        Temp = 273.15 + 20

        Q_inf = current_inputs['flow_rate']
        Q_was = current_inputs['Q_was']
        Q_ext = current_inputs['Q_ext']
        V_an = current_inputs['V_an']
        V_ae = current_inputs['V_ae']
        KLa_aer1 = current_inputs['KLa_aer1']
        KLa_aer2 = current_inputs['KLa_aer2']
        O3_split_internal = current_inputs['O3_split_internal']
        C1_surface_area = current_inputs['C1_surface_area']
        C1_height = current_inputs['C1_height']

        # Use sim_id for unique object naming in parallel runs
        influent = qs.WasteStream(f'influent_{sim_id}', T=Temp)
        effluent = qs.WasteStream(f'effluent_{sim_id}', T=Temp)
        int_recycle = qs.WasteStream(f'internal_recycle_{sim_id}', T=Temp)
        ext_recycle = qs.WasteStream(f'external_recycle_{sim_id}', T=Temp)
        wastage = qs.WasteStream(f'wastage_{sim_id}', T=Temp)

        influent_concentrations = {
            key.replace('inf_', ''): value for key, value in current_inputs.items() if key.startswith('inf_')
        }
        influent.set_flow_by_concentration(Q_inf, concentrations=influent_concentrations, units=('m3/d', 'mg/L'))

        aer1 = qs.processes.DiffusedAeration(f'aer1_{sim_id}', DO_ID='S_O2', KLa=KLa_aer1, DOsat=8.0, V=V_ae)
        aer2 = qs.processes.DiffusedAeration(f'aer2_{sim_id}', DO_ID='S_O2', KLa=KLa_aer2, DOsat=8.0, V=V_ae)

        A1 = qs.sanunits.CSTR(f'A1_{sim_id}', ins=[influent, int_recycle, ext_recycle], V_max=V_an, aeration=None, suspended_growth_model=asm2d_model)
        A2 = qs.sanunits.CSTR(f'A2_{sim_id}', ins=A1-0, V_max=V_an, aeration=None, suspended_growth_model=asm2d_model)
        O1 = qs.sanunits.CSTR(f'O1_{sim_id}', ins=A2-0, V_max=V_ae, aeration=aer1, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O2 = qs.sanunits.CSTR(f'O2_{sim_id}', ins=O1-0, V_max=V_ae, aeration=aer1, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O3 = qs.sanunits.CSTR(f'O3_{sim_id}', ins=O2-0, outs=[int_recycle, 'treated'], split=[O3_split_internal, 1 - O3_split_internal], V_max=V_ae, aeration=aer2, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        C1 = qs.sanunits.FlatBottomCircularClarifier(f'C1_{sim_id}', ins=O3-1, outs=[effluent, ext_recycle, wastage], underflow=Q_ext, wastage=Q_was, surface_area=C1_surface_area, height=C1_height, N_layer=10, feed_layer=5, X_threshold=3000, v_max=474, v_max_practical=250, rh=5.76e-4, rp=2.86e-3, fns=2.28e-3)

        sys = qs.System(f'sys_{sim_id}', path=(A1, A2, O1, O2, O3, C1), recycle=(int_recycle, ext_recycle))

        # --- 3. Set initial conditions if provided ---
        if init_cond_df is not None:
            # Helper function to apply initial conditions, adapted for parallel runs
            def batch_init(system_obj, dataframe_with_init_cond, sim_index):
                dct = dataframe_with_init_cond.to_dict('index')
                u = system_obj.flowsheet.unit

                for unit_id_prefix in ['A1', 'A2', 'O1', 'O2', 'O3']:
                    unit_obj = getattr(u, f'{unit_id_prefix}_{sim_index}')
                    valid_concs = {comp: conc for comp, conc in dct[unit_id_prefix].items() if not pd.isna(conc)}
                    unit_obj.set_init_conc(**valid_concs)

                clarifier_obj = getattr(u, f'C1_{sim_index}')
                c1s = {k: v for k, v in dct['C1_s'].items() if not pd.isna(v) and v > 0}
                c1x = {k: v for k, v in dct['C1_x'].items() if not pd.isna(v) and v > 0}
                tss = [v for v in dct['C1_tss'].values() if not pd.isna(v) and v > 0]
                
                clarifier_obj.set_init_solubles(**c1s)
                clarifier_obj.set_init_sludge_solids(**c1x)
                clarifier_obj.set_init_TSS(tss)
            
            # Apply the initial conditions for the current simulation run
            batch_init(sys, init_cond_df, sim_id)

        # --- 4. Run the simulation ---
        sys.set_tolerance(rmol=1e-6)
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        
        # --- 5. Collect and return results ---
        all_results = {}

        streams_to_report = {
            'Target_A1_eff': A1.outs[0], 'Target_A2_eff': A2.outs[0], 'Target_O1_eff': O1.outs[0],
            'Target_O2_eff': O2.outs[0], 'Target_O3_to_C1': O3.outs[1], 'Target_Effluent': effluent,
            'Target_Wastage': wastage, 'Target_Int_Recycle': int_recycle, 'Target_Ext_Recycle': ext_recycle,
        }
        composite_properties = ['COD', 'BOD', 'TN', 'TKN', 'TP']
        composite_methods = ['TSS', 'VSS']
        
        for name, stream_obj in streams_to_report.items():
            if stream_obj.F_mass > 0:
                for comp_id, conc in zip(stream_obj.components.IDs, stream_obj.conc):
                    all_results[f'{name}_{comp_id} (mg/L)'] = conc
                for prop_name in composite_properties:
                    all_results[f'{name}_{prop_name} (mg/L)'] = getattr(stream_obj, prop_name)
                for method_name in composite_methods:
                    all_results[f'{name}_{method_name} (mg/L)'] = getattr(stream_obj, f'get_{method_name}')()
            else:
                for comp_id in effluent.components.IDs:
                    all_results[f'{name}_{comp_id} (mg/L)'] = np.nan
                for prop_name in composite_properties:
                    all_results[f'{name}_{prop_name} (mg/L)'] = np.nan
                for method_name in composite_methods:
                    all_results[f'{name}_{method_name} (mg/L)'] = np.nan

        srt_val = np.nan
        try:
            srt_val = qs.utils.get_SRT(sys, ('X_H', 'X_PAO', 'X_AUT'))
        except Exception:
            pass

        # Combine all data for this run, using sim_id as the simulation number
        run_data = {
            'simulation_number': sim_id,
            **current_inputs,
            **all_results,
            'SRT_days': srt_val
        }
        
        return run_data
    except Exception as e:
        print(f"Error in simulation {sim_id}: {e}")
        return None
    finally:
        qs.main_flowsheet.clear()

# Main simulation function
def run_simulation():
    """
    Main function to drive the simulation. It now appends results to existing
    files and continues simulation numbering from where it left off.
    """
    data_filepath = os.path.join('data', 'data.xlsx')

    print(f"Loading input variables from {data_filepath}...")
    try:
        input_df = pd.read_excel(data_filepath, sheet_name='input_config', index_col=0)
    except Exception as e:
        print(f"Error loading 'input_config' sheet: {e}. Please ensure it exists and is formatted correctly.")
        return

    print("Loading initial conditions from 'initial_conditions' sheet...")
    try:
        init_cond_df = pd.read_excel(data_filepath, sheet_name='initial_conditions', index_col=0)
        print("Successfully loaded initial conditions.")
    except Exception as e:
        print(f"Warning: Could not load 'initial_conditions' sheet: {e}. Simulations will start from default zero concentrations.")
        init_cond_df = None

    while True:
        try:
            num_simulations = int(input("Enter the number of simulations to run: "))
            if num_simulations > 0: break
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    vary_inputs = False
    if num_simulations > 1:
        while True:
            choice = input("Vary inputs randomly? (y/n): ").lower()
            if choice in ('y', 'n'):
                vary_inputs = (choice == 'y')
                break
            print("Invalid choice. Please enter 'y' or 'n'.")

    while True:
        try:
            batch_size = int(input("Enter the batch size for processing (e.g., 100): "))
            if batch_size > 0: break
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # REVISED: Determine the starting simulation number by reading the existing results file.
    start_sim_offset = 0
    try:
        with pd.ExcelFile(data_filepath) as xls:
            if 'results_input' in xls.sheet_names:
                existing_df = pd.read_excel(xls, sheet_name='results_input', usecols=['simulation_number'])
                if not existing_df.empty and existing_df['simulation_number'].notna().any():
                    start_sim_offset = int(existing_df['simulation_number'].max())
                    print(f"Found existing data. New simulations will continue from number {start_sim_offset + 1}.")
    except FileNotFoundError:
        print("Results file not found. Starting new file from simulation number 1.")
    except (ValueError, KeyError):
        print("Warning: Could not read previous simulation number from 'results_input'. Starting from 1.")
    except Exception as e:
        print(f"An unexpected error occurred while reading '{data_filepath}': {e}. Starting from 1.")

    total_successful = 0
    
    # REMOVED: The block that deleted previous results sheets is gone.
    # Data will now be appended by the pd.ExcelWriter logic below.

    print(f"\nStarting {num_simulations} simulations in batches of {batch_size}...")
    for i in tqdm(range(0, num_simulations, batch_size), desc="Processing Batches"):
        batch_end = min(i + batch_size, num_simulations)
        
        with Parallel(n_jobs=-1) as parallel:
            # REVISED: Pass a unique, continuous simulation ID to each run.
            # The ID is based on the last simulation number found (start_sim_offset)
            # plus the index of the simulation in the current run (run_idx).
            batch_results = parallel(
                delayed(run_single_simulation)(
                    sim_id=start_sim_offset + run_idx + 1,
                    input_df=input_df,
                    vary_inputs=vary_inputs,
                    init_cond_df=init_cond_df
                ) for run_idx in range(i, batch_end)
            )

        successful_batch_results = [r for r in batch_results if r is not None]
        total_successful += len(successful_batch_results)

        if not successful_batch_results:
            continue

        results_df = pd.DataFrame(successful_batch_results)
        
        input_cols = ['simulation_number'] + input_df.index.to_list()
        input_results_df = results_df.reindex(columns=input_cols)
        
        output_cols = ['simulation_number'] + [col for col in results_df.columns if col not in input_df.index and col != 'simulation_number']
        output_results_df = results_df.reindex(columns=output_cols)

        # This block correctly appends new rows to existing sheets or creates them if they don't exist.
        try:
            with pd.ExcelWriter(data_filepath, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                # Check if sheet exists to decide on writing headers
                header_input = 'results_input' not in writer.book.sheetnames
                startrow_input = 0 if header_input else writer.book['results_input'].max_row
                input_results_df.to_excel(writer, sheet_name='results_input', index=False, header=header_input, startrow=startrow_input)

                header_output = 'results_output' not in writer.book.sheetnames
                startrow_output = 0 if header_output else writer.book['results_output'].max_row
                output_results_df.to_excel(writer, sheet_name='results_output', index=False, header=header_output, startrow=startrow_output)
        except Exception as e:
            print(f"\nAn error occurred while saving batch results: {e}")
            break
        
        del batch_results, successful_batch_results, results_df, input_results_df, output_results_df
        gc.collect()

    print(f"\n{total_successful}/{num_simulations} simulations completed successfully.")

    if total_successful == 0:
        print("No results to save or analyze.")
        return

    print(f"All results have been incrementally saved to '{data_filepath}'.")
    
    if total_successful > 1 or start_sim_offset > 0:
        print("Calculating final statistics from the complete saved results...")
        try:
            # Read all results back to generate comprehensive statistics
            final_output_df = pd.read_excel(data_filepath, sheet_name='results_output')
            stats_df = final_output_df.drop(columns=['simulation_number']).describe().T
            
            # Replace the old statistics sheet with the new, updated one
            with pd.ExcelWriter(data_filepath, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                stats_df.to_excel(writer, sheet_name='results_statistics')
            
            print(f"Final statistics saved to sheet 'results_statistics' in '{data_filepath}'.")
        except Exception as e:
            print(f"An error occurred during final statistics calculation: {e}")

if __name__ == '__main__':
    run_simulation()