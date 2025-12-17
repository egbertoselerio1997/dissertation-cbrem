import os
import sys
from pathlib import Path
import qsdsan as qs
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
import gc
import uuid
from scipy.stats import qmc

# Allow importing shared naming utilities
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
from naming import CONCENTRATION_SUFFIX, canonical_base_name, legacy_identifier  # noqa: E402

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_single_simulation(sim_id, inputs, init_cond_df):
    """
    Runs a single simulation and directly processes the results into the final
    data structures for CSTR, Clarifier, and flows.
    This function is designed to be parallelized.
    """
    try:
        def formatted_name(prefix: str, compound_id: str) -> str:
            base = canonical_base_name(compound_id) or compound_id
            return f"{prefix}_{base}{CONCENTRATION_SUFFIX}"

        simulation_uid = f"sim-{uuid.uuid4().hex}"

        # --- 1. Get the inputs for the current simulation run ---
        current_inputs = inputs

        # --- 2. Build the wastewater treatment system using the current inputs ---
        cmps = qs.processes.create_asm2d_cmps()
        qs.set_thermo(cmps)
        asm2d_model = qs.processes.ASM2d()
        Temp = 273.15 + 20

        # Extract operational parameters from inputs
        Q_inf = current_inputs['flow_rate']
        Q_was = current_inputs['Q_was']
        Q_ext = current_inputs['Q_ext']
        V_A1, V_A2 = current_inputs['V_A1'], current_inputs['V_A2']
        V_O1, V_O2, V_O3 = current_inputs['V_O1'], current_inputs['V_O2'], current_inputs['V_O3']
        KLa_A1, KLa_A2 = current_inputs['KLa_A1'], current_inputs['KLa_A2']
        KLa_O1, KLa_O2, KLa_O3 = current_inputs['KLa_O1'], current_inputs['KLa_O2'], current_inputs['KLa_O3']
        O3_split_internal = current_inputs['O3_split_internal']
        C1_surface_area = current_inputs['C1_surface_area']
        C1_height = current_inputs['C1_height']

        # Create streams with unique names for parallel runs
        influent = qs.WasteStream(f'influent_{sim_id}', T=Temp)
        effluent = qs.WasteStream(f'effluent_{sim_id}', T=Temp)
        int_recycle = qs.WasteStream(f'internal_recycle_{sim_id}', T=Temp)
        ext_recycle = qs.WasteStream(f'external_recycle_{sim_id}', T=Temp)
        wastage = qs.WasteStream(f'wastage_{sim_id}', T=Temp)

        influent_concentrations = {}
        for key, value in current_inputs.items():
            if key.startswith(('inf_', 'influent_')):
                raw_token = key.split('_', 1)[1]
                descriptive_base = canonical_base_name(raw_token) or raw_token
                legacy_id = legacy_identifier(descriptive_base)
                influent_concentrations[legacy_id] = value
        influent.set_flow_by_concentration(Q_inf, concentrations=influent_concentrations, units=('m3/d', 'mg/L'))

        # Add aeration for all CSTRs
        aer_A1 = qs.processes.DiffusedAeration(f'aer_A1_{sim_id}', DO_ID='S_O2', KLa=KLa_A1, DOsat=8.0, V=V_A1)
        aer_A2 = qs.processes.DiffusedAeration(f'aer_A2_{sim_id}', DO_ID='S_O2', KLa=KLa_A2, DOsat=8.0, V=V_A2)
        aer_O1 = qs.processes.DiffusedAeration(f'aer_O1_{sim_id}', DO_ID='S_O2', KLa=KLa_O1, DOsat=8.0, V=V_O1)
        aer_O2 = qs.processes.DiffusedAeration(f'aer_O2_{sim_id}', DO_ID='S_O2', KLa=KLa_O2, DOsat=8.0, V=V_O2)
        aer_O3 = qs.processes.DiffusedAeration(f'aer_O3_{sim_id}', DO_ID='S_O2', KLa=KLa_O3, DOsat=8.0, V=V_O3)

        # Add aeration argument to A1 and A2 CSTRs
        A1 = qs.sanunits.CSTR(f'A1_{sim_id}', ins=[influent, int_recycle, ext_recycle], V_max=V_A1, aeration=aer_A1, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        A2 = qs.sanunits.CSTR(f'A2_{sim_id}', ins=A1-0, V_max=V_A2, aeration=aer_A2, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O1 = qs.sanunits.CSTR(f'O1_{sim_id}', ins=A2-0, V_max=V_O1, aeration=aer_O1, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O2 = qs.sanunits.CSTR(f'O2_{sim_id}', ins=O1-0, V_max=V_O2, aeration=aer_O2, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O3 = qs.sanunits.CSTR(f'O3_{sim_id}', ins=O2-0, outs=[int_recycle, 'treated'], split=[O3_split_internal, 1 - O3_split_internal], V_max=V_O3, aeration=aer_O3, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        C1 = qs.sanunits.FlatBottomCircularClarifier(f'C1_{sim_id}', ins=O3-1, outs=[effluent, ext_recycle, wastage], underflow=Q_ext, wastage=Q_was, surface_area=C1_surface_area, height=C1_height)

        sys = qs.System(f'sys_{sim_id}', path=(A1, A2, O1, O2, O3, C1), recycle=(int_recycle, ext_recycle))

        # --- 3. Set initial conditions (if provided) ---
        if init_cond_df is not None:
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
                clarifier_obj.set_init_solubles(**c1s); clarifier_obj.set_init_sludge_solids(**c1x); clarifier_obj.set_init_TSS(tss)
            batch_init(sys, init_cond_df, sim_id)

        # --- 4. Run the simulation ---
        sys.set_tolerance(rmol=1e-6)
        sys.simulate(t_span=(0, 180), method='BDF', state_reset_hook='reset_cache')

        # --- 5. Collect and process results directly ---
        
        def get_stream_data(stream_obj):
            if stream_obj.F_mass <= 0: return None
            data = {f'{comp_id}': conc for comp_id, conc in zip(stream_obj.components.IDs, stream_obj.conc)}
            composite_properties = ['COD', 'BOD', 'TN', 'TKN', 'TP', 'TSS', 'VSS', 'TOC', 'TC']
            for prop in composite_properties:
                getter = getattr(stream_obj, f'get_{prop}', getattr(stream_obj, prop, None))
                data[prop] = getter() if callable(getter) else getter
            return data

        stream_results = {
            'A1_in': get_stream_data(A1.ins[0]), 'A1_eff': get_stream_data(A1.outs[0]),
            'A2_eff': get_stream_data(A2.outs[0]), 'O1_eff': get_stream_data(O1.outs[0]),
            'O2_eff': get_stream_data(O2.outs[0]), 'O3_to_C1': get_stream_data(O3.outs[1]),
            'Effluent': get_stream_data(effluent), 'Wastage': get_stream_data(wastage),
        }
        
        # Calculate key flow rates in m3/d
        Q_int_daily = int_recycle.F_vol * 24
        wastage_daily = wastage.F_vol * 24
        effluent_daily = effluent.F_vol * 24
        
        epsilon = 1e-9
        total_cycle_flow = (Q_inf + Q_ext) / (1 - O3_split_internal + epsilon)

        # --- Prepare CSTR datasets ---
        cstr_input_rows, cstr_output_rows = [], []
        units_to_process = {
            'A1': {'in_stream_key': 'A1_in', 'out_stream_key': 'A1_eff', 'V': V_A1, 'KLa': KLa_A1},
            'A2': {'in_stream_key': 'A1_eff', 'out_stream_key': 'A2_eff', 'V': V_A2, 'KLa': KLa_A2},
            'O1': {'in_stream_key': 'A2_eff', 'out_stream_key': 'O1_eff', 'V': V_O1, 'KLa': KLa_O1},
            'O2': {'in_stream_key': 'O1_eff', 'out_stream_key': 'O2_eff', 'V': V_O2, 'KLa': KLa_O2},
            'O3': {'in_stream_key': 'O2_eff', 'out_stream_key': 'O3_to_C1', 'V': V_O3, 'KLa': KLa_O3}
        }
        
        all_inf_components = [c for c in cmps.IDs if c not in ['H2O']]
        composite_properties = ['COD', 'BOD', 'TN', 'TKN', 'TP', 'TSS', 'VSS', 'TOC', 'TC']
        all_out_components = list(all_inf_components) + composite_properties + ['X_MeOH','X_MeP','H2O']

        for _, details in units_to_process.items():
            in_data = stream_results.get(details['in_stream_key'])
            out_data = stream_results.get(details['out_stream_key'])
            if not in_data or not out_data: continue

            input_row = {
                'simulation_id': simulation_uid, 'simulation_number': sim_id,
                'Q_raw_inf': Q_inf, 'Q_int': Q_int_daily,
                'Q_was': Q_was, 'Q_ext': Q_ext, 'V': details['V'], 'KLa': details['KLa']
            }
            for comp_id in all_inf_components:
                input_row[formatted_name('influent', comp_id)] = in_data.get(comp_id, 0)
            for prop in composite_properties:
                input_row[formatted_name('influent', prop)] = in_data.get(prop, 0)
            cstr_input_rows.append(input_row)
            
            output_row = {'simulation_id': simulation_uid, 'simulation_number': sim_id}
            for comp_id in all_out_components:
                output_row[formatted_name('effluent', comp_id)] = out_data.get(comp_id, 0)
            cstr_output_rows.append(output_row)
            
        # --- Prepare Clarifier dataset ---
        clarifier_input_row, clarifier_output_row = None, None
        c1_in_data, c1_eff_data, c1_was_data = stream_results.get('O3_to_C1'), stream_results.get('Effluent'), stream_results.get('Wastage')

        if all([c1_in_data, c1_eff_data, c1_was_data]):
            clarifier_input_row = {
                'simulation_id': simulation_uid, 'simulation_number': sim_id, 'Q_raw_inf': Q_inf,
                'C1_surface_area': C1_surface_area, 'C1_height': C1_height,
                'Q_int': Q_int_daily, 'Q_was': Q_was, 'Q_ext': Q_ext
            }
            for comp_id in all_inf_components:
                clarifier_input_row[formatted_name('influent', comp_id)] = c1_in_data.get(comp_id, 0)
            for prop in composite_properties:
                clarifier_input_row[formatted_name('influent', prop)] = c1_in_data.get(prop, 0)
            
            clarifier_output_row = {'simulation_id': simulation_uid, 'simulation_number': sim_id}
            for comp in all_out_components:
                clarifier_output_row[formatted_name('effluent', comp)] = c1_eff_data.get(comp, 0)
                clarifier_output_row[formatted_name('wastage', comp)] = c1_was_data.get(comp, 0)
        
        # --- Prepare Flows dataset ---
        flows_row = {
            'simulation_number': sim_id, 'Raw influent flow': Q_inf,
            'Final effluent flow': effluent_daily, 'Wastage flow': wastage_daily,
            'Total cycle flow': total_cycle_flow, 'External recycle flow': Q_ext,
            'Split internal ratio': O3_split_internal, 'Internal recycle flow': Q_int_daily
        }
        
        return {
            "cstr_inputs": cstr_input_rows, "cstr_outputs": cstr_output_rows,
            "clarifier_input": [clarifier_input_row] if clarifier_input_row else [],
            "clarifier_output": [clarifier_output_row] if clarifier_output_row else [],
            "flows": [flows_row]
        }
    except Exception as e:
        print(f"Error in simulation {sim_id}: {e}")
        return None
    finally:
        qs.main_flowsheet.clear()

def run_simulation():
    config_dir = os.path.join('data', 'config')
    data_filepath = os.path.join(config_dir, 'simulation_training_config.xlsx')
    os.makedirs(config_dir, exist_ok=True)

    print(f"Loading input variables from {data_filepath}...")
    try:
        input_df = pd.read_excel(data_filepath, sheet_name='input_config', index_col=0)
    except Exception as e:
        print(f"Error loading 'input_config' sheet: {e}. Please ensure it exists and is formatted correctly.")
        return

    print("Loading initial conditions...")
    try:
        init_cond_df = pd.read_excel(data_filepath, sheet_name='initial_conditions', index_col=0)
    except Exception:
        print("Warning: Could not load 'initial_conditions'. Starting from default concentrations.")
        init_cond_df = None

    start_sim_offset, cstr_start_offset = 0, 0
    try:
        with pd.ExcelFile(data_filepath) as xls:
            if 'all_input_flows' in xls.sheet_names:
                df_flows = pd.read_excel(xls, sheet_name='all_input_flows', usecols=['simulation_number'])
                if not df_flows.empty and df_flows['simulation_number'].notna().any():
                    start_sim_offset = int(df_flows['simulation_number'].max())
                    print(f"Found existing data. New plant simulations will continue from number {start_sim_offset + 1}.")
            
            if 'all_input_cstr' in xls.sheet_names:
                df_cstr = pd.read_excel(xls, sheet_name='all_input_cstr', usecols=['simulation_number'])
                if not df_cstr.empty and df_cstr['simulation_number'].notna().any():
                    cstr_start_offset = int(df_cstr['simulation_number'].max())
                    print(f"Found existing CSTR data. New CSTR rows will be numbered starting from {cstr_start_offset + 1}.")
    except FileNotFoundError:
        print("Results file not found. Starting new file from simulation number 1.")
    except Exception as e:
        print(f"Warning reading '{data_filepath}': {e}. Starting from 1.")

    num_simulations = int(input("Enter the number of simulations to run: "))
    vary_inputs = False
    if num_simulations > 1:
        vary_inputs = input("Vary inputs using Latin Hypercube Sampling? (y/n): ").lower() == 'y'
    batch_size = int(input("Enter batch size for processing (e.g., 100): "))

    all_input_cstr_rows, all_output_cstr_rows = [], []
    all_input_clarifier_rows, all_output_clarifier_rows = [], []
    all_flows_rows = []
    
    total_successful = 0

    # Prepare samples for all simulations
    if vary_inputs:
        randomizable_vars = input_df[input_df['randomizable'] == 1]
        l_bounds = randomizable_vars['min'].values
        u_bounds = randomizable_vars['max'].values
        
        # Correctly initialize the sampler with a Generator instance
        sampler = qmc.LatinHypercube(d=len(randomizable_vars), seed=np.random.default_rng())
        
        sample = sampler.random(n=num_simulations)
        scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
    
    all_inputs = []
    for i in range(num_simulations):
        sim_inputs = {}
        if vary_inputs:
            random_values = {var: scaled_sample[i, j] for j, var in enumerate(randomizable_vars.index)}
            for var in input_df.index:
                if var in random_values:
                    sim_inputs[var] = random_values[var]
                else:
                    sim_inputs[var] = input_df.loc[var, 'baseline']
        else:
            for var in input_df.index:
                sim_inputs[var] = input_df.loc[var, 'baseline']
        all_inputs.append(sim_inputs)


    print(f"\nStarting {num_simulations} simulations in batches of {batch_size}...")
    for i in tqdm(range(0, num_simulations, batch_size), desc="Processing Batches"):
        batch_end = min(i + batch_size, num_simulations)
        
        with Parallel(n_jobs=-1) as parallel:
            batch_results = parallel(
                delayed(run_single_simulation)(
                    sim_id=start_sim_offset + run_idx + 1,
                    inputs=all_inputs[run_idx],
                    init_cond_df=init_cond_df
                ) for run_idx in range(i, batch_end)
            )

        successful_results = [r for r in batch_results if r is not None]
        total_successful += len(successful_results)
        if not successful_results: continue

        batch_input_cstr = [row for res in successful_results for row in res['cstr_inputs']]
        batch_output_cstr = [row for res in successful_results for row in res['cstr_outputs']]
        batch_input_c1 = [row for res in successful_results for row in res['clarifier_input']]
        batch_output_c1 = [row for res in successful_results for row in res['clarifier_output']]
        batch_flows = [row for res in successful_results for row in res['flows']]
        
        all_input_cstr_rows.extend(batch_input_cstr)
        all_output_cstr_rows.extend(batch_output_cstr)
        all_input_clarifier_rows.extend(batch_input_c1)
        all_output_clarifier_rows.extend(batch_output_c1)
        all_flows_rows.extend(batch_flows)
        
        del batch_results, successful_results
        gc.collect()

    print(f"\n{total_successful}/{num_simulations} simulations completed successfully.")
    if not all_input_cstr_rows:
        print("No new results to save.")
        return

    print("Simulations complete. Merging new data with existing results and saving...")
    
    # --- Define final column order based on generated data ---
    inflow_keys = set()
    for row in all_input_cstr_rows + all_input_clarifier_rows:
        inflow_keys.update(k for k in row.keys() if k.startswith('influent_'))
    inf_cols = sorted(inflow_keys)

    final_cstr_cols = ['simulation_id', 'simulation_number', 'Q_raw_inf', 'Q_int', 'Q_was', 'Q_ext', 'V', 'KLa'] + inf_cols
    final_clarifier_cols = ['simulation_id', 'simulation_number', 'Q_raw_inf', 'C1_surface_area', 'C1_height', 'Q_int', 'Q_was', 'Q_ext'] + inf_cols
    
    # Create DataFrames from the newly generated data
    df_new_input_cstr = pd.DataFrame(all_input_cstr_rows).reindex(columns=final_cstr_cols)
    df_new_output_cstr = pd.DataFrame(all_output_cstr_rows)
    df_new_input_clarifier = pd.DataFrame(all_input_clarifier_rows).reindex(columns=final_clarifier_cols)
    df_new_output_clarifier = pd.DataFrame(all_output_clarifier_rows)
    df_new_flows = pd.DataFrame(all_flows_rows)

    def reorder_with_ids(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        front = [c for c in ['simulation_id', 'simulation_number'] if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        return df[front + rest]

    df_new_output_cstr = reorder_with_ids(df_new_output_cstr)
    df_new_output_clarifier = reorder_with_ids(df_new_output_clarifier)

    if not df_new_input_cstr.empty:
        df_new_input_cstr['simulation_number'] = range(cstr_start_offset + 1, cstr_start_offset + len(df_new_input_cstr) + 1)
        df_new_output_cstr['simulation_number'] = range(cstr_start_offset + 1, cstr_start_offset + len(df_new_output_cstr) + 1)

    dfs_new = {
        'all_input_cstr': df_new_input_cstr, 'all_output_cstr': df_new_output_cstr,
        'all_input_clarifier': df_new_input_clarifier, 'all_output_clarifier': df_new_output_clarifier,
        'all_input_flows': df_new_flows,
    }
    
    dfs_to_write = {}
    for sheet_name, new_df in dfs_new.items():
        if new_df.empty: continue
        try:
            existing_df = pd.read_excel(data_filepath, sheet_name=sheet_name)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except (FileNotFoundError, ValueError): 
            combined_df = new_df
        dfs_to_write[sheet_name] = combined_df

    try:
        mode = 'a' if os.path.exists(data_filepath) else 'w'
        with pd.ExcelWriter(data_filepath, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
            for sheet_name, df_to_write in dfs_to_write.items():
                df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                
        print(f"All data successfully merged and saved to '{data_filepath}'.")
    except Exception as e:
        print(f"An error occurred while saving the final Excel file: {e}")

if __name__ == '__main__':
    run_simulation()
