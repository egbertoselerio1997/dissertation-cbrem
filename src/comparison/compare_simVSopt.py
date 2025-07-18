# File: run_validation_simulation.py
# Description: This script runs a single QSDsan simulation using optimal
#              parameters to validate a surrogate model's predictions.

import os
import qsdsan as qs
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_simple_inputs(filepath: str):
    """
    Loads inputs for simple models (CSTR, Clarifier) from the specified optimization results file.
    """
    print(f"1. Loading inputs for simple model from '{filepath}'...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file was not found at '{filepath}'.")

    try:
        xls = pd.read_excel(filepath, sheet_name=None)
        
        df_dec_vars = xls['optimal_decision_variables']
        decision_inputs = {}
        # Handle unit-specific variables to create unique keys (e.g., 'KLa_O1')
        for _, row in df_dec_vars.iterrows():
            variable = row['Variable']
            unit = row['Process Unit']
            value = row['Optimal Value']
            if variable in ['KLa', 'V']:
                key = f"{variable}_{unit}"
            else:
                key = variable
            decision_inputs[key] = value

        # Rename Q_raw_inf to flow_rate for compatibility with simulation functions
        if 'Q_raw_inf' in decision_inputs:
            decision_inputs['flow_rate'] = decision_inputs.pop('Q_raw_inf')
        print("   - Loaded optimal decision variables.")
        
        df_influent = xls['default_influent_quality']
        influent_inputs = {f"inf_{k}": v for k, v in df_influent.set_index('Variable')['Value (mg/L)'].to_dict().items()}
        print("   - Loaded default influent concentrations.")
        
        df_predicted_effluent = xls['optimal_predicted_effluent']
        print("   - Loaded predicted effluent quality for validation.")

        all_inputs = {**decision_inputs, **influent_inputs}
        return all_inputs, df_predicted_effluent

    except KeyError as e:
        raise KeyError(f"A required sheet {e} was not found in '{filepath}'.")

def load_aao_inputs(optimization_filepath: str, initial_cond_filepath: str):
    """
    Loads inputs specifically for the AAO WWTP model from separate files.
    """
    print(f"1. Loading inputs for AAO model...")
    print(f"   - Loading optimization results from '{optimization_filepath}'")
    if not os.path.exists(optimization_filepath):
        raise FileNotFoundError(f"The optimization results file was not found at '{optimization_filepath}'.")

    print(f"   - Loading initial conditions from '{initial_cond_filepath}'")
    if not os.path.exists(initial_cond_filepath):
        raise FileNotFoundError(f"The initial conditions file was not found at '{initial_cond_filepath}'.")

    try:
        # Load data from the optimization results file
        xls_opt = pd.read_excel(optimization_filepath, sheet_name=None)
        
        df_dec_vars = xls_opt['optimal_decision_variables']
        decision_inputs = {}
        generic_vars_needing_unit = ['KLa', 'V']

        for _, row in df_dec_vars.iterrows():
            variable = row['Variable']
            unit = row['Process Unit']
            value = row['Optimal Value']
            
            if variable in generic_vars_needing_unit:
                key = f"{variable}_{unit}"
            else:
                key = variable
            decision_inputs[key] = value

        # Rename Q_raw_inf to flow_rate for compatibility with simulation functions
        if 'Q_raw_inf' in decision_inputs:
            decision_inputs['flow_rate'] = decision_inputs.pop('Q_raw_inf')
        print("   - Loaded optimal decision variables.")

        df_influent = xls_opt['default_influent_quality']
        influent_inputs = {f"inf_{k}": v for k, v in df_influent.set_index('Variable')['Value (mg/L)'].to_dict().items()}
        print("   - Loaded default influent concentrations.")

        df_predicted_effluent = xls_opt['optimal_predicted_effluent']
        print("   - Loaded predicted effluent quality for validation.")

        # Load initial conditions from the separate data file
        try:
            xls_init = pd.read_excel(initial_cond_filepath, sheet_name='initial_conditions')
            init_cond_df = xls_init.set_index(xls_init.columns[0])
            print("   - Loaded initial conditions.")
        except (KeyError, FileNotFoundError):
            init_cond_df = None
            print("   - WARNING: 'initial_conditions' sheet not found in the initial conditions file. Simulation may fail.")

        all_inputs = {**decision_inputs, **influent_inputs}
        return all_inputs, df_predicted_effluent, init_cond_df

    except KeyError as e:
        raise KeyError(f"A required sheet {e} was not found in '{optimization_filepath}'.")


def _get_simulated_composites(stream):
    """
    Calculate composite properties by first updating the stream's flow rate
    to its final state from the dynamic simulation, then calling the built-in
    properties and methods.
    """
    if stream.state is None or stream.F_mass <= 0:
        return {'BOD': 0, 'COD': 0, 'TN': 0, 'TKN': 0, 'TP': 0, 'TSS': 0, 'VSS': 0}

    # Ensure stream state is updated for composite property calculations
    stream.state
    
    return {
        'BOD': stream.BOD,
        'COD': stream.COD,
        'TN': stream.TN,
        'TKN': stream.TKN,
        'TP': stream.TP,
        'TSS': stream.get_TSS(),
        'VSS': stream.get_VSS(),
    }


def run_single_cstr_simulation(inputs: dict):
    print("\n2. Setting up and running the QSDsan CSTR validation simulation...")
    simulated_effluent = {}
    try:
        cmps = qs.processes.create_asm2d_cmps()
        qs.set_thermo(cmps)
        influent_concentrations = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        ws = qs.WasteStream('influent_ws_validation')
        ws.set_flow_by_concentration(
            flow_tot=inputs['flow_rate'], concentrations=influent_concentrations, units=('m3/d', 'mg/L')
        )
        asm2d_model = qs.processes.ASM2d()
        # Assuming the CSTR being tested is O1 for validation purposes
        aeration_process = qs.processes.DiffusedAeration(
            ID='O1_aeration', DO_ID='S_O2', KLa=inputs['KLa_O1'], DOsat=8.0, V=inputs['V_O1']
        )
        o1_reactor = qs.sanunits.CSTR(
            ID='O1_validation', ins=ws.copy(), V_max=inputs['V_O1'], aeration=aeration_process,
            suspended_growth_model=asm2d_model, DO_ID='S_O2'
        )
        sys = qs.System('sys_validation', path=(o1_reactor,))
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")
        
        effluent_stream = o1_reactor.outs[0]
        for idx, comp_id in enumerate(effluent_stream.components.IDs):
            simulated_effluent[f'Effluent_{comp_id}'] = effluent_stream.conc[idx]
        
        composites = _get_simulated_composites(effluent_stream)
        for key, val in composites.items():
            simulated_effluent[f'Effluent_{key}'] = val

    except Exception as e:
        print(f"\nAN ERROR OCCURRED during the QSDsan simulation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        qs.main_flowsheet.clear()
    return simulated_effluent

def run_single_clarifier_simulation(inputs: dict):
    print("\n2. Setting up and running the QSDsan Clarifier validation simulation...")
    simulated_results = {}
    try:
        cmps = qs.processes.create_asm2d_cmps()
        qs.set_thermo(cmps)
        influent_concentrations = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        clarifier_in = qs.WasteStream('clarifier_in_validation')
        clarifier_in.set_flow_by_concentration(
            flow_tot=inputs['flow_rate'], concentrations=influent_concentrations, units=('m3/d', 'mg/L')
        )
        
        c1_clarifier = qs.sanunits.FlatBottomCircularClarifier(
            ID='C1_validation', ins=clarifier_in, outs=('effluent', 'underflow', 'wastage'),
            surface_area=inputs['C1_surface_area'], height=inputs['C1_height'],
            underflow=inputs['Q_ext'], wastage=inputs['Q_was'],
            N_layer=10, feed_layer=5, X_threshold=3000, v_max=474,
            v_max_practical=250, rh=5.76e-4, rp=2.86e-3, fns=2.28e-3
        )
        sys = qs.System('sys_validation', path=(c1_clarifier,))
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")

        streams_to_report = {'Effluent': c1_clarifier.outs[0], 'Wastage': c1_clarifier.outs[2]}
        for prefix, stream in streams_to_report.items():
            for idx, comp_id in enumerate(stream.components.IDs):
                simulated_results[f'{prefix}_{comp_id}'] = stream.conc[idx]
            
            composites = _get_simulated_composites(stream)
            for key, val in composites.items():
                 simulated_results[f'{prefix}_{key}'] = val

    except Exception as e:
        print(f"\nAN ERROR OCCURRED during the QSDsan simulation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        qs.main_flowsheet.clear()
    return simulated_results

def run_full_aao_simulation(inputs: dict, init_cond_df: pd.DataFrame = None):
    print("\n2. Setting up and running the full A2/O WWTP validation simulation...")
    simulated_results = {}
    try:
        # --- 1. System setup ---
        cmps = qs.processes.create_asm2d_cmps()
        qs.set_thermo(cmps)
        asm2d_model = qs.processes.ASM2d()
        Temp = 273.15 + 20

        # --- 2. Extract operational parameters and create streams ---
        Q_inf = inputs['flow_rate']
        Q_was = inputs['Q_was']
        Q_ext = inputs['Q_ext']
        O3_split_internal = inputs['O3_split_internal']

        influent = qs.WasteStream('influent_validation', T=Temp)
        effluent = qs.WasteStream('effluent_validation', T=Temp)
        int_recycle = qs.WasteStream('internal_recycle_validation', T=Temp)
        ext_recycle = qs.WasteStream('external_recycle_validation', T=Temp)
        wastage = qs.WasteStream('wastage_validation', T=Temp)
        
        influent_concentrations = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        influent.set_flow_by_concentration(Q_inf, concentrations=influent_concentrations, units=('m3/d', 'mg/L'))

        # --- 3. Build the WWTP topology with correct connections ---
        aer_A1 = qs.processes.DiffusedAeration('aer_A1_validation', DO_ID='S_O2', KLa=inputs['KLa_A1'], DOsat=8.0, V=inputs['V_A1'])
        aer_A2 = qs.processes.DiffusedAeration('aer_A2_validation', DO_ID='S_O2', KLa=inputs['KLa_A2'], DOsat=8.0, V=inputs['V_A2'])
        aer_O1 = qs.processes.DiffusedAeration('aer_O1_validation', DO_ID='S_O2', KLa=inputs['KLa_O1'], DOsat=8.0, V=inputs['V_O1'])
        aer_O2 = qs.processes.DiffusedAeration('aer_O2_validation', DO_ID='S_O2', KLa=inputs['KLa_O2'], DOsat=8.0, V=inputs['V_O2'])
        aer_O3 = qs.processes.DiffusedAeration('aer_O3_validation', DO_ID='S_O2', KLa=inputs['KLa_O3'], DOsat=8.0, V=inputs['V_O3'])
        
        A1 = qs.sanunits.CSTR('A1_validation', ins=[influent, int_recycle, ext_recycle], V_max=inputs['V_A1'], aeration=aer_A1, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        A2 = qs.sanunits.CSTR('A2_validation', ins=A1-0, V_max=inputs['V_A2'], aeration=aer_A2, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O1 = qs.sanunits.CSTR('O1_validation', ins=A2-0, V_max=inputs['V_O1'], aeration=aer_O1, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O2 = qs.sanunits.CSTR('O2_validation', ins=O1-0, V_max=inputs['V_O2'], aeration=aer_O2, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O3 = qs.sanunits.CSTR('O3_validation', ins=O2-0, outs=[int_recycle, 'treated'], split=[O3_split_internal, 1 - O3_split_internal], V_max=inputs['V_O3'], aeration=aer_O3, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        
        C1 = qs.sanunits.FlatBottomCircularClarifier(
            'C1_validation', ins=O3-1, outs=[effluent, ext_recycle, wastage],
            surface_area=inputs['C1_surface_area'], height=inputs['C1_height'],
            underflow=Q_ext, wastage=Q_was
        )

        sys = qs.System('sys_validation', path=(A1, A2, O1, O2, O3, C1), recycle=(int_recycle, ext_recycle))

        # --- 4. Set initial conditions ---
        if init_cond_df is not None:
            print("   - Applying initial conditions...")
            def batch_init_static(system_obj, dataframe_with_init_cond):
                dct = dataframe_with_init_cond.to_dict('index')
                u = system_obj.flowsheet.unit
                for unit_id in ['A1', 'A2', 'O1', 'O2', 'O3']:
                    unit_obj = getattr(u, f'{unit_id}_validation')
                    valid_concs = {comp: conc for comp, conc in dct[unit_id].items() if not pd.isna(conc)}
                    if valid_concs: unit_obj.set_init_conc(**valid_concs)
                clarifier_obj = getattr(u, 'C1_validation')
                c1s = {k: v for k, v in dct.get('C1_s', {}).items() if not pd.isna(v) and v > 0}
                c1x = {k: v for k, v in dct.get('C1_x', {}).items() if not pd.isna(v) and v > 0}
                tss = [v for v in dct.get('C1_tss', {}).values() if not pd.isna(v) and v > 0]
                if c1s: clarifier_obj.set_init_solubles(**c1s)
                if c1x: clarifier_obj.set_init_sludge_solids(**c1x)
                if tss: clarifier_obj.set_init_TSS(tss)
            batch_init_static(sys, init_cond_df)

        # --- 5. Run the simulation ---
        sys.set_tolerance(rmol=1e-6)
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")
        
        # --- 6. Collect results from all relevant streams ---
        streams_to_report = {
            'A1_eff': A1.outs[0],
            'A2_eff': A2.outs[0],
            'O1_eff': O1.outs[0],
            'O2_eff': O2.outs[0],
            'O3_eff': O3.outs[1], # Corrected outlet: stream to clarifier
            'Effluent': C1.outs[0],
            'Wastage': C1.outs[2],
        }
        for prefix, stream in streams_to_report.items():
            for idx, comp_id in enumerate(stream.components.IDs):
                simulated_results[f'{prefix}_{comp_id}'] = stream.conc[idx]
            composites = _get_simulated_composites(stream)
            for key, val in composites.items():
                 simulated_results[f'{prefix}_{key}'] = val

    except Exception as e:
        print(f"\nAN ERROR OCCURRED during the QSDsan simulation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        qs.main_flowsheet.clear()
        
    return simulated_results

def display_comparison_report(predicted_df: pd.DataFrame, simulated_dict: dict):
    print("\n3. Comparing Surrogate Model Predictions vs. Detailed QSDsan Simulation:")
    print("=" * 80)
    comparison_data = []

    for _, row in predicted_df.iterrows():
        pred_key = row['Component']
        sim_key = None
        
        # Parse the prediction key to build the corresponding simulation key
        if pred_key.endswith('_Effluent'):
            prop_id = pred_key.replace('_Effluent', '')
            sim_key = f"Effluent_{prop_id}"
        elif pred_key.endswith('_Wastage'):
            prop_id = pred_key.replace('_Wastage', '')
            sim_key = f"Wastage_{prop_id}"
        else: # Assumes reactor unit (e.g., S_A_A1)
            try:
                parts = pred_key.split('_')
                unit_id = parts[-1]
                prop_id = '_'.join(parts[:-1])
                sim_key = f"{unit_id}_eff_{prop_id}"
            except IndexError:
                continue

        if sim_key and sim_key in simulated_dict:
            predicted_val = row['Predicted Value (mg/L)']
            simulated_val = simulated_dict[sim_key]
            difference = simulated_val - predicted_val
            relative_diff = (difference / predicted_val * 100) if abs(predicted_val) > 1e-6 else np.nan
            comparison_data.append({
                'Component': pred_key,
                'Predicted (Surrogate) mg/L': predicted_val,
                'Simulated (QSDsan) mg/L': simulated_val,
                'Absolute Difference': difference,
                'Relative Difference (%)': relative_diff
            })
            
    if not comparison_data:
        print("Could not generate comparison report. No matching components found.")
        print("\nDebug Info:")
        print("Example prediction keys:", predicted_df['Component'].head().tolist())
        print("Example simulation keys:", list(simulated_dict.keys())[:5])
        return
        
    df_report = pd.DataFrame(comparison_data)
    pd.options.display.float_format = '{:,.4f}'.format
    print(df_report.to_string(index=False))
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

def run_cstr_validation():
    data_filepath = os.path.join('data', 'optimization_results.xlsx')
    try:
        optimal_inputs, predicted_df = load_simple_inputs(data_filepath)
        simulated_concs = run_single_cstr_simulation(optimal_inputs)
        if simulated_concs:
            display_comparison_report(predicted_df, simulated_concs)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nVALIDATION FAILED: {e}")

def run_clarifier_validation():
    data_filepath = os.path.join('data', 'optimization_results.xlsx')
    try:
        optimal_inputs, predicted_df = load_simple_inputs(data_filepath)
        simulated_results = run_single_clarifier_simulation(optimal_inputs)
        if simulated_results:
            display_comparison_report(predicted_df, simulated_results)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nVALIDATION FAILED: {e}")

def run_aao_validation():
    optimization_filepath = os.path.join('data', 'optimization_results.xlsx')
    initial_cond_filepath = os.path.join('data', 'data.xlsx')
    try:
        optimal_inputs, predicted_df, init_cond_df = load_aao_inputs(
            optimization_filepath, initial_cond_filepath
        )
        simulated_results = run_full_aao_simulation(optimal_inputs, init_cond_df)
        if simulated_results:
            display_comparison_report(predicted_df, simulated_results)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nVALIDATION FAILED: {e}")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Wastewater Treatment Plant Model Validation")
    print("-" * 40)
    while True:
        choice = input(
            "Which model would you like to validate?\n"
            "1. cstr\n"
            "2. clarifier\n"
            "3. aao\n"
            "Enter your choice: "
        ).lower().strip()
        
        if choice in ['1', 'cstr']:
            run_cstr_validation()
            break
        elif choice in ['2', 'clarifier']:
            run_clarifier_validation()
            break
        elif choice in ['3', 'aao']:
            run_aao_validation()
            break
        else:
            print("\nInvalid choice. Please enter 'cstr', 'clarifier', or 'aao' (or 1, 2, 3).")

if __name__ == '__main__':
    main()