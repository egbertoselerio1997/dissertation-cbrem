# File: run_validation_simulation.py
# Description: This script runs a single QSDsan simulation using optimal
#              parameters to validate a surrogate model's predictions.

import os
import qsdsan as qs
import pandas as pd
import numpy as np
import warnings
import traceback # Added for detailed error logging

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_simple_inputs(filepath: str):
    """
    Loads inputs for simple, single-unit models (e.g., isolated CSTR) from the optimization results file.
    """
    print(f"1. Loading inputs for simple model from '{filepath}'...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file was not found at '{filepath}'.")

    try:
        xls = pd.read_excel(filepath, sheet_name=None)
        df_dec_vars = xls['optimal_decision_variables']
        decision_inputs = df_dec_vars.set_index('Variable')['Optimal Value'].to_dict()
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
    Loads inputs for complex flowsheet models (AAO, AS Plant) from separate files.
    """
    print(f"1. Loading inputs for flowsheet model...")
    print(f"   - Loading optimization results from '{optimization_filepath}'")
    if not os.path.exists(optimization_filepath):
        raise FileNotFoundError(f"The optimization results file was not found at '{optimization_filepath}'.")

    print(f"   - Loading initial conditions from '{initial_cond_filepath}'")
    if not os.path.exists(initial_cond_filepath):
        raise FileNotFoundError(f"The initial conditions file was not found at '{initial_cond_filepath}'.")

    try:
        xls_opt = pd.read_excel(optimization_filepath, sheet_name=None)
        df_dec_vars = xls_opt['optimal_decision_variables']
        decision_inputs = {}
        generic_vars_needing_unit = ['KLa', 'V']

        for _, row in df_dec_vars.iterrows():
            variable, unit, value = row['Variable'], row['Process Unit'], row['Optimal Value']
            key = f"{variable}_{unit}" if variable in generic_vars_needing_unit else variable
            decision_inputs[key] = value

        if 'Q_raw_inf' in decision_inputs:
            decision_inputs['flow_rate'] = decision_inputs.pop('Q_raw_inf')
        print("   - Loaded optimal decision variables.")

        df_influent = xls_opt['default_influent_quality']
        influent_inputs = {f"inf_{k}": v for k, v in df_influent.set_index('Variable')['Value (mg/L)'].to_dict().items()}
        print("   - Loaded default influent concentrations.")

        df_predicted_effluent = xls_opt['optimal_predicted_effluent']
        print("   - Loaded predicted effluent quality for validation.")

        try:
            xls_init = pd.read_excel(initial_cond_filepath, sheet_name='initial_conditions')
            init_cond_df = xls_init.set_index(xls_init.columns[0])
            print("   - Loaded initial conditions.")
        except (KeyError, FileNotFoundError):
            init_cond_df = None
            print("   - WARNING: 'initial_conditions' sheet not found. Simulation may fail.")

        all_inputs = {**decision_inputs, **influent_inputs}
        return all_inputs, df_predicted_effluent, init_cond_df

    except KeyError as e:
        raise KeyError(f"A required sheet {e} was not found in '{optimization_filepath}'.")


def _get_simulated_composites(stream):
    """Calculate composite properties from a QSDsan stream."""
    if stream.state is None or stream.F_mass <= 0: return {'BOD': 0, 'COD': 0, 'TN': 0, 'TKN': 0, 'TP': 0, 'TSS': 0, 'VSS': 0}
    stream.state
    return {'BOD': stream.BOD, 'COD': stream.COD, 'TN': stream.TN, 'TKN': stream.TKN, 'TP': stream.TP, 'TSS': stream.get_TSS(), 'VSS': stream.get_VSS()}


def run_single_cstr_simulation(inputs: dict):
    print("\n2. Setting up and running the QSDsan CSTR validation simulation...")
    simulated_effluent = {}
    raw_influent = None
    try:
        cmps = qs.processes.create_asm2d_cmps(); qs.set_thermo(cmps)
        influent_concentrations = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        ws = qs.WasteStream('influent_ws_validation')
        ws.set_flow_by_concentration(inputs['flow_rate'], concentrations=influent_concentrations, units=('m3/d', 'mg/L'))
        raw_influent = ws
        asm2d_model = qs.processes.ASM2d()
        aeration_process = qs.processes.DiffusedAeration('CSTR_aeration', DO_ID='S_O2', KLa=inputs['KLa'], DOsat=8.0, V=inputs['V'])
        cstr_reactor = qs.sanunits.CSTR('CSTR_validation', ins=ws.copy(), V_max=inputs['V'], aeration=aeration_process, suspended_growth_model=asm2d_model, DO_ID='S_O2')
        sys = qs.System('sys_validation', path=(cstr_reactor,))
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")
        
        effluent_stream = cstr_reactor.outs[0]
        for idx, comp_id in enumerate(effluent_stream.components.IDs):
            simulated_effluent[f'{comp_id}_Effluent'] = effluent_stream.conc[idx]
        composites = _get_simulated_composites(effluent_stream)
        for key, val in composites.items():
            simulated_effluent[f'{key}_Effluent'] = val
    except Exception as e:
        print(f"\nAN ERROR OCCURRED during the QSDsan simulation: {e}"); traceback.print_exc(); return None, None
    finally:
        qs.main_flowsheet.clear()
    return simulated_effluent, raw_influent

def run_as_plant_simulation(inputs: dict, init_cond_df: pd.DataFrame = None):
    print("\n2. Setting up and running the AS Plant (CSTR+Clarifier) validation simulation...")
    simulated_results = {}
    raw_influent = None
    try:
        cmps = qs.processes.create_asm2d_cmps(); qs.set_thermo(cmps)
        asm2d_model = qs.processes.ASM2d(); Temp = 273.15 + 20
        Q_inf, Q_was, Q_ext = inputs['flow_rate'], inputs['Q_was'], inputs['Q_ext']
        CSTR1_split_internal = inputs['CSTR1_split_internal']

        influent, effluent, int_recycle, ext_recycle, wastage = (qs.WasteStream(ID, T=Temp) for ID in 
            ['influent_validation', 'effluent_validation', 'internal_recycle_validation', 'external_recycle_validation', 'wastage_validation'])
        
        influent_concentrations = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        influent.set_flow_by_concentration(Q_inf, concentrations=influent_concentrations, units=('m3/d', 'mg/L'))
        raw_influent = influent

        aer_CSTR1 = qs.processes.DiffusedAeration('aer_CSTR1_validation', DO_ID='S_O2', KLa=inputs['KLa_CSTR1'], DOsat=8.0, V=inputs['V_CSTR1'])
        CSTR1 = qs.sanunits.CSTR('CSTR1_validation', ins=[influent, int_recycle, ext_recycle], outs=['treated_cstr', int_recycle], 
                                split=[1 - CSTR1_split_internal, CSTR1_split_internal], V_max=inputs['V_CSTR1'], 
                                aeration=aer_CSTR1, DO_ID='S_O2', suspended_growth_model=asm2d_model)
        C1 = qs.sanunits.FlatBottomCircularClarifier('C1_validation', ins=CSTR1-0, outs=[effluent, ext_recycle, wastage],
                                                    surface_area=inputs['C1_surface_area'], height=inputs['C1_height'],
                                                    underflow=Q_ext, wastage=Q_was)
        sys = qs.System('sys_validation', path=(CSTR1, C1), recycle=(int_recycle, ext_recycle))

        if init_cond_df is not None:
            print("   - Applying initial conditions...")
            def batch_init_static(system_obj, df):
                dct, u = df.to_dict('index'), system_obj.flowsheet.unit
                if 'CSTR1' in dct:
                    concs = {c: v for c, v in dct['CSTR1'].items() if pd.notna(v)}
                    if concs: getattr(u, 'CSTR1_validation').set_init_conc(**concs)
                c1s = {k: v for k, v in dct.get('C1_s', {}).items() if pd.notna(v) and v > 0}
                c1x = {k: v for k, v in dct.get('C1_x', {}).items() if pd.notna(v) and v > 0}
                tss = [v for v in dct.get('C1_tss', {}).values() if pd.notna(v) and v > 0]
                clarifier = getattr(u, 'C1_validation')
                if c1s: clarifier.set_init_solubles(**c1s)
                if c1x: clarifier.set_init_sludge_solids(**c1x)
                if tss: clarifier.set_init_TSS(tss)
            
            df_as_init = init_cond_df.rename(index={'A1': 'CSTR1'}) if 'A1' in init_cond_df.index else init_cond_df
            batch_init_static(sys, df_as_init)

        sys.set_tolerance(rmol=1e-6)
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")
        
        streams_to_report = {'CSTR1': CSTR1.outs[0], 'Effluent': C1.outs[0], 'Wastage': C1.outs[2]}
        for prefix, stream in streams_to_report.items():
            for idx, comp_id in enumerate(stream.components.IDs):
                simulated_results[f'{comp_id}_{prefix}'] = stream.conc[idx]
            composites = _get_simulated_composites(stream)
            for key, val in composites.items():
                 simulated_results[f'{key}_{prefix}'] = val
    except Exception as e:
        print(f"\nAN ERROR OCCURRED during the QSDsan simulation: {e}"); traceback.print_exc(); return None, None
    finally:
        qs.main_flowsheet.clear()
    return simulated_results, raw_influent

def run_full_aao_simulation(inputs: dict, init_cond_df: pd.DataFrame = None):
    print("\n2. Setting up and running the full A2/O WWTP validation simulation...")
    simulated_results = {}
    raw_influent = None
    try:
        cmps = qs.processes.create_asm2d_cmps(); qs.set_thermo(cmps)
        asm2d_model = qs.processes.ASM2d(); Temp = 273.15 + 20
        Q_inf, Q_was, Q_ext = inputs['flow_rate'], inputs['Q_was'], inputs['Q_ext']
        O3_split_internal = inputs['O3_split_internal']

        influent, effluent, int_recycle, ext_recycle, wastage = (qs.WasteStream(ID, T=Temp) for ID in 
            ['influent_validation', 'effluent_validation', 'internal_recycle_validation', 'external_recycle_validation', 'wastage_validation'])
        influent_concentrations = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        influent.set_flow_by_concentration(Q_inf, concentrations=influent_concentrations, units=('m3/d', 'mg/L'))
        raw_influent = influent

        aerations = {uid: qs.processes.DiffusedAeration(f'aer_{uid}_validation', DO_ID='S_O2', KLa=inputs[f'KLa_{uid}'], DOsat=8.0, V=inputs[f'V_{uid}'])
                     for uid in ['A1', 'A2', 'O1', 'O2', 'O3']}
        
        A1 = qs.sanunits.CSTR('A1_validation', ins=[influent, int_recycle, ext_recycle], V_max=inputs['V_A1'], aeration=aerations['A1'], DO_ID='S_O2', suspended_growth_model=asm2d_model)
        A2 = qs.sanunits.CSTR('A2_validation', ins=A1-0, V_max=inputs['V_A2'], aeration=aerations['A2'], DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O1 = qs.sanunits.CSTR('O1_validation', ins=A2-0, V_max=inputs['V_O1'], aeration=aerations['O1'], DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O2 = qs.sanunits.CSTR('O2_validation', ins=O1-0, V_max=inputs['V_O2'], aeration=aerations['O2'], DO_ID='S_O2', suspended_growth_model=asm2d_model)
        O3 = qs.sanunits.CSTR('O3_validation', ins=O2-0, outs=['treated', int_recycle], split=[1 - O3_split_internal, O3_split_internal], V_max=inputs['V_O3'], aeration=aerations['O3'], DO_ID='S_O2', suspended_growth_model=asm2d_model)
        C1 = qs.sanunits.FlatBottomCircularClarifier('C1_validation', ins=O3-0, outs=[effluent, ext_recycle, wastage],
                                                    surface_area=inputs['C1_surface_area'], height=inputs['C1_height'],
                                                    underflow=Q_ext, wastage=Q_was)
        sys = qs.System('sys_validation', path=(A1, A2, O1, O2, O3, C1), recycle=(int_recycle, ext_recycle))

        if init_cond_df is not None:
            print("   - Applying initial conditions...")
            def batch_init_static(system_obj, df):
                dct, u = df.to_dict('index'), system_obj.flowsheet.unit
                for unit_id in ['A1', 'A2', 'O1', 'O2', 'O3']:
                    concs = {c: v for c, v in dct[unit_id].items() if pd.notna(v)}
                    if concs: getattr(u, f'{unit_id}_validation').set_init_conc(**concs)
                clarifier = getattr(u, 'C1_validation')
                c1s = {k: v for k, v in dct.get('C1_s', {}).items() if pd.notna(v) and v > 0}
                c1x = {k: v for k, v in dct.get('C1_x', {}).items() if pd.notna(v) and v > 0}
                tss = [v for v in dct.get('C1_tss', {}).values() if pd.notna(v) and v > 0]
                if c1s: clarifier.set_init_solubles(**c1s); 
                if c1x: clarifier.set_init_sludge_solids(**c1x); 
                if tss: clarifier.set_init_TSS(tss)
            batch_init_static(sys, init_cond_df)

        sys.set_tolerance(rmol=1e-6)
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")
        
        streams_to_report = {'A1': A1.outs[0], 'A2': A2.outs[0], 'O1': O1.outs[0], 'O2': O2.outs[0], 
                             'O3': O3.outs[0], 'Effluent': C1.outs[0], 'Wastage': C1.outs[2]}
        for prefix, stream in streams_to_report.items():
            for idx, comp_id in enumerate(stream.components.IDs):
                simulated_results[f'{comp_id}_{prefix}'] = stream.conc[idx]
            composites = _get_simulated_composites(stream)
            for key, val in composites.items():
                 simulated_results[f'{key}_{prefix}'] = val
    except Exception as e:
        print(f"\nAN ERROR OCCURRED during the QSDsan simulation: {e}"); traceback.print_exc(); return None, None
    finally:
        qs.main_flowsheet.clear()
    return simulated_results, raw_influent

def display_comparison_report(predicted_df: pd.DataFrame, simulated_dict: dict):
    print("\n3. Comparing Surrogate Model Predictions vs. Detailed QSDsan Simulation:"); print("=" * 80)
    comparison_data = []
    for _, row in predicted_df.iterrows():
        pred_key = row['Component']
        sim_key = pred_key # Keys are designed to match directly
        if sim_key in simulated_dict:
            predicted_val = row['Predicted Value (mg/L)']
            simulated_val = simulated_dict[sim_key]
            difference = simulated_val - predicted_val
            relative_diff = (difference / predicted_val * 100) if abs(predicted_val) > 1e-6 else np.nan
            comparison_data.append({'Component': pred_key, 'Predicted (Surrogate) mg/L': predicted_val,
                                     'Simulated (QSDsan) mg/L': simulated_val, 'Absolute Difference': difference,
                                     'Relative Difference (%)': relative_diff})
            
    if not comparison_data:
        print("Could not generate comparison report. No matching components found.")
        print("\nDebug Info:\nExample prediction keys:", predicted_df['Component'].head().tolist())
        print("Example simulation keys:", list(simulated_dict.keys())[:5]); return
        
    df_report = pd.DataFrame(comparison_data)
    pd.options.display.float_format = '{:,.4f}'.format
    print(df_report.to_string(index=False))
    mape = df_report['Relative Difference (%)'].abs().mean()
    print("-" * 80); print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%"); print("=" * 80)
    if mape < 5: print("\nConclusion: Excellent agreement. The surrogate model is highly accurate.")
    elif mape < 15: print("\nConclusion: Good agreement. The surrogate model is reliable.")
    else: print("\nConclusion: Moderate to poor agreement. The surrogate model shows deviation.")

def run_cstr_validation():
    data_filepath = os.path.join('data', 'optimization_results.xlsx')
    try:
        optimal_inputs, predicted_df = load_simple_inputs(data_filepath)
        simulated_concs, raw_influent = run_single_cstr_simulation(optimal_inputs)
        if simulated_concs: 
            display_comparison_report(predicted_df, simulated_concs)
            if raw_influent:
                print("\n\nRaw Influent Characteristics (QSDsan Calculation):")
                print("-" * 55)
                raw_influent.show()
                print("-" * 55)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nVALIDATION FAILED: {e}")

def run_clarifier_validation():
    optimization_filepath = os.path.join('data', 'optimization_results.xlsx')
    initial_cond_filepath = os.path.join('data', 'data.xlsx')
    try:
        optimal_inputs, predicted_df, init_cond_df = load_aao_inputs(optimization_filepath, initial_cond_filepath)
        simulated_results, raw_influent = run_as_plant_simulation(optimal_inputs, init_cond_df)
        if simulated_results: 
            display_comparison_report(predicted_df, simulated_results)
            if raw_influent:
                print("\n\nRaw Influent Characteristics (QSDsan Calculation):")
                print("-" * 55)
                raw_influent.show()
                print("-" * 55)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nVALIDATION FAILED: {e}")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}"); traceback.print_exc()

def run_aao_validation():
    optimization_filepath = os.path.join('data', 'optimization_results.xlsx')
    initial_cond_filepath = os.path.join('data', 'data.xlsx')
    try:
        optimal_inputs, predicted_df, init_cond_df = load_aao_inputs(optimization_filepath, initial_cond_filepath)
        simulated_results, raw_influent = run_full_aao_simulation(optimal_inputs, init_cond_df)
        if simulated_results: 
            display_comparison_report(predicted_df, simulated_results)
            if raw_influent:
                print("\n\nRaw Influent Characteristics (QSDsan Calculation):")
                print("-" * 55)
                raw_influent.show()
                print("-" * 55)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nVALIDATION FAILED: {e}")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}"); traceback.print_exc()

def main():
    print("Wastewater Treatment Plant Model Validation"); print("-" * 40)
    while True:
        choice = input("Which model would you like to validate?\n1. cstr (Single CSTR unit)\n2. clarifier (CSTR + Clarifier Plant)\n3. aao (Full AAO plant)\nEnter your choice: ").lower().strip()
        if choice in ['1', 'cstr']: run_cstr_validation(); break
        elif choice in ['2', 'clarifier']: run_clarifier_validation(); break
        elif choice in ['3', 'aao']: run_aao_validation(); break
        else: print("\nInvalid choice. Please enter 'cstr', 'clarifier', or 'aao' (or 1, 2, 3).")

if __name__ == '__main__':
    main()