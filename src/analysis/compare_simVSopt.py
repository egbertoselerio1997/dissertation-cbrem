import os
import qsdsan as qs
import pandas as pd
import numpy as np
import warnings
import traceback
import matplotlib.pyplot as plt

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Plotting Configuration ---
COMPONENT_MAP = { # Symbol: Readable Name
    'S_O2': 'Dissolved Oxygen', 'S_N2': 'Dinitrogen', 'S_NH4': 'Ammonia',
    'S_NO3': 'Nitrate and Nitrite', 'S_PO4': 'Orthophosphates',
    'S_F': 'Fermentable Organics', 'S_A': 'Acetate',
    'S_I': 'Soluble Inert Organics', 'S_ALK': 'Alkalinity',
    'X_I': 'Particulate Inert Organics', 'X_S': 'Slowly Biodegradable Substrate',
    'X_H': 'Heterotrophic Organisms', 'X_PAO': 'Phosphate-Accumulating Organisms',
    'X_PP': 'Poly-phosphate', 'X_PHA': 'Polyhydroxyalkanoates',
    'X_AUT': 'Nitrifying Organisms', 'X_MeOH': 'Metal Hydroxides', 'X_MeP': 'Metal Phosphate'
}

COMPOSITE_MAP = { # Symbol: Readable Name
    'BOD': 'Biochemical Oxygen Demand', 'COD': 'Chemical Oxygen Demand', 'TN': 'Total Nitrogen',
    'TKN': 'Total Kjahl Nitrogen', 'TP': 'Total Phosphorus',
    'TSS': 'Total Suspended Solids', 'VSS': 'Volatile Suspended Solids'
}

def generate_and_save_plots(system, sim_type, tracked_units):
    """
    Generates and saves time-series plots for components and composites for tracked units.
    """
    print(f"   - Generating and saving time-series plots for {sim_type} simulation...")
    base_plot_dir = os.path.join('plots', 'comparison', sim_type)
    
    for unit in tracked_units:
        # Ensure the unit has a scope with recorded data
        if not hasattr(unit, 'scope') or unit.scope.record is None or unit.scope.record.shape[0] == 0:
            print(f"     - Skipping plots for {unit.ID}: No dynamic data recorded.")
            continue
            
        unit_name = unit.ID.replace('_validation', '')
        unit_dir = os.path.join(base_plot_dir, unit_name)
        comp_dir = os.path.join(unit_dir, 'components')
        composite_dir = os.path.join(unit_dir, 'composites')
        os.makedirs(comp_dir, exist_ok=True)
        os.makedirs(composite_dir, exist_ok=True)
        
        scope = unit.scope
        time = scope.time_series
        components = qs.get_thermo().chemicals

        # Plot components
        print(f"     - Plotting time-series components for {unit_name}...")
        for i, comp_id in enumerate(components.IDs):
            if comp_id == 'H2O': continue
            data = scope.record[:, i]
            readable_name = COMPONENT_MAP.get(comp_id, comp_id)
            
            plt.figure(figsize=(10, 6))
            plt.plot(time, data)
            plt.title(f'{readable_name} ({comp_id}) in {unit_name}')
            plt.xlabel('Time [d]')
            plt.ylabel('Concentration [mg/L]')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f'{comp_id}.png'))
            plt.close()

        # Plot composites
        print(f"     - Plotting time-series composites for {unit_name}...")
        temp_ws = qs.WasteStream('temp_plot_stream')
        composite_data = {key: [] for key in COMPOSITE_MAP.keys()}
        
        for state_vector in scope.record:
            concentrations = state_vector[:-1]
            flow_rate_m3d = state_vector[-1]
            
            if flow_rate_m3d > 1e-9:
                conc_dict = dict(zip(components.IDs, concentrations))
                temp_ws.set_flow_by_concentration(
                    flow_tot=flow_rate_m3d,
                    concentrations=conc_dict,
                    units=('m3/d', 'mg/L')
                )
            else:
                temp_ws.empty()
            
            if temp_ws.F_mass <= 0:
                for key in COMPOSITE_MAP.keys():
                    composite_data[key].append(0)
            else:
                composite_data['BOD'].append(temp_ws.BOD)
                composite_data['COD'].append(temp_ws.COD)
                composite_data['TN'].append(temp_ws.TN)
                composite_data['TKN'].append(temp_ws.TKN)
                composite_data['TP'].append(temp_ws.TP)
                composite_data['TSS'].append(temp_ws.get_TSS())
                composite_data['VSS'].append(temp_ws.get_VSS())

        for comp_id, data in composite_data.items():
            readable_name = COMPOSITE_MAP.get(comp_id, comp_id)
            
            plt.figure(figsize=(10, 6))
            plt.plot(time, data)
            plt.title(f'{readable_name} ({comp_id}) in {unit_name}')
            plt.xlabel('Time [d]')
            plt.ylabel('Concentration [mg/L]')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(composite_dir, f'{comp_id}.png'))
            plt.close()
            
    print("   - Time-series plot generation complete.")


def generate_treatment_train_plots(sim_type, stages, stage_names, predicted_df):
    """
    Generates and saves plots showing concentration profiles across the treatment train,
    comparing simulation against multiple surrogate model predictions.
    """
    print(f"   - Generating and saving treatment train plots for {sim_type} simulation...")
    train_dir = os.path.join('plots', 'comparison', sim_type, 'treatment_train')
    comp_dir = os.path.join(train_dir, 'components')
    composite_dir = os.path.join(train_dir, 'composites')
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(composite_dir, exist_ok=True)

    components = qs.get_thermo().chemicals

    # --- Load additional surrogate model predictions ---
    other_surrogates_data = {}
    surrogate_predictions_path = os.path.join('data', 'surrogate_predictions', sim_type, 'surrogate_model_predictions.xlsx')
    try:
        if os.path.exists(surrogate_predictions_path):
            other_surrogates_data = pd.read_excel(surrogate_predictions_path, sheet_name=None)
            other_surrogates_data.pop('optimal_predicted_effluent', None)
            print(f"     - Loaded additional surrogate predictions from {len(other_surrogates_data)} models.")
    except Exception as e:
        print(f"     - WARNING: Could not load or parse additional surrogate predictions from '{surrogate_predictions_path}'. Reason: {e}")
        other_surrogates_data = {}

    # Mapping from plot stage names to prefixes used in the prediction DataFrame
    stage_prefix_map = {
        'Raw Influent': None, # No prediction for input
        'A1': 'A1', 'A2': 'A2', 'O1': 'O1', 'O2': 'O2', 'O3': 'O3',
        'CSTR1 Effluent': 'CSTR1',
        'Final Effluent': 'Effluent',
    }

    # Helper function to extract predictions for a given component from a dataframe
    def get_predictions_for_component(df, comp_id, stages_map, stage_name_list):
        predictions = []
        for stage in stage_name_list:
            prefix = stages_map.get(stage)
            if not prefix:
                predictions.append(np.nan)
                continue
            
            pred_key = f'{comp_id}_{prefix}'
            pred_row = df[df['Component'] == pred_key]
            pred_val = pred_row['Predicted Value (mg/L)'].iloc[0] if not pred_row.empty else np.nan
            predictions.append(pred_val)
        return predictions

    # Plot components across the train
    print("     - Plotting components across the treatment train...")
    for i, comp_id in enumerate(components.IDs):
        if comp_id == 'H2O': continue
        
        simulated_concentrations = [stream.conc[i] for stream in stages]
        readable_name = COMPONENT_MAP.get(comp_id, comp_id)

        optimal_predicted_concentrations = get_predictions_for_component(predicted_df, comp_id, stage_prefix_map, stage_names)
        
        plt.figure(figsize=(12, 7))
        plt.plot(stage_names, simulated_concentrations, marker='o', linestyle='-', label='Simulated (QSDsan)', zorder=10, lw=2.5)
        plt.plot(stage_names, optimal_predicted_concentrations, marker='x', linestyle='--', label='Optimization Model', zorder=5, lw=1.5)

        for model_name, model_df in other_surrogates_data.items():
            other_surrogate_preds = get_predictions_for_component(model_df, comp_id, stage_prefix_map, stage_names)
            plt.plot(stage_names, other_surrogate_preds, marker='.', linestyle=':', label=f'{model_name.upper()} Surrogate', alpha=0.8, zorder=3)

        plt.title(f'Treatment Train Profile for {readable_name} ({comp_id})')
        plt.xlabel('Treatment Stage')
        plt.ylabel('Concentration [mg/L]')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, f'{comp_id}.png'))
        plt.close()

    # Plot composites across the train
    print("     - Plotting composites across the treatment train...")
    for comp_id, readable_name in COMPOSITE_MAP.items():
        simulated_concentrations = [_get_simulated_composites(stream)[comp_id] for stream in stages]
        
        optimal_predicted_concentrations = get_predictions_for_component(predicted_df, comp_id, stage_prefix_map, stage_names)
            
        plt.figure(figsize=(12, 7))
        plt.plot(stage_names, simulated_concentrations, marker='o', linestyle='-', label='Simulated (QSDsan)', zorder=10, lw=2.5)
        plt.plot(stage_names, optimal_predicted_concentrations, marker='x', linestyle='--', label='Optimization Model', zorder=5, lw=1.5)

        for model_name, model_df in other_surrogates_data.items():
            other_surrogate_preds = get_predictions_for_component(model_df, comp_id, stage_prefix_map, stage_names)
            plt.plot(stage_names, other_surrogate_preds, marker='.', linestyle=':', label=f'{model_name.upper()} Surrogate', alpha=0.8, zorder=3)
            
        plt.title(f'Treatment Train Profile for {readable_name} ({comp_id})')
        plt.xlabel('Treatment Stage')
        plt.ylabel('Concentration [mg/L]')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(composite_dir, f'{comp_id}.png'))
        plt.close()

    print("   - Treatment train plot generation complete.")


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


def run_single_cstr_simulation(inputs: dict, sim_type: str, enable_plotting: bool):
    print("\n2. Setting up and running the QSDsan CSTR validation simulation...")
    simulated_effluent = {}
    raw_influent = None
    try:
        cmps = qs.processes.create_asm2d_cmps(); qs.set_thermo(cmps)
        
        all_influent_data = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        valid_component_ids = cmps.IDs
        influent_concentrations = {key: value for key, value in all_influent_data.items() if key in valid_component_ids}

        ws = qs.WasteStream('influent_ws_validation')
        ws.set_flow_by_concentration(inputs['flow_rate'], concentrations=influent_concentrations, units=('m3/d', 'mg/L'))
        raw_influent = ws
        
        asm2d_model = qs.processes.ASM2d()
        aeration_process = qs.processes.DiffusedAeration('CSTR_aeration', DO_ID='S_O2', KLa=inputs['KLa'], DOsat=8.0, V=inputs['V'])
        cstr_reactor = qs.sanunits.CSTR('CSTR_validation', ins=ws.copy(), V_max=inputs['V'], aeration=aeration_process, suspended_growth_model=asm2d_model, DO_ID='S_O2')
        sys = qs.System('sys_validation', path=(cstr_reactor,))
        sys.set_dynamic_tracker(cstr_reactor)
        sys.simulate(t_span=(0, 180), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")
        
        if enable_plotting:
            generate_and_save_plots(sys, sim_type, [cstr_reactor])

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

def run_as_plant_simulation(inputs: dict, sim_type: str, init_cond_df: pd.DataFrame = None, predicted_df: pd.DataFrame = None, enable_plotting: bool = False):
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
        
        all_influent_data = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        valid_component_ids = cmps.IDs
        influent_concentrations = {key: value for key, value in all_influent_data.items() if key in valid_component_ids}
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
        tracked_units = [CSTR1, C1]
        sys.set_dynamic_tracker(*tracked_units)
        sys.simulate(t_span=(0, 180), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")
        
        if enable_plotting:
            generate_and_save_plots(sys, sim_type, tracked_units)
            stages = [raw_influent, CSTR1.outs[0], C1.outs[0]]
            stage_names = ['Raw Influent', 'CSTR1 Effluent', 'Final Effluent']
            generate_treatment_train_plots(sim_type, stages, stage_names, predicted_df)
        
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

def run_full_aao_simulation(inputs: dict, sim_type: str, init_cond_df: pd.DataFrame = None, predicted_df: pd.DataFrame = None, enable_plotting: bool = False):
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
        
        all_influent_data = {k.replace('inf_', ''): v for k, v in inputs.items() if k.startswith('inf_')}
        valid_component_ids = cmps.IDs
        influent_concentrations = {key: value for key, value in all_influent_data.items() if key in valid_component_ids}

        influent.set_flow_by_concentration(Q_inf, concentrations=influent_concentrations, units=('m3/d', 'mg/L'))
        raw_influent = influent

        aerations = {uid: qs.processes.DiffusedAeration(f'aer_{uid}_validation', DO_ID='S_O2', KLa=inputs[f'KLa_{uid}'], DOsat=8.0, V=inputs[f'V_{uid}'])
                     for uid in ['O1', 'O2', 'O3']}
        
        A1 = qs.sanunits.CSTR('A1_validation', ins=[influent, int_recycle, ext_recycle], V_max=inputs['V_A1'], aeration=None, suspended_growth_model=asm2d_model)
        A2 = qs.sanunits.CSTR('A2_validation', ins=A1-0, V_max=inputs['V_A2'], aeration=None, suspended_growth_model=asm2d_model)
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
                    if unit_id in dct:
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
        tracked_units = [A1, A2, O1, O2, O3, C1]
        sys.set_dynamic_tracker(*tracked_units)
        sys.simulate(t_span=(0, 180), method='BDF', state_reset_hook='reset_cache')
        print("   - Simulation completed successfully.")
        
        if enable_plotting:
            generate_and_save_plots(sys, sim_type, tracked_units)
            stages = [raw_influent, A1.outs[0], A2.outs[0], O1.outs[0], O2.outs[0], O3.outs[0], C1.outs[0]]
            stage_names = ['Raw Influent', 'A1', 'A2', 'O1', 'O2', 'O3', 'Final Effluent']
            generate_treatment_train_plots(sim_type, stages, stage_names, predicted_df)

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

def run_cstr_validation(sim_type: str, enable_plotting: bool):
    data_filepath = os.path.join('data', 'optimization_results.xlsx')
    try:
        optimal_inputs, predicted_df = load_simple_inputs(data_filepath)
        simulated_concs, raw_influent = run_single_cstr_simulation(optimal_inputs, sim_type, enable_plotting)
        if simulated_concs: 
            display_comparison_report(predicted_df, simulated_concs)
            if raw_influent:
                print("\n\nRaw Influent Characteristics (QSDsan Calculation):")
                print("-" * 55)
                raw_influent.show()
                print("-" * 55)
    except (FileNotFoundError, KeyError) as e:
        print(f"\nVALIDATION FAILED: {e}")

def run_cstr_and_clarifier_validation(sim_type: str, enable_plotting: bool):
    optimization_filepath = os.path.join('data', 'optimization_results.xlsx')
    initial_cond_filepath = os.path.join('data', 'data.xlsx')
    try:
        optimal_inputs, predicted_df, init_cond_df = load_aao_inputs(optimization_filepath, initial_cond_filepath)
        simulated_results, raw_influent = run_as_plant_simulation(
            optimal_inputs, sim_type, init_cond_df, predicted_df, enable_plotting
        )
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

def run_aao_validation(sim_type: str, enable_plotting: bool):
    optimization_filepath = os.path.join('data', 'optimization_results.xlsx')
    initial_cond_filepath = os.path.join('data', 'data.xlsx')
    try:
        optimal_inputs, predicted_df, init_cond_df = load_aao_inputs(optimization_filepath, initial_cond_filepath)
        simulated_results, raw_influent = run_full_aao_simulation(
            optimal_inputs, sim_type, init_cond_df, predicted_df, enable_plotting
        )
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
    
    # Ask user if they want to generate plots
    while True:
        plot_choice = input("Do you want to generate plots for the simulation? (y/n): ").lower().strip()
        if plot_choice in ['y', 'n']:
            enable_plotting = plot_choice == 'y'
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    while True:
        choice = input("Which model would you like to validate?\n1. cstr (Single CSTR unit)\n2. cstr_and_clarifier (CSTR + Clarifier Plant)\n3. aao (Full AAO plant)\nEnter your choice: ").lower().strip()
        if choice in ['1', 'cstr']: 
            run_cstr_validation(sim_type='cstr', enable_plotting=enable_plotting)
            break
        elif choice in ['2', 'cstr_and_clarifier']: 
            run_cstr_and_clarifier_validation(sim_type='cstr_and_clarifier', enable_plotting=enable_plotting)
            break
        elif choice in ['3', 'aao']: 
            run_aao_validation(sim_type='aao', enable_plotting=enable_plotting)
            break
        else: 
            print("\nInvalid choice. Please enter 'cstr', 'cstr_and_clarifier', or 'aao' (or 1, 2, 3).")

if __name__ == '__main__':
    main()