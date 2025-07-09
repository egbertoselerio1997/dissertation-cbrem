import os
import qsdsan as qs
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import warnings
from functools import partial
import traceback # Import traceback for detailed error diagnostics

# Import necessary classes for the new model
from qsdsan import processes as pc
from qsdsan.aeration import Aeration # For passive aeration

# Suppress the specific pkg_resources UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")

try:
    from scipy.stats import norm
except ImportError:
    print("SciPy not found. Please install it using: pip install scipy")
    exit()

try:
    import optuna
except ImportError:
    print("Optuna not found. Please install it using: pip install optuna")
    exit()

try:
    from rich.live import Live
    from rich.panel import Panel
except ImportError:
    print("Rich not found. Please install it using: pip install rich")
    exit()


class OptunaProgressCallback:
    """
    Optuna callback to provide live visual feedback and handle early stopping based on patience.
    """
    def __init__(self, patience, max_calls):
        self.patience = patience
        self.max_calls = max_calls
        self.best_value = -np.inf
        self.n_no_improvement = 0
        self.live = None

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        current_value = study.best_value if study.best_value is not None else -np.inf
        
        if current_value > self.best_value:
            self.best_value = current_value
            self.n_no_improvement = 0
        else:
            self.n_no_improvement += 1

        header_text = f"Trial: {trial.number + 1}/{self.max_calls} | "
        header_text += f"Best Avg Std: {self.best_value:.4f} | "
        header_text += f"Patience: {self.n_no_improvement}/{self.patience}"
        
        status_panel = Panel(header_text, title="[bold green]Bayesian Optimization Status[/bold green]", border_style="green")

        if self.live:
            self.live.update(status_panel)

        if self.n_no_improvement >= self.patience:
            study.stop()

def run_single_simulation(i, input_df, vary_inputs, probability_val=None):
    """
    Runs a single simulation instance of a trickling filter modeled as CSTRs in series.
    This function is designed to be parallelized.
    """
    try:
        # Each parallel worker needs its own QSDsan environment setup
        cmps = pc.create_asm2d_cmps()
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

        # Prepare influent WasteStream
        influent_concentrations = {
            key.replace('inf_', ''): value
            for key, value in current_inputs.items() if key.startswith('inf_')
        }
        ws = qs.WasteStream(f'influent_ws_{i}')
        ws.set_flow_by_concentration(
            flow_tot=current_inputs['flow_rate'], 
            concentrations=influent_concentrations, 
            units=('m3/d', 'mg/L')
        )

        asm2d_model = pc.ASM2d()

        # Get trickling filter parameters from the input dataframe
        N_stages = int(current_inputs['N_stages'])
        biomass_retention_fraction = current_inputs['biomass_retention_fraction']
        KLa_value = current_inputs['KLa']
        
        # Calculate total volume and volume per stage
        V_total = ws.F_vol * (current_inputs['HRT'] / 24)
        V_stage = V_total / N_stages

        # Define biomass components for retention
        # Get all component IDs in the ASM2d model for accurate splitting
        all_component_ids = cmps.IDs 
        particulate_biomass_IDs = ('X_AUT', 'X_H', 'X_PAO', 'X_PHA', 'X_PP', 'X_I', 'X_S')
        
        # Create a split dictionary: 
        #   biomass_retention_fraction for biomass components (recycled)
        #   0.0 for soluble components (go to next stage, not recycled in splitter)
        #   Note: Splitter 'split' attribute defines fraction going to the *first* outlet.
        #   So, 1-fraction to pass through.
        split_dct = {
            cmp_id: biomass_retention_fraction if cmp_id in particulate_biomass_IDs else 0.0
            for cmp_id in all_component_ids
        }

        # Initialize lists to hold system units and recycle streams
        unit_path = []
        recycle_streams = [] # Collects streams that will be explicitly defined as recycles in the System
        
        # This stream represents the liquid flowing forward from one stage to the next.
        # It starts as the main influent to the first stage.
        current_forward_flow = ws.copy()

        # Create the series of CSTRs, Mixers, and Splitters in a loop
        for j in range(N_stages):
            # Define streams for this stage
            # Mixer receives fresh influent/forward flow and recycled biomass from its own stage
            mixer_in_main = qs.WasteStream(f'M_in_main_{j}_{i}')
            mixer_in_recycle = qs.WasteStream(f'M_in_recycle_{j}_{i}')

            # Mixer combines inputs for the CSTR
            mixer = qs.sanunits.Mixer(ID=f'M_{j}_{i}', ins=(mixer_in_main, mixer_in_recycle))
            
            # Create the passive aeration model for the CSTR stage
            aeration_sys = Aeration(
                ID=f'aeration_{j}_{i}',
                DO_ID='S_O2',
                KLa=KLa_value,
                V=V_stage # Volume for KLa calculation
            )
            
            # CSTR for the current stage
            reactor_stage = qs.sanunits.CSTR(
                ID=f'R_{j}_{i}',
                ins=mixer.outs[0], # Mixer output goes to CSTR input
                V_max=V_stage,
                aeration=aeration_sys, # Use the aeration process object
                suspended_growth_model=asm2d_model,
                DO_ID='S_O2'
            )
            
            # Splitter to simulate biomass retention
            # Outlet 0 is recycle (biomass-rich), Outlet 1 is forward flow (liquid-rich)
            splitter = qs.sanunits.Splitter(
                ID=f'S_{j}_{i}',
                ins=reactor_stage.outs[0], # CSTR output goes to Splitter input
                split=split_dct # Define split fractions for each component
            )
            
            # === Connect the units for the current stage ===
            
            # Set the main input for the current stage's mixer
            mixer_in_main.copy_like(current_forward_flow) # Connect the previous stage's forward flow

            # The recycle stream from this stage's splitter loops back to this stage's mixer
            mixer_in_recycle.copy_like(splitter.outs[0])
            recycle_streams.append(mixer_in_recycle) # Add this stream to the system's recycle list

            # The forward flow for the *next* stage is the second outlet of this splitter
            current_forward_flow = splitter.outs[1]
            
            # Add all units created in this stage to the system path
            unit_path.extend([mixer_in_main, mixer_in_recycle, mixer, reactor_stage, splitter])

        # The final effluent is the 'current_forward_flow' after the last stage
        final_effluent = current_forward_flow 
        
        # Create the full system with all stages and explicit recycle loops
        # Note: qs.System automatically identifies the primary influent (ws) and primary effluent (final_effluent)
        sys = qs.System(f'sys_{i}', path=unit_path, recycle=recycle_streams, 
                        ins=[ws], outs=[final_effluent]) # Explicitly define system ins/outs for clarity
        
        sys.simulate(t_span=(0, 50), method='BDF', state_reset_hook='reset_cache')
        
        # Get results from the entire system.
        # The .results() method of System aggregates results from all units.
        outputs = sys.results(with_units=False)

        # The final effluent stream's concentrations
        effluent_stream_final = sys.outs[0] # This should be the 'final_effluent' defined above
        component_ids = sys.components.IDs # Get component IDs from the system's compiled components
        
        effluent_concs = effluent_stream_final.conc
        
        effluent_data = {}
        for idx, comp_id in enumerate(component_ids):
            effluent_data[f'Effluent_{comp_id} (mg/L)'] = effluent_concs[idx]

        run_data = {
            'simulation_number': i + 1,
            **current_inputs,
            **outputs,
            **effluent_data
        }
        
        return run_data
    except Exception as e:
        print(f"\n--- Simulation {i} FAILED ---")
        print(f"Error: {e}")
        print("Traceback:")
        traceback.print_exc() # Print full traceback
        print("----------------------------\n")
        return None

def objective_function(trial, base_input_df, optim_vars, num_simulations, probability_val):
    """
    Objective function for Optuna. It runs a set of stochastic
    simulations and returns the average standard deviation of the outputs.
    """
    input_df = base_input_df.copy()
    input_df['default'] = input_df['baseline']
    
    # Suggest new parameters for this trial
    for var in optim_vars:
        baseline_val = base_input_df.loc[var, 'baseline']
        std_dev = base_input_df.loc[var, 'std. dev.']
        low = max(0, baseline_val - 3 * std_dev)
        high = baseline_val + 3 * std_dev
        input_df.loc[var, 'default'] = trial.suggest_float(var, low, high)

    all_results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(i, input_df, vary_inputs=True, probability_val=probability_val)
        for i in range(num_simulations)
    )

    successful_results = [r for r in all_results if r is not None]

    if not successful_results:
        raise optuna.TrialPruned("All simulations failed for this parameter set.")

    results_df = pd.DataFrame(successful_results)
    output_cols = [col for col in results_df.columns if col not in input_df.index.to_list() and col != 'simulation_number']
    output_results_df = results_df[output_cols]

    if output_results_df.empty or len(output_results_df) < 2:
        return 0.0

    stats_df = output_results_df.describe().T
    
    if 'std' not in stats_df.columns or stats_df['std'].isnull().all():
        return 0.0

    avg_std = stats_df['std'].mean()
    
    return avg_std

def analyze_and_save_plots(study):
    """Generates and saves standard Optuna visualization plots."""

    if not study.trials:
        print("No trials to analyze.")
        return

    plots_dir = os.path.join('data', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("\nGenerating optimization analysis plots...")

    # Plot 1: Optimization History
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(plots_dir, "optimization_history.html"))

    # Plot 2: Parameter Importances
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 1:
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(os.path.join(plots_dir, "param_importances.html"))
        except Exception as e:
            # This can fail for various reasons, e.g., if all trials have the same value.
            print(f"Could not generate parameter importance plot: {e}")
    else:
        print("Skipping parameter importance plot (requires more than one completed trial).")
    
    print(f"Analysis plots saved as HTML files in the '{plots_dir}' directory.")

def run_bayesian_optimization(input_df, num_simulations_per_eval, max_calls, patience, probability_val):
    """
    Sets up and runs the Bayesian Optimization with Optuna.
    """
    optim_vars = input_df[input_df['randomizable'] == 1].index.tolist()
    
    objective = partial(objective_function,
                        base_input_df=input_df,
                        optim_vars=optim_vars,
                        num_simulations=num_simulations_per_eval,
                        probability_val=probability_val)

    print(f"\nStarting Bayesian Optimization for a maximum of {max_calls} trials...")
    print(f"Stopping if no improvement is seen for {patience} consecutive trials.")
    print(f"Optimizing default values for: {optim_vars}")
    print(f"Each trial will run {num_simulations_per_eval} stochastic simulations.")
    
    # Use TPE sampler, a form of Bayesian Optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    progress_callback = OptunaProgressCallback(patience=patience, max_calls=max_calls)
    
    interrupted = False
    initial_panel = Panel("Starting...", title="[bold green]Bayesian Optimization Status[/bold green]", border_style="green")
    with Live(initial_panel, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
        progress_callback.live = live
        try:
            # Optuna's warnings for repeated parameters can be verbose, so we hide them.
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(
                objective,
                n_trials=max_calls,
                callbacks=[progress_callback],
                n_jobs=1  # Run trials sequentially; parallelization is inside the objective
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\nOptimization stopped by user.")
        finally:
            optuna.logging.set_verbosity(optuna.logging.INFO) # Restore logging verbosity

    completed_trials = len(study.trials)
    if not interrupted and completed_trials < max_calls and study.best_trial:
         print("\nOptimization stopped early due to meeting the convergence criteria.")

    analyze_and_save_plots(study)

    try:
        best_params = study.best_params
        best_objective_value = study.best_value
    except ValueError: # This happens if no trials were completed successfully
        best_params = {}
        best_objective_value = None


    print("\nOptimization finished.")
    print(f"Total trials performed: {completed_trials}")
    if best_objective_value is not None:
        print(f"Best objective value (Maximized Avg. Std. Dev.): {best_objective_value:.4f}")
        print("Best default values found:")
        for var, val in best_params.items():
            print(f"  {var}: {val:.4f}")
    else:
        print("No successful trials completed.")
        return {}
        
    return best_params

def _prompt_for_missing_parameter(param_name, default_value, input_df_row):
    """Helper function to prompt user for missing parameters."""
    print(f"\nParameter '{param_name}' not found in data.xlsx.")
    while True:
        user_choice = input(f"Use default value {default_value}? (y/n/input_custom): ").lower()
        if user_choice == 'y':
            return default_value
        elif user_choice == 'n': # This option will be treated as input_custom
             user_choice = 'input_custom'
        
        if user_choice == 'input_custom':
            while True:
                try:
                    custom_value = float(input(f"Enter custom value for '{param_name}': "))
                    # Validate if value falls within a reasonable range (e.g., non-negative for N_stages, fraction for biomass_retention_fraction)
                    if param_name == 'N_stages' and custom_value < 1:
                        print("N_stages must be at least 1.")
                        continue
                    if param_name == 'biomass_retention_fraction' and not (0 <= custom_value <= 1):
                        print("Biomass retention fraction must be between 0 and 1.")
                        continue
                    if param_name == 'KLa' and custom_value < 0:
                        print("KLa must be non-negative.")
                        continue
                    return custom_value
                except ValueError:
                    print("Invalid input. Please enter a numerical value.")
        else:
            print("Invalid choice. Please enter 'y', 'n', or 'input_custom'.")


def run_simulation():
    """
    Main function to drive the simulation based on user inputs and Excel data.
    """
    data_filepath = os.path.join('data', 'data.xlsx')

    print(f"Loading input variables from {data_filepath}...")
    try:
        input_df = pd.read_excel(data_filepath, sheet_name='input_config', index_col=0)
    except FileNotFoundError:
        print(f"Error: '{data_filepath}' not found. Please create it or run the script again to generate a template.")
        # Optionally, create a template data.xlsx here.
        return
    except ValueError:
        print(f"Error: 'input_config' sheet not found in '{data_filepath}'.")
        return

    # --- Ensure all required parameters are present, prompt user if missing ---
    required_new_params = {
        'N_stages': {'default': 10, 'randomizable': 0, 'std. dev.': 0},
        'biomass_retention_fraction': {'default': 0.98, 'randomizable': 0, 'std. dev.': 0},
        'KLa': {'default': 200, 'randomizable': 0, 'std. dev.': 0},
    }

    for param_name, defaults in required_new_params.items():
        if param_name not in input_df.index:
            chosen_value = _prompt_for_missing_parameter(param_name, defaults['default'], None)
            
            # Create a new row for the missing parameter
            new_row = pd.Series({
                'default': chosen_value,
                'baseline': chosen_value, # Baseline starts as the chosen default
                'randomizable': defaults['randomizable'],
                'std. dev.': defaults['std. dev.']
            }, name=param_name)
            input_df = pd.concat([input_df, pd.DataFrame([new_row])])
            print(f"Added '{param_name}' with value {chosen_value} to the input configuration.")

    # Convert numeric columns explicitly, as newly added rows might not have correct types
    for col in ['default', 'baseline', 'randomizable', 'std. dev.']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    input_df['randomizable'] = input_df['randomizable'].astype(int)

    # --- End of parameter check and prompting ---

    while True:
        choice = input("\nRun Bayesian Optimization to find best default values? (y/n): ").lower()
        if choice in ('y', 'n'):
            run_optimization = (choice == 'y')
            break
        print("Invalid choice. Please enter 'y' or 'n'.")

    num_simulations = 1
    vary_inputs = False
    probability_val = None

    if run_optimization:
        while True:
            try:
                max_calls = int(input("Enter the maximum number of optimization iterations: "))
                if max_calls > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        while True:
            try:
                patience = int(input("Enter convergence patience (stop after N iterations with no improvement): "))
                if patience > 0 and patience < max_calls:
                    break
                print(f"Patience must be a positive number less than the max iterations ({max_calls}).")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        while True:
            try:
                n_sims_per_eval = int(input("Enter number of stochastic simulations per iteration: "))
                if n_sims_per_eval > 1:
                    break
                print("Please enter a number greater than 1 for meaningful statistics.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        while True:
            try:
                probability_val = float(input("Enter the probability for the confidence interval (e.g., 0.95): "))
                if 0 < probability_val < 1:
                    break
                print("Please enter a probability value between 0 and 1 (exclusive).")
            except ValueError:
                print("Invalid input. Please enter a floating-point number.")

        best_defaults = run_bayesian_optimization(input_df, n_sims_per_eval, max_calls, patience, probability_val)
        
        if best_defaults:
            print("\nUpdating default values in memory with optimized results...")
            for var, val in best_defaults.items():
                input_df.loc[var, 'default'] = val

            print(f"Saving optimal default values to '{data_filepath}'...")
            try:
                # Use openpyxl engine with if_sheet_exists='replace' to update existing sheet
                with pd.ExcelWriter(data_filepath, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                    input_df.to_excel(writer, sheet_name='input_config')
                print("Optimal defaults saved successfully.")
            except Exception as e:
                print(f"An error occurred while saving the optimal defaults: {e}")
        else:
            print("Optimization did not yield a result. Using original default values for the final run.")

        print("\nNow, configure the final simulation run with the optimized default values.")
        while True:
            try:
                num_simulations = int(input("Enter the number of final simulations to run: "))
                if num_simulations > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        vary_inputs = True

    else:
        while True:
            try:
                num_simulations = int(input("Enter the number of simulations to run: "))
                if num_simulations > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        if num_simulations > 1:
            while True:
                choice = input("Vary inputs randomly? (y/n): ").lower()
                if choice in ('y', 'n'):
                    vary_inputs = (choice == 'y')
                    break
                print("Invalid choice. Please enter 'y' or 'n'.")
            
            if vary_inputs:
                while True:
                    try:
                        probability_val = float(input("Enter the probability for the confidence interval (e.g., 0.95): "))
                        if 0 < probability_val < 1:
                            break
                        print("Please enter a probability value between 0 and 1 (exclusive).")
                    except ValueError:
                        print("Invalid input. Please enter a floating-point number.")

    print("\nStarting simulations...")
    if num_simulations > 1:
        all_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_single_simulation)(i, input_df, vary_inputs, probability_val)
            for i in range(num_simulations)
        )
    else:
        # For a single simulation, run in the main process to allow direct debugging
        cmps = pc.create_asm2d_cmps()
        qs.set_thermo(cmps)
        all_results = [run_single_simulation(0, input_df, vary_inputs, probability_val)]
        print("  Completed simulation 1/1")

    final_successful_results = [r for r in all_results if r is not None]
    if not final_successful_results:
        print("\nAll final simulations failed. No results to save.")
        return

    print(f"\n{len(final_successful_results)}/{num_simulations} simulations completed successfully. Saving results to {data_filepath}...")
    results_df = pd.DataFrame(final_successful_results)
    
    # Ensure all columns from input_df.index are present in results_df before slicing
    missing_cols = [col for col in input_df.index.to_list() if col not in results_df.columns]
    for m_col in missing_cols:
        results_df[m_col] = np.nan # Add missing columns with NaN values to avoid KeyError

    input_cols = ['simulation_number'] + input_df.index.to_list()
    input_results_df = results_df[input_cols]
    
    output_cols = ['simulation_number'] + [col for col in results_df.columns if col not in input_df.index.to_list() and col != 'simulation_number']
    output_results_df = results_df[output_cols]
    
    try:
        with pd.ExcelWriter(data_filepath, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            input_results_df.to_excel(writer, sheet_name='results_input', index=False)
            output_results_df.to_excel(writer, sheet_name='results_output', index=False)

            if len(final_successful_results) > 1:
                print("Calculating statistics for output results...")
                stats_df = output_results_df.drop(columns=['simulation_number']).describe().T
                stats_df.to_excel(writer, sheet_name='results_statistics')
                print(f"Results and statistics successfully saved to '{data_filepath}'.")
            else:
                print(f"Results successfully saved to '{data_filepath}'.")

    except FileNotFoundError:
        print(f"Error: '{data_filepath}' was not found during the save operation.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

if __name__ == '__main__':
    run_simulation()