import os
import sys
import pyomo.environ as pyo
import pandas as pd
import joblib
import itertools
import traceback

# --- PyTorch Class Definition (Required for loading the joblib file) ---
# This is a placeholder required by joblib to unpickle the model file.
# The actual model logic is rebuilt in Pyomo using the stored coefficients.
try:
    import torch
    import torch.nn as nn

    class CoupledCLEFOModel(nn.Module):
        def __init__(self, n_dep, n_indep, n_inter):
            super().__init__()
            self.n_dep, self.Upsilon, self.B, self.Theta, self.Gamma, self.Lambda = [None] * 6
        def forward(self, X, Z):
            return None
except ImportError:
    # If torch is not installed, provide a dummy class.
    print("Warning: PyTorch not found. A dummy class will be used for model unpickling.")
    class CoupledCLEFOModel: pass


class WWTPPlantOptimizer:
    """
    An optimizer for a full Wastewater Treatment Plant (WWTP) flowsheet.
    
    This class encapsulates the entire process:
    1. Loading plant configuration from an Excel file.
    2. Building a Pyomo Non-Linear Programming (NLP) model representing the entire plant.
    3. Handling interconnections and recycle streams.
    4. Solving the model to find optimal operating conditions.
    5. Reporting and saving the results.
    """

    def __init__(self, config_path: str):
        """Initializes the optimizer by loading all configuration data."""
        print("--- Initializing WWTP Plant Optimizer ---")
        self.config_path = config_path
        self._load_configuration()
        self._process_data()
        self.model = None
        self.results = None
        # Define a bound for the negative penalty. At this value, satisfaction is 0.
        self.NEGATIVE_PENALTY_BOUND = -10

    def _unify_comp_name(self, name: str) -> str:
        """Normalizes component names from various sources to a base name."""
        if name.startswith('inf_'):
            return name[4:]
        if 'Target' in name:
            base = name.split('(')[0].strip()
            # Handle specific stream prefixes
            for prefix in ['Target_Effluent_', 'Target_Wastage_']:
                if base.startswith(prefix):
                    return base.replace(prefix, '')
        return name # Return as-is if no pattern matches

    def _load_configuration(self):
        """Loads all data from the Excel configuration file."""
        print("1. Loading configuration from Excel file...")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at '{self.config_path}'")
        
        self.xls = pd.read_excel(self.config_path, sheet_name=None)
        
        # Define the fixed process flow
        self.units = ['A1', 'A2', 'O1', 'O2', 'O3', 'C1']
        self.cstr_units = ['A1', 'A2', 'O1', 'O2', 'O3']
        self.clarifier_units = ['C1']
        
        print(f"   - Plant configuration: {' -> '.join(self.units)}")

    def _process_data(self):
        """Processes the loaded dataframes to prepare for model building."""
        print("2. Processing and structuring configuration data...")
        
        # --- Surrogate Models ---
        df_config = self.xls['config']
        self.surrogate_models = {}
        for unit_type in ['cstr', 'clarifier']:
            path_row = df_config[df_config['Process Unit'] == unit_type]
            if path_row.empty: raise ValueError(f"No model path for '{unit_type}' in 'config' sheet.")
            model_path = path_row['Value'].iloc[0]
            if not os.path.exists(model_path): raise FileNotFoundError(f"Surrogate model not found: {model_path}")
            
            model_bundle = joblib.load(model_path)
            self.surrogate_models[unit_type] = {
                'x_scaler': model_bundle['x_scaler'], 'y_scaler': model_bundle['y_scaler'],
                'coeffs': {k: v for k, v in model_bundle.items() if k not in ['model', 'x_scaler', 'y_scaler']}
            }
            print(f"   - Loaded and processed '{unit_type}' surrogate model from '{model_path}'")

        # --- Decision Variables ---
        df_dvar = self.xls['decision_var']
        self.plant_wide_dvars = df_dvar[df_dvar['Process Unit'] == 'plant-wide']['I_variables'].tolist()
        self.cstr_dvars = df_dvar[df_dvar['Process Unit'] == 'cstr']['I_variables'].tolist()
        self.clarifier_dvars = df_dvar[df_dvar['Process Unit'] == 'clarifier']['I_variables'].tolist()
        print(f"   - Plant-wide Decision Variables: {self.plant_wide_dvars}")
        print(f"   - CSTR-specific Decision Variables: {self.cstr_dvars}")
        print(f"   - Clarifier-specific Decision Variables: {self.clarifier_dvars}")

        # --- Component Names ---
        df_influent = self.xls['raw_influent_compound_conc']
        self.influent_params = pd.Series(df_influent.Value.values, index=df_influent.Variable).to_dict()
        
        c1_outputs = self.surrogate_models['clarifier']['y_scaler'].feature_names_in_
        self.all_components = sorted(list(set([self._unify_comp_name(c) for c in c1_outputs])))
        print(f"   - Unified {len(self.all_components)} components for material balance.")

        # --- Goals and Bounds ---
        df_goals = self.xls['fuzzy_goal']
        df_bounds = self.xls['decision_var_bound']
        self.bounds = {row.Variable: (row.LowerBound, row.UpperBound) for _, row in df_bounds.iterrows()}
        self.effluent_goals = {row.Goal: {'target': row.Target, 'max': row.Max} for _, row in df_goals.iterrows() if pd.notna(row.Target)}
        print(f"   - Identified {len(self.bounds)} decision variable bounds.")
        print(f"   - Identified {len(self.effluent_goals)} fuzzy effluent goals.")

    def build_pyomo_model(self):
        """Constructs the full Pyomo model for the WWTP."""
        print("3. Building Pyomo NLP model for the entire plant...")
        self.model = m = pyo.ConcreteModel("WWTP_Plant_Optimization")

        # --- SETS ---
        m.UNITS = pyo.Set(initialize=self.units)
        m.CSTR_UNITS = pyo.Set(initialize=self.cstr_units)
        m.CLARIFIER_UNITS = pyo.Set(initialize=self.clarifier_units)
        m.COMPONENTS = pyo.Set(initialize=self.all_components)
        m.GOALS = pyo.Set(initialize=self.effluent_goals.keys())

        # --- DECISION VARIABLES ---
        for dv in self.plant_wide_dvars:
            if dv in self.bounds:
                setattr(m, dv, pyo.Var(bounds=self.bounds[dv], initialize=sum(self.bounds[dv])/2, within=pyo.NonNegativeReals))
            else:
                setattr(m, dv, pyo.Var(initialize=1000, within=pyo.NonNegativeReals))

        m.cstr_dvars = pyo.Var(m.CSTR_UNITS, pyo.Set(initialize=self.cstr_dvars), 
                               bounds=lambda m, u, dv: self.bounds[dv], 
                               initialize=lambda m, u, dv: sum(self.bounds[dv])/2, within=pyo.NonNegativeReals)
        m.clarifier_dvars = pyo.Var(m.CLARIFIER_UNITS, pyo.Set(initialize=self.clarifier_dvars),
                                    bounds=lambda m, u, dv: self.bounds[dv],
                                    initialize=lambda m, u, dv: sum(self.bounds[dv])/2, within=pyo.NonNegativeReals)

        # --- DYNAMIC BOUNDS CONSTRAINTS ---
        m.Q_int_upper_bound = pyo.Constraint(expr=m.Q_int <= m.Q_raw_inf)
        m.Q_was_upper_bound = pyo.Constraint(expr=m.Q_was <= m.Q_raw_inf)
        
        # --- STATE VARIABLES (Stream Compositions) ---
        m.stream_conc = pyo.Var(m.UNITS, m.COMPONENTS, within=pyo.Reals, initialize=10)
        m.C1_wastage_conc = pyo.Var(m.COMPONENTS, within=pyo.Reals, initialize=10)
        m.C1_effluent_conc = pyo.Var(m.COMPONENTS, within=pyo.Reals, initialize=10)

        # --- FUZZY LOGIC VARIABLES ---
        m.lambda_o = pyo.Var(initialize=0.5, within=pyo.Reals)
        m.lambda_g = pyo.Var(m.GOALS, initialize=0.5, within=pyo.Reals)
        m.lambda_neg = pyo.Var(initialize=0.5, within=pyo.Reals)

        # --- OBJECTIVE FUNCTION ---
        m.objective = pyo.Objective(expr=m.lambda_o, sense=pyo.maximize)
        m.satisfaction_rule = pyo.Constraint(m.GOALS, rule=lambda m, g: m.lambda_o <= m.lambda_g[g])
        m.neg_satisfaction_rule = pyo.Constraint(expr=m.lambda_o <= m.lambda_neg)

        # --- PROCESS UNIT BLOCKS ---
        self._add_unit_blocks()

        # --- GOAL AND PENALTY CONSTRAINTS ---
        self._add_goal_constraints()
        self._add_negative_penalty_constraints()

        print("...Pyomo model build complete.")

    def _add_unit_blocks(self):
        """Creates a Pyomo Block for each process unit in the flowsheet."""
        for unit in self.cstr_units: self._build_unit_submodel(unit, 'cstr')
        for unit in self.clarifier_units: self._build_unit_submodel(unit, 'clarifier')
    
    def _build_unit_submodel(self, unit_name, unit_type):
        """Helper to build the surrogate model for a single unit."""
        m = self.model
        b = pyo.Block()
        setattr(m, unit_name, b)

        model_data = self.surrogate_models[unit_type]
        x_scaler, y_scaler, coeffs = model_data['x_scaler'], model_data['y_scaler'], model_data['coeffs']

        b.M_names = list(x_scaler.feature_names_in_)
        b.K_names = list(y_scaler.feature_names_in_)
        b.M, b.K = pyo.Set(initialize=b.M_names), pyo.Set(initialize=b.K_names)
        b.L = pyo.Set(initialize=itertools.combinations(b.M_names, 2), dimen=2)
        m_map, k_map = {n: i for i, n in enumerate(b.M_names)}, {n: i for i, n in enumerate(b.K_names)}
        l_map = {n: i for i, n in enumerate(list(itertools.combinations(b.M_names, 2)))}
        
        b.X, b.X_s, b.Y_s = pyo.Var(b.M), pyo.Var(b.M), pyo.Var(b.K)

        @b.Constraint(b.M)
        def input_assembly_rule(b, M_name):
            if M_name in self.plant_wide_dvars: return b.X[M_name] == getattr(m, M_name)
            if unit_type == 'cstr' and M_name in self.cstr_dvars: return b.X[M_name] == m.cstr_dvars[unit_name, M_name]
            if unit_type == 'clarifier' and M_name in self.clarifier_dvars: return b.X[M_name] == m.clarifier_dvars[unit_name, M_name]
            
            unified_name = self._unify_comp_name(M_name)
            
            if unit_name == 'A1':
                C_raw = self.influent_params.get(f"inf_{unified_name}", 0)
                C_int = m.stream_conc['O3', unified_name]
                C_ext = m.C1_wastage_conc[unified_name]
                inlet_mass = (C_raw * m.Q_raw_inf) + (C_int * m.Q_int) + (C_ext * m.Q_ext)
                total_inflow = m.Q_raw_inf + m.Q_int + m.Q_ext
                return b.X[M_name] * total_inflow == inlet_mass
            else:
                prev_unit = self.units[self.units.index(unit_name) - 1]
                return b.X[M_name] == m.stream_conc[prev_unit, unified_name]

        @b.Constraint(b.M)
        def x_scaling_rule(b, M_name):
            idx = m_map[M_name]
            mu, sigma = x_scaler.mean_[idx], x_scaler.scale_[idx]
            return b.X_s[M_name] == (b.X[M_name] - mu) / sigma
        
        @b.Constraint(b.K)
        def clefo_model_rule(b, K_name):
            k_idx = k_map[K_name]
            term1 = b.Y_s[K_name]
            term2 = sum(coeffs['Lambda'][k_idx, m_map[m_n]] * b.Y_s[K_name] * b.X_s[m_n] for m_n in b.M)
            term3 = sum(coeffs['Gamma'][k_idx, k_map[kp]] * b.Y_s[kp] for kp in b.K)
            rhs_const = coeffs['Upsilon'][k_idx, 0]
            rhs_B = sum(coeffs['B'][k_idx, m_map[m_n]] * b.X_s[m_n] for m_n in b.M)
            rhs_Theta = sum(coeffs['Theta'][k_idx, l_map[l_p]] * b.X_s[l_p[0]] * b.X_s[l_p[1]] for l_p in b.L)
            return term1 - term2 - term3 == rhs_const + rhs_B + rhs_Theta

        @b.Constraint(b.K)
        def y_unscaling_rule(b, K_name):
            k_idx = k_map[K_name]
            mu, sigma = y_scaler.mean_[k_idx], y_scaler.scale_[k_idx]
            unscaled_Y = b.Y_s[K_name] * sigma + mu
            unified_name = self._unify_comp_name(K_name)
            
            if unit_type == 'cstr':
                return m.stream_conc[unit_name, unified_name] == unscaled_Y
            else:
                if 'Effluent' in K_name:
                    return m.C1_effluent_conc[unified_name] == unscaled_Y
                elif 'Wastage' in K_name:
                    return m.C1_wastage_conc[unified_name] == unscaled_Y
            return pyo.Constraint.Skip

    def _add_goal_constraints(self):
        """Adds the fuzzy goal constraints for effluent quality."""
        m = self.model
        m.FuzzyRules = pyo.ConstraintList()
        epsilon = 1e-9
        for goal_name, limits in self.effluent_goals.items():
            x_min, x_max = limits['target'], limits['max']
            denominator = x_max - x_min
            comp_name = self._unify_comp_name(goal_name)
            x = m.C1_effluent_conc[comp_name]
            m.FuzzyRules.add(m.lambda_g[goal_name] == 1 - (x - x_min) / (denominator + epsilon))

    def _add_negative_penalty_constraints(self):
        """Adds soft constraints to penalize negative state variable values."""
        m = self.model
        m.NegativePenaltyRules = pyo.ConstraintList()
        
        def penalty_rule(variable):
            return 1 - (variable / self.NEGATIVE_PENALTY_BOUND)

        for u in m.CSTR_UNITS:
            for c in m.COMPONENTS:
                m.NegativePenaltyRules.add(m.lambda_neg <= penalty_rule(m.stream_conc[u, c]))

        for c in m.COMPONENTS:
            m.NegativePenaltyRules.add(m.lambda_neg <= penalty_rule(m.C1_effluent_conc[c]))
            m.NegativePenaltyRules.add(m.lambda_neg <= penalty_rule(m.C1_wastage_conc[c]))

    def solve(self, solver='ipopt', tee=True, max_iterations=100000):
        """Solves the optimization problem."""
        if self.model is None:
            raise RuntimeError("Model has not been built yet. Call build_pyomo_model() first.")
        
        print(f"\n4. Solving the optimization problem with '{solver}' (max_iterations={max_iterations})...")
        solver_instance = pyo.SolverFactory(solver)
        if not solver_instance.available(exception_flag=False):
            print(f"ERROR: Solver '{solver}' not found. Please install it.", file=sys.stderr)
            return None
        
        solver_options = {'max_iter': max_iterations}
        self.results = solver_instance.solve(self.model, tee=tee, options=solver_options)
        return self.results
    
    def report_results(self):
        """
        Saves the optimization results to an Excel file in the format required
        for the validation script.
        """
        if self.results is None or self.model is None:
            print("No results to report.")
            return

        m = self.model
        term_cond = self.results.solver.termination_condition

        if term_cond not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal]:
            print("\n" + "="*80)
            print("OPTIMIZATION FAILED".center(80))
            print("="*80)
            print(f"Solver Status: {self.results.solver.status}, Termination: {term_cond}")
            if term_cond == pyo.TerminationCondition.maxIterations:
                print("Solver stopped at maximum iterations. Consider increasing 'max_iterations' or checking model scaling.")
            else:
                print("Problem may be infeasible or solver encountered other issues.")
            print("="*80)
            return

        print("\n---> Optimal Solution Found <---")
        print(f"   - Overall Satisfaction Level (lambda_o): {pyo.value(m.lambda_o):.4f}")
        print(f"   - Saving results to data/optimization_results.xlsx")

        # --- 1. Assemble 'optimal_decision_variables' worksheet ---
        dvar_data = []
        for dv in self.plant_wide_dvars:
            if dv == 'Q_raw_inf':
                dvar_data.append({
                    'Process Unit': 'plant-wide',
                    'Variable': 'Q_raw_inf',
                    'Optimal Value': pyo.value(m.Q_raw_inf)
                })
            elif dv != 'Q_int':
                dvar_data.append({
                    'Process Unit': 'plant-wide',
                    'Variable': dv,
                    'Optimal Value': pyo.value(getattr(m, dv))
                })

        for u, dv in m.cstr_dvars:
            dvar_data.append({'Process Unit': u, 'Variable': dv, 'Optimal Value': pyo.value(m.cstr_dvars[u, dv])})
        for u, dv in m.clarifier_dvars:
            dvar_data.append({'Process Unit': u, 'Variable': dv, 'Optimal Value': pyo.value(m.clarifier_dvars[u, dv])})

        q_int_val = pyo.value(m.Q_int)
        q_raw_inf_val = pyo.value(m.Q_raw_inf)
        q_ext_val = pyo.value(m.Q_ext)
        denominator = q_raw_inf_val + q_int_val + q_ext_val
        o3_split_val = q_int_val / denominator if denominator != 0 else 0
        dvar_data.append({
            'Process Unit': 'O3',
            'Variable': 'O3_split_internal',
            'Optimal Value': o3_split_val
        })
        df_dvars = pd.DataFrame(dvar_data)

        # --- 2. Assemble 'default_influent_quality' worksheet ---
        influent_data = []
        for key, value in self.influent_params.items():
            var_name = key.replace('inf_', '')
            influent_data.append({'Variable': var_name, 'Value (mg/L)': value})
        df_influent = pd.DataFrame(influent_data)

        # --- 3. Assemble 'optimal_predicted_effluent' worksheet ---
        predicted_data = []
        for u in self.cstr_units:
            for c in self.all_components:
                predicted_data.append({
                    'Component': f"{c}_{u}",
                    'Predicted Value (mg/L)': pyo.value(m.stream_conc[u, c])
                })
        for c in self.all_components:
            predicted_data.append({
                'Component': f"{c}_Effluent",
                'Predicted Value (mg/L)': pyo.value(m.C1_effluent_conc[c])
            })
        for c in self.all_components:
            predicted_data.append({
                'Component': f"{c}_Wastage",
                'Predicted Value (mg/L)': pyo.value(m.C1_wastage_conc[c])
            })
        df_predicted = pd.DataFrame(predicted_data)

        # --- 4. Save all data to Excel file ---
        output_path = os.path.join('data', 'optimization_results.xlsx')
        os.makedirs('data', exist_ok=True)
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_dvars.to_excel(writer, sheet_name='optimal_decision_variables', index=False)
                df_influent.to_excel(writer, sheet_name='default_influent_quality', index=False)
                df_predicted.to_excel(writer, sheet_name='optimal_predicted_effluent', index=False)
            print(f"   - Successfully saved results to '{output_path}'.")
            print("="*80)
        except Exception as e:
            print(f"ERROR: Failed to save results to Excel. Reason: {e}", file=sys.stderr)
            traceback.print_exc()

if __name__ == "__main__":
    try:
        config_file = os.path.join('data', 'optimization_config.xlsx')
        optimizer = WWTPPlantOptimizer(config_path=config_file)
        optimizer.build_pyomo_model()
        optimizer.solve(solver='ipopt', tee=True, max_iterations=100000)
        optimizer.report_results()
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: A required file was not found. Details: {e}", file=sys.stderr)
        print("Please ensure the 'data/optimization_config.xlsx' file exists and the script is run from the correct directory.", file=sys.stderr)
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}", file=sys.stderr)
        traceback.print_exc()