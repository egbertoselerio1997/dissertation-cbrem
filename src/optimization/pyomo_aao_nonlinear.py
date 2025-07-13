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

    def _unify_comp_name(self, name: str) -> str:
        """Normalizes component names from various sources to a base name."""
        if name.startswith('inf_'):
            return name[4:]
        if 'Target' in name:
            base = name.split('(')[0].strip()
            if 'Effluent' in base:
                return base.replace('Target_Effluent_', '')
            if 'Wastage' in base:
                return base.replace('Target_Wastage_', '')
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
            if path_row.empty:
                raise ValueError(f"No model path for '{unit_type}' in 'config' sheet.")
            model_path = path_row['Value'].iloc[0]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Surrogate model not found: {model_path}")
            
            # The custom class must be in scope for joblib to unpickle successfully
            model_bundle = joblib.load(model_path)
            self.surrogate_models[unit_type] = {
                'x_scaler': model_bundle['x_scaler'],
                'y_scaler': model_bundle['y_scaler'],
                'coeffs': {k: v for k, v in model_bundle.items() if k not in ['model', 'x_scaler', 'y_scaler']}
            }
            print(f"   - Loaded and processed '{unit_type}' surrogate model from '{model_path}'")

        # --- Decision Variables ---
        df_dvar = self.xls['decision_var']
        self.shared_dvars = df_dvar[df_dvar['Process Unit'] == 'clarifier, cstr']['I_variables'].tolist()
        self.cstr_dvars = df_dvar[df_dvar['Process Unit'] == 'cstr']['I_variables'].tolist()
        self.clarifier_dvars = df_dvar[df_dvar['Process Unit'] == 'clarifier']['I_variables'].tolist()
        print(f"   - Shared Decision Variables: {self.shared_dvars}")
        print(f"   - CSTR-specific Decision Variables: {self.cstr_dvars}")
        print(f"   - Clarifier-specific Decision Variables: {self.clarifier_dvars}")

        # --- Component Names ---
        df_influent = self.xls['influent_compound_conc']
        self.influent_params = pd.Series(df_influent.Value.values, index=df_influent.Variable).to_dict()
        self.components = sorted([self._unify_comp_name(c) for c in self.influent_params.keys()])
        
        # Map surrogate model input/output names to unified component names
        self.cstr_input_map = {self._unify_comp_name(n): n for n in self.surrogate_models['cstr']['x_scaler'].feature_names_in_}
        self.cstr_output_map = {self._unify_comp_name(n): n for n in self.surrogate_models['cstr']['y_scaler'].feature_names_in_}
        
        c1_outputs = self.surrogate_models['clarifier']['y_scaler'].feature_names_in_
        self.c1_effluent_map = {self._unify_comp_name(n): n for n in c1_outputs if 'Effluent' in n}
        self.c1_wastage_map = {self._unify_comp_name(n): n for n in c1_outputs if 'Wastage' in n}
        # Create a set of all unique component names across the entire system
        self.all_components = sorted(list(set(self.components) | set(self.cstr_output_map.keys()) | set(self.c1_effluent_map.keys()) | set(self.c1_wastage_map.keys())))

        print(f"   - Unified {len(self.all_components)} components for material balance.")

        # --- Goals and Bounds ---
        df_goals = self.xls['fuzzy_goal']
        
        try:
            # The prompt implies bounds are in a separate sheet, which is good practice.
            df_bounds = self.xls['decision_var_bound']
            self.bounds = {row.Variable: (row.LowerBound, row.UpperBound) for _, row in df_bounds.iterrows()}
        except KeyError:
            raise KeyError("The 'decision_var_bound' worksheet is missing from the Excel file. It's required for setting variable bounds.")
        except AttributeError:
             raise AttributeError("The 'decision_var_bound' worksheet must contain 'Variable', 'LowerBound', and 'UpperBound' columns.")

        self.fuzzy_goals = {
            row.Goal: {'target': row.Target, 'max': row.Max} 
            for _, row in df_goals.iterrows() if pd.notna(row.Target)
        }
        self.cost_goals = {k for k in self.fuzzy_goals if k in ['CAPEX', 'AOC']}
        self.effluent_goals = {k for k in self.fuzzy_goals if k not in self.cost_goals}
        print(f"   - Identified {len(self.bounds)} decision variable bounds.")
        print(f"   - Identified {len(self.fuzzy_goals)} fuzzy goals ({len(self.cost_goals)} cost, {len(self.effluent_goals)} effluent).")


        # --- Cost Models ---
        self.cost_params = {
            'cstr': self.xls['cost_var_cstr'].set_index('Variable')['Value'].to_dict(),
            'clarifier': self.xls['cost_var_clarifier'].set_index('Variable')['Value'].to_dict(),
        }
        self.cost_calcs = {
            'cstr': self.xls['capex_calc_cstr'],
            'clarifier': self.xls['capex_calc_clarifier'],
        }

    def build_pyomo_model(self):
        """Constructs the full Pyomo model for the WWTP."""
        print("3. Building Pyomo NLP model for the entire plant...")
        self.model = m = pyo.ConcreteModel("WWTP_Plant_Optimization")

        # --- SETS ---
        m.UNITS = pyo.Set(initialize=self.units)
        m.CSTR_UNITS = pyo.Set(initialize=self.cstr_units)
        m.CLARIFIER_UNITS = pyo.Set(initialize=self.clarifier_units)
        m.COMPONENTS = pyo.Set(initialize=self.all_components)
        m.GOALS = pyo.Set(initialize=self.fuzzy_goals.keys())

        # --- DECISION VARIABLES ---
        # Shared variables
        for dv in self.shared_dvars:
            setattr(m, dv, pyo.Var(bounds=self.bounds[dv], initialize=sum(self.bounds[dv])/2))
        
        # Unit-specific variables
        m.cstr_dvars = pyo.Var(m.CSTR_UNITS, pyo.Set(initialize=self.cstr_dvars), 
                               bounds=lambda m, u, dv: self.bounds[dv], 
                               initialize=lambda m, u, dv: sum(self.bounds[dv])/2)
        m.clarifier_dvars = pyo.Var(m.CLARIFIER_UNITS, pyo.Set(initialize=self.clarifier_dvars),
                                    bounds=lambda m, u, dv: self.bounds[dv],
                                    initialize=lambda m, u, dv: sum(self.bounds[dv])/2)

        # --- STATE VARIABLES (Stream Compositions and Flows) ---
        m.stream_in_conc = pyo.Var(m.UNITS, m.COMPONENTS, within=pyo.NonNegativeReals, initialize=10)
        m.stream_out_conc = pyo.Var(m.UNITS, m.COMPONENTS, within=pyo.NonNegativeReals, initialize=10)
        m.c1_wastage_conc = pyo.Var(m.COMPONENTS, within=pyo.NonNegativeReals, initialize=10) # Specific to C1 wastage
        m.O3_split_internal = pyo.Var(within=pyo.NonNegativeReals, initialize=0.5) # Internal recycle flow from O3 to A1
        m.Q_ext = pyo.Var(within=pyo.NonNegativeReals, initialize=100) # External recycle flow from C1 to A1

        # --- FUZZY LOGIC VARIABLES ---
        m.lambda_o = pyo.Var(initialize=0.5)
        m.lambda_g = pyo.Var(m.GOALS, initialize=0.5)

        # --- OBJECTIVE FUNCTION ---
        m.objective = pyo.Objective(expr=m.lambda_o, sense=pyo.maximize)
        m.satisfaction_rule = pyo.Constraint(m.GOALS, rule=lambda m, g: m.lambda_o <= m.lambda_g[g])

        # --- PROCESS UNIT BLOCKS ---
        self._add_unit_blocks()

        # --- INTERCONNECTION & MATERIAL BALANCE CONSTRAINTS ---
        self._add_interconnection_constraints()

        # --- COST & GOAL CONSTRAINTS ---
        self._add_cost_and_goal_constraints()

        print("...Pyomo model build complete.")

    def _add_unit_blocks(self):
        """Creates a Pyomo Block for each process unit in the flowsheet."""
        m = self.model
        
        # Add blocks for CSTR units
        for unit in m.CSTR_UNITS:
            self._build_unit_submodel(unit, 'cstr')
        
        # Add block for the Clarifier unit
        for unit in m.CLARIFIER_UNITS:
            self._build_unit_submodel(unit, 'clarifier')
    
    def _build_unit_submodel(self, unit_name, unit_type):
        """Helper to build the surrogate and cost model for a single unit."""
        m = self.model
        b = pyo.Block()
        setattr(m, unit_name, b)

        model_data = self.surrogate_models[unit_type]
        x_scaler = model_data['x_scaler']
        y_scaler = model_data['y_scaler']
        coeffs = model_data['coeffs']

        # Map variable names to indices for efficient lookup
        b.M_names = [n for n in x_scaler.feature_names_in_]
        b.K_names = [n for n in y_scaler.feature_names_in_]
        b.M = pyo.Set(initialize=b.M_names)
        b.K = pyo.Set(initialize=b.K_names)
        b.L = pyo.Set(initialize=itertools.combinations(b.M_names, 2), dimen=2)
        m_map = {name: i for i, name in enumerate(b.M_names)}
        k_map = {name: i for i, name in enumerate(b.K_names)}
        l_map = {name: i for i, name in enumerate(list(itertools.combinations(b.M_names, 2)))}
        
        # --- Surrogate Model Variables ---
        b.X = pyo.Var(b.M) # Unscaled inputs
        b.X_s = pyo.Var(b.M) # Scaled inputs
        b.Y_s = pyo.Var(b.K) # Scaled outputs
        
        # --- Surrogate Model Constraints ---
        # 1. Assemble input vector X
        @b.Constraint(b.M)
        def input_assembly_rule(b, M_name):
            # M_name is the original name from the surrogate model
            if M_name in self.shared_dvars:
                return b.X[M_name] == getattr(m, M_name)
            if unit_type == 'cstr' and M_name in self.cstr_dvars:
                return b.X[M_name] == m.cstr_dvars[unit_name, M_name]
            if unit_type == 'clarifier' and M_name in self.clarifier_dvars:
                return b.X[M_name] == m.clarifier_dvars[unit_name, M_name]
            # It must be a stream component
            unified_name = self._unify_comp_name(M_name)
            return b.X[M_name] == m.stream_in_conc[unit_name, unified_name]

        # 2. Scale inputs
        @b.Constraint(b.M)
        def x_scaling_rule(b, M_name):
            idx = m_map[M_name]
            mu, sigma = x_scaler.mean_[idx], x_scaler.scale_[idx]
            return b.X_s[M_name] == (b.X[M_name] - mu) / sigma
        
        # 3. CLEFO model equation
        @b.Constraint(b.K)
        def clefo_model_rule(b, K_name):
            k_idx = k_map[K_name]
            # Y_s(k) - sum(Λ*Y_s*X_s) - sum(Γ*Y_s) = Y + B*X_s + Θ*Z_s
            term1 = b.Y_s[K_name]
            term2 = sum(coeffs['Lambda'][k_idx, m_map[m_n]] * b.Y_s[K_name] * b.X_s[m_n] for m_n in b.M)
            term3 = sum(coeffs['Gamma'][k_idx, k_map[kp]] * b.Y_s[kp] for kp in b.K)
            rhs_const = coeffs['Upsilon'][k_idx, 0]
            rhs_B = sum(coeffs['B'][k_idx, m_map[m_n]] * b.X_s[m_n] for m_n in b.M)
            rhs_Theta = sum(coeffs['Theta'][k_idx, l_map[l_p]] * b.X_s[l_p[0]] * b.X_s[l_p[1]] for l_p in b.L)
            return term1 - term2 - term3 == rhs_const + rhs_B + rhs_Theta

        # 4. Unscale outputs and link to main model variables
        @b.Constraint(b.K)
        def y_unscaling_rule(b, K_name):
            k_idx = k_map[K_name]
            mu, sigma = y_scaler.mean_[k_idx], y_scaler.scale_[k_idx]
            unscaled_Y = b.Y_s[K_name] * sigma + mu
            
            # Link the unscaled output to the correct main model variable
            unified_name = self._unify_comp_name(K_name)

            if unit_type == 'clarifier' and 'Wastage' in K_name:
                return m.c1_wastage_conc[unified_name] == unscaled_Y
            # All other outputs are main stream outputs (CSTR outputs or C1 Effluent)
            return m.stream_out_conc[unit_name, unified_name] == unscaled_Y

        # --- Cost Model ---
        cost_context = {dv: getattr(m, dv) for dv in self.shared_dvars}
        if unit_type == 'cstr':
            cost_context.update({dv: m.cstr_dvars[unit_name, dv] for dv in self.cstr_dvars})
        else: # clarifier
            cost_context.update({dv: m.clarifier_dvars[unit_name, dv] for dv in self.clarifier_dvars})
        
        # Add fixed cost parameters
        for p_name, p_val in self.cost_params[unit_type].items():
            if p_name not in cost_context:
                cost_context[p_name] = p_val
        
        # Add cost calculation variables and rules
        b.CostRules = pyo.ConstraintList()
        df_capex_calc = self.cost_calcs[unit_type]
        for _, row in df_capex_calc.iterrows():
            var_name = row['Output Variable']
            calc_str = row['Calculation']
            var = pyo.Var(within=pyo.Reals, initialize=0)
            setattr(b, var_name, var)
            cost_context[var_name] = var
            # Safely evaluate the expression string to create a Pyomo expression
            rhs_expr = eval(calc_str, {"__builtins__": None}, cost_context)
            b.CostRules.add(var == rhs_expr)

    def _add_interconnection_constraints(self):
        """Adds material balance constraints that link the process units."""
        m = self.model
        m.connections = pyo.ConstraintList()

        # --- Sequential Connections ---
        # Output of unit i is input to unit i+1
        for i in range(len(self.units) - 1):
            up_unit, down_unit = self.units[i], self.units[i+1]
            for comp in m.COMPONENTS:
                m.connections.add(m.stream_in_conc[down_unit, comp] == m.stream_out_conc[up_unit, comp])

        # --- A1 Inlet Mixer (Recycle streams) ---
        # Implements: C_in_A1 * Q_total = C_inf*Q_inf + C_O3*Q_int + C_was*Q_ext
        m.A1_inlet_mixer = pyo.ConstraintList()
        Q_inf = m.flow_rate
        Q_int_recycle = m.O3_split_internal # Decision variable
        Q_ext_recycle = m.Q_ext # Variable predicted by C1 model
        total_flow_A1 = Q_inf + Q_int_recycle + Q_ext_recycle

        for comp in m.COMPONENTS:
            # Get influent concentration, default to 0 if not present in influent list
            C_inf = self.influent_params.get(f"inf_{comp}", 0) 
            C_o3 = m.stream_out_conc['O3', comp]
            C_c1_was = m.c1_wastage_conc[comp]
            
            inlet_mass = C_inf * Q_inf + C_o3 * Q_int_recycle + C_c1_was * Q_ext_recycle
            m.A1_inlet_mixer.add(m.stream_in_conc['A1', comp] * total_flow_A1 == inlet_mass)

    def _add_cost_and_goal_constraints(self):
        """Adds total cost calculations and fuzzy goal constraints."""
        m = self.model
        
        # --- Total Cost Calculation ---
        m.Total_CAPEX = pyo.Var(within=pyo.Reals)
        m.Total_AOC = pyo.Var(within=pyo.Reals)
        
        m.Total_CAPEX_rule = pyo.Constraint(expr=m.Total_CAPEX == sum(getattr(m, u).CAPEX for u in m.UNITS))
        m.Total_AOC_rule = pyo.Constraint(expr=m.Total_AOC == sum(getattr(m, u).AOC for u in m.UNITS))

        # --- Fuzzy Goal Constraints ---
        # Implements: lambda_g = 1 - (x - target) / (max - target)
        m.FuzzyRules = pyo.ConstraintList()
        epsilon = 1e-9 # To prevent division by zero

        for goal_name, limits in self.fuzzy_goals.items():
            x_min, x_max = limits['target'], limits['max']
            denominator = x_max - x_min
            
            if goal_name == 'CAPEX':
                x = m.Total_CAPEX
            elif goal_name == 'AOC':
                x = m.Total_AOC
            else: # Effluent quality goal
                comp_name = self._unify_comp_name(goal_name)
                # This constraint applies to the final effluent from C1
                x = m.stream_out_conc['C1', comp_name]

            m.FuzzyRules.add(m.lambda_g[goal_name] == 1 - (x - x_min) / (denominator + epsilon))

    def solve(self, solver='ipopt', tee=True, max_iterations=10000):
        """Solves the optimization problem."""
        if self.model is None:
            raise RuntimeError("Model has not been built yet. Call build_pyomo_model() first.")
        
        print(f"\n4. Solving the optimization problem with '{solver}' (max_iterations={max_iterations})...")
        solver_instance = pyo.SolverFactory(solver)
        if not solver_instance.available(exception_flag=False):
            print(f"ERROR: Solver '{solver}' not found. Please install it.", file=sys.stderr)
            return None
        
        # Pass solver options to increase the maximum number of iterations.
        solver_options = {'max_iter': max_iterations}
        self.results = solver_instance.solve(self.model, tee=tee, options=solver_options)
        return self.results
    
    def report_results(self):
        """Displays and saves a comprehensive report of the optimization results."""
        if self.results is None or self.model is None:
            print("No results to report.")
            return

        m = self.model
        term_cond = self.results.solver.termination_condition

        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS REPORT".center(80))
        print("="*80)
        print(f"Solver Status: {self.results.solver.status}, Termination: {term_cond}")

        if term_cond not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal]:
            print("\n---> No optimal solution found. <---")
            if term_cond == pyo.TerminationCondition.maxIterations:
                print("The solver stopped because it reached the maximum number of iterations.")
                print("Consider increasing the 'max_iterations' parameter in the solve() method.")
            else:
                print("The problem may be infeasible or the solver may have encountered other issues.")
            print("="*80)
            return

        print("\n---> Optimal Solution Found <---\n")
        
        # --- Satisfaction Levels ---
        print(f"Overall Satisfaction Level (lambda_o): {pyo.value(m.lambda_o):.4f}\n")
        df_satisfaction_report = pd.DataFrame(
            [{'Goal': g, 'Satisfaction': pyo.value(m.lambda_g[g])} for g in m.GOALS]
        ).sort_values(by='Goal').set_index('Goal')
        print("--- Individual Goal Satisfaction (lambda_g) ---")
        print(df_satisfaction_report)

        # --- Decision Variables ---
        print("\n--- Optimal Decision Variables ---")
        dvar_data = []
        for dv in self.shared_dvars:
            dvar_data.append({'Variable': dv, 'Unit': 'Shared', 'Value': pyo.value(getattr(m, dv))})
        for u, dv in m.cstr_dvars:
            dvar_data.append({'Variable': dv, 'Unit': u, 'Value': pyo.value(m.cstr_dvars[u, dv])})
        for u, dv in m.clarifier_dvars:
            dvar_data.append({'Variable': dv, 'Unit': u, 'Value': pyo.value(m.clarifier_dvars[u, dv])})
        df_dvars_report = pd.DataFrame(dvar_data).sort_values(by=['Unit', 'Variable'])
        print(df_dvars_report.to_string(index=False))

        # --- Costs ---
        print("\n--- Plant-Wide Costs ---")
        capex_goal = self.fuzzy_goals.get('CAPEX', {})
        aoc_goal = self.fuzzy_goals.get('AOC', {})
        print(f"Total CAPEX: ${pyo.value(m.Total_CAPEX):>15,.2f} "
              f"(Target: ${capex_goal.get('target', 0):,.0f}, Max: ${capex_goal.get('max', 0):,.0f})")
        print(f"Total AOC:   ${pyo.value(m.Total_AOC):>15,.2f} / yr "
              f"(Target: ${aoc_goal.get('target', 0):,.0f}, Max: ${aoc_goal.get('max', 0):,.0f})")

        # --- Final Effluent Quality ---
        print("\n--- Final Effluent Quality (C1 Output) ---")
        eff_data_report = []
        for goal_name in self.effluent_goals:
            comp = self._unify_comp_name(goal_name)
            val = pyo.value(m.stream_out_conc['C1', comp])
            goal_limits = self.fuzzy_goals[goal_name]
            eff_data_report.append({
                'Component': goal_name,
                'Value (mg/L)': val,
                'Target': goal_limits['target'],
                'Max': goal_limits['max']
            })
        df_effluent_report = pd.DataFrame(eff_data_report).set_index('Component')
        print(df_effluent_report)
        
        print("="*80)
        
        # --- Save to Excel ---
        output_folder = 'data'
        output_filename = 'data.xlsx'
        output_path = os.path.join(output_folder, output_filename)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n5. Saving detailed results to '{output_path}'...")

        try:
            # 1. Prepare optimal_satisfaction
            satisfaction_data = [{'Metric': 'Overall Satisfaction Level (lambda_o)', 'Value': pyo.value(m.lambda_o)}]
            for g in sorted(m.GOALS):
                satisfaction_data.append({'Metric': f'Individual Goal Satisfaction ({g})', 'Value': pyo.value(m.lambda_g[g])})
            df_satisfaction = pd.DataFrame(satisfaction_data)

            # 2. Prepare optimal_decision_variables
            dvar_data_out = []
            for dv in self.shared_dvars:
                dvar_data_out.append({'Variable': dv, 'Process Unit': 'Shared', 'Optimal Value': pyo.value(getattr(m, dv))})
            for u, dv in m.cstr_dvars:
                dvar_data_out.append({'Variable': dv, 'Process Unit': u, 'Optimal Value': pyo.value(m.cstr_dvars[u, dv])})
            for u, dv in m.clarifier_dvars:
                dvar_data_out.append({'Variable': dv, 'Process Unit': u, 'Optimal Value': pyo.value(m.clarifier_dvars[u, dv])})
            df_decision_vars = pd.DataFrame(dvar_data_out).sort_values(by=['Process Unit', 'Variable'])
            
            # 3. Prepare optimal_capex_breakdown
            capex_data = []
            capex_keywords = ['CAPEX', 'purch', 'C_const', 'C_equip', 'C_install']
            for unit in self.units:
                unit_type = 'cstr' if unit in self.cstr_units else 'clarifier'
                unit_block = getattr(m, unit)
                df_cost_calc = self.cost_calcs[unit_type]
                for _, row in df_cost_calc.iterrows():
                    var_name = row['Output Variable']
                    if any(keyword in var_name for keyword in capex_keywords):
                        value = pyo.value(getattr(unit_block, var_name))
                        description = row['Description']
                        capex_data.append({'Component': description, 'Value': value, 'Unit': '$', 'Notes': unit})
            df_capex = pd.DataFrame(capex_data)

            # 4. Prepare optimal_aoc_breakdown
            aoc_data = []
            aoc_keywords = ['AOC', 'cost', 'power', 'cons']
            for unit in self.units:
                unit_type = 'cstr' if unit in self.cstr_units else 'clarifier'
                unit_block = getattr(m, unit)
                df_cost_calc = self.cost_calcs[unit_type]
                for _, row in df_cost_calc.iterrows():
                    var_name = row['Output Variable']
                    if any(keyword in var_name for keyword in aoc_keywords):
                        value = pyo.value(getattr(unit_block, var_name))
                        description = row['Description']
                        aoc_data.append({'Component': description, 'Value': value, 'Unit': '$/yr', 'Notes': unit})
            df_aoc = pd.DataFrame(aoc_data)

            # 5. Prepare optimal_predicted_effluent
            effluent_data = []
            # Report effluent from all sequential process units (A1, A2, O1, O2, O3, C1).
            # The main output stream from C1 is the final treated plant effluent.
            for unit in self.units:
                for comp in sorted(self.all_components):
                    value = pyo.value(m.stream_out_conc[unit, comp])
                    effluent_data.append({'Component': f'{comp}_{unit}', 'Predicted Value (mg/L)': value})
            # Report the wastage stream from C1 separately
            for comp in sorted(self.all_components):
                value = pyo.value(m.c1_wastage_conc[comp])
                effluent_data.append({'Component': f'{comp}_Wastage', 'Predicted Value (mg/L)': value})
            df_effluent = pd.DataFrame(effluent_data)

            # 6. Prepare default_influent_quality
            influent_data = [{'Variable': self._unify_comp_name(k), 'Value (mg/L)': v} for k, v in self.influent_params.items()]
            df_influent = pd.DataFrame(influent_data)

            # Use ExcelWriter to save to specific sheets, preserving others
            mode = 'a' if os.path.exists(output_path) else 'w'
            with pd.ExcelWriter(output_path, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
                df_satisfaction.to_excel(writer, sheet_name='optimal_satisfaction', index=False)
                df_decision_vars.to_excel(writer, sheet_name='optimal_decision_variables', index=False)
                df_capex.to_excel(writer, sheet_name='optimal_capex_breakdown', index=False)
                df_aoc.to_excel(writer, sheet_name='optimal_aoc_breakdown', index=False)
                df_effluent.to_excel(writer, sheet_name='optimal_predicted_effluent', index=False)
                df_influent.to_excel(writer, sheet_name='default_influent_quality', index=False)
            
            print("...Save successful.")

        except Exception as e:
            print(f"ERROR: Failed to save results to Excel. Reason: {e}", file=sys.stderr)
            traceback.print_exc()


if __name__ == "__main__":
    # Ensure the solver is in the system's PATH if needed.
    # On some systems, you might need to provide the full path to the solver executable.
    
    try:
        # Define the path to the configuration file (assuming it's in a 'data' subfolder)
        config_file = os.path.join('data', 'optimization_config.xlsx')

        # 1. Initialize the optimizer with the configuration file
        optimizer = WWTPPlantOptimizer(config_path=config_file)
        
        # 2. Build the Pyomo model
        optimizer.build_pyomo_model()
        
        # 3. Solve the optimization problem
        # The 'tee=True' argument will print the solver's log to the console.
        optimizer.solve(solver='ipopt', tee=True, max_iterations=10000)
        
        # 4. Report and save the results
        optimizer.report_results()

    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: A required file was not found.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("Please ensure the configuration Excel file and surrogate model files are in the correct locations.", file=sys.stderr)
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED DURING EXECUTION: {e}", file=sys.stderr)
        traceback.print_exc()