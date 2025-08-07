import os
import sys
import pyomo.environ as pyo
import pandas as pd
import joblib
import itertools
import traceback
import numpy as np

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
    An optimizer for a Wastewater Treatment Plant (WWTP) flowsheet or individual units.
    
    This class encapsulates the entire process:
    1. Loading plant configuration from an Excel file.
    2. Building a Pyomo Non-Linear Programming (NLP) model for the selected scope.
    3. Handling interconnections or isolated unit inputs.
    4. Solving the model to find optimal operating conditions.
    5. Reporting and saving the results to a standardized Excel file.
    """

    def __init__(self, config_path: str):
        """Initializes the optimizer by loading all configuration data."""
        print("--- Initializing WWTP Optimizer ---")
        self.config_path = config_path
        self._load_configuration()
        self._process_data()
        self.model = None
        self.results = None
        self.optimization_target = None
        self.active_model_level_dvars = []

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
        return name

    def _load_configuration(self):
        """Loads all data from the Excel configuration file."""
        print("1. Loading configuration from Excel file...")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at '{self.config_path}'")
        
        self.xls = pd.read_excel(self.config_path, sheet_name=None)
        
        self.full_flow_units = ['A1', 'A2', 'O1', 'O2', 'O3', 'C1']
        self.full_flow_cstr_units = ['A1', 'A2', 'O1', 'O2', 'O3']
        self.full_flow_clarifier_units = ['C1']

    def _process_data(self):
        """Processes the loaded dataframes to prepare for model building."""
        print("2. Processing and structuring configuration data...")
        
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

        df_dvar = self.xls['decision_var']
        self.plant_wide_dvars = df_dvar[df_dvar['Process Unit'] == 'plant-wide']['I_variables'].tolist()
        self.cstr_dvars = df_dvar[df_dvar['Process Unit'] == 'cstr']['I_variables'].tolist()
        self.clarifier_dvars = df_dvar[df_dvar['Process Unit'] == 'clarifier']['I_variables'].tolist()
        print(f"   - Plant-wide Decision Variables: {self.plant_wide_dvars}")
        print(f"   - CSTR-specific Decision Variables: {self.cstr_dvars}")
        print(f"   - Clarifier-specific Decision Variables: {self.clarifier_dvars}")

        df_influent = self.xls['raw_influent_compound_conc']
        self.influent_params = pd.Series(df_influent.Value.values, index=df_influent.Variable).to_dict()
        
        c1_outputs = self.surrogate_models['clarifier']['y_scaler'].feature_names_in_
        self.all_components = sorted(list(set([self._unify_comp_name(c) for c in c1_outputs])))
        print(f"   - Unified {len(self.all_components)} components for material balance.")

        df_bounds = self.xls['decision_var_bound']
        self.bounds = {row.Variable: (row.LowerBound, row.UpperBound) for _, row in df_bounds.iterrows()}
        # Define reasonable bounds for concentrations for use in McCormick relaxations
        self.UNSCALED_VAR_BOUNDS = UNSCALED_VAR_BOUNDS # (min, max) for component concentrations in mg/L

        df_goals = self.xls['fuzzy_goal']
        self.effluent_goals = {row.Goal: {'target': row.Target, 'max': row.Max} for _, row in df_goals.iterrows() if pd.notna(row.Target)}
        print(f"   - Identified {len(self.bounds)} decision variable bounds.")
        print(f"   - Identified {len(self.effluent_goals)} fuzzy effluent goals.")

    def build_pyomo_model(self, optimization_target: str):
        """Constructs the Pyomo model based on the selected target."""
        print(f"3. Building Pyomo NLP model for '{optimization_target.upper()}'...")
        self.model = m = pyo.ConcreteModel(f"WWTP_{optimization_target}_Optimization")
        self.optimization_target = optimization_target

        if optimization_target == 'aao':
            self.active_units = self.full_flow_units
            self.active_cstr_units = self.full_flow_cstr_units
            self.active_clarifier_units = self.full_flow_clarifier_units
            self.active_model_level_dvars = self.plant_wide_dvars
        elif optimization_target == 'cstr':
            self.active_units = ['CSTR1']
            self.active_cstr_units = ['CSTR1']
            self.active_clarifier_units = []
            self.active_model_level_dvars = ['Q_raw_inf']
        elif optimization_target == 'clarifier': # Now means Activated Sludge model
            self.active_units = ['CSTR1', 'C1']
            self.active_cstr_units = ['CSTR1']
            self.active_clarifier_units = ['C1']
            self.active_model_level_dvars = self.plant_wide_dvars
        else:
            raise ValueError(f"Unknown optimization target: {optimization_target}")

        m.UNITS = pyo.Set(initialize=self.active_units)
        m.CSTR_UNITS = pyo.Set(initialize=self.active_cstr_units)
        m.CLARIFIER_UNITS = pyo.Set(initialize=self.active_clarifier_units)
        m.COMPONENTS = pyo.Set(initialize=self.all_components)
        m.GOALS = pyo.Set(initialize=self.effluent_goals.keys())

        for dv in self.active_model_level_dvars:
            if dv in self.bounds:
                setattr(m, dv, pyo.Var(bounds=self.bounds[dv], initialize=sum(self.bounds[dv])/2, within=pyo.NonNegativeReals))
            else:
                setattr(m, dv, pyo.Var(initialize=1000, within=pyo.NonNegativeReals))

        if optimization_target in ['aao', 'clarifier']: # AAO and AS models have these constraints
            m.Q_int_upper_bound = pyo.Constraint(expr=m.Q_int <= m.Q_raw_inf)
            m.Q_was_upper_bound = pyo.Constraint(expr=m.Q_was <= m.Q_raw_inf)
        
        if self.active_cstr_units:
            m.cstr_dvars = pyo.Var(m.CSTR_UNITS, pyo.Set(initialize=self.cstr_dvars), 
                                   bounds=lambda m, u, dv: self.bounds[dv], 
                                   initialize=lambda m, u, dv: sum(self.bounds[dv])/2, within=pyo.NonNegativeReals)
            m.stream_conc = pyo.Var(m.CSTR_UNITS, m.COMPONENTS, bounds=self.UNSCALED_VAR_BOUNDS, initialize=10)

        if self.active_clarifier_units:
            m.clarifier_dvars = pyo.Var(m.CLARIFIER_UNITS, pyo.Set(initialize=self.clarifier_dvars),
                                        bounds=lambda m, u, dv: self.bounds[dv],
                                        initialize=lambda m, u, dv: sum(self.bounds[dv])/2, within=pyo.NonNegativeReals)
            m.C1_wastage_conc = pyo.Var(m.COMPONENTS, bounds=self.UNSCALED_VAR_BOUNDS, initialize=10)
            m.C1_effluent_conc = pyo.Var(m.COMPONENTS, bounds=self.UNSCALED_VAR_BOUNDS, initialize=10)

        if optimization_target in ['aao', 'clarifier']:
            # --- Mass Balance Linearization ---
            m.mass_flow_ext_recycle = pyo.Var(m.COMPONENTS) # C_ext * Q_ext
            if optimization_target == 'aao':
                m.mass_flow_int_recycle = pyo.Var(m.COMPONENTS) # C_int * Q_int (from O3)
            elif optimization_target == 'clarifier':
                m.mass_flow_AS_recycle = pyo.Var(m.COMPONENTS) # C_int * Q_int (from CSTR1)

            C_L, C_U = self.UNSCALED_VAR_BOUNDS
            Q_ext_L, Q_ext_U = self.bounds['Q_ext']
            Q_int_L, Q_int_U = self.bounds['Q_int']

            @m.Constraint(m.COMPONENTS)
            def mc_ext_1(m, c): return m.mass_flow_ext_recycle[c] >= C_L * m.Q_ext + Q_ext_L * m.C1_wastage_conc[c] - C_L * Q_ext_L
            @m.Constraint(m.COMPONENTS)
            def mc_ext_2(m, c): return m.mass_flow_ext_recycle[c] >= C_U * m.Q_ext + Q_ext_U * m.C1_wastage_conc[c] - C_U * Q_ext_U
            @m.Constraint(m.COMPONENTS)
            def mc_ext_3(m, c): return m.mass_flow_ext_recycle[c] <= C_U * m.Q_ext + Q_ext_L * m.C1_wastage_conc[c] - C_U * Q_ext_L
            @m.Constraint(m.COMPONENTS)
            def mc_ext_4(m, c): return m.mass_flow_ext_recycle[c] <= C_L * m.Q_ext + Q_ext_U * m.C1_wastage_conc[c] - C_L * Q_ext_U

            if optimization_target == 'aao':
                @m.Constraint(m.COMPONENTS)
                def mc_int_aao_1(m, c): return m.mass_flow_int_recycle[c] >= C_L * m.Q_int + Q_int_L * m.stream_conc['O3', c] - C_L * Q_int_L
                @m.Constraint(m.COMPONENTS)
                def mc_int_aao_2(m, c): return m.mass_flow_int_recycle[c] >= C_U * m.Q_int + Q_int_U * m.stream_conc['O3', c] - C_U * Q_int_U
                @m.Constraint(m.COMPONENTS)
                def mc_int_aao_3(m, c): return m.mass_flow_int_recycle[c] <= C_U * m.Q_int + Q_int_L * m.stream_conc['O3', c] - C_U * Q_int_L
                @m.Constraint(m.COMPONENTS)
                def mc_int_aao_4(m, c): return m.mass_flow_int_recycle[c] <= C_L * m.Q_int + Q_int_U * m.stream_conc['O3', c] - C_L * Q_int_U
            elif optimization_target == 'clarifier':
                @m.Constraint(m.COMPONENTS)
                def mc_int_as_1(m, c): return m.mass_flow_AS_recycle[c] >= C_L * m.Q_int + Q_int_L * m.stream_conc['CSTR1', c] - C_L * Q_int_L
                @m.Constraint(m.COMPONENTS)
                def mc_int_as_2(m, c): return m.mass_flow_AS_recycle[c] >= C_U * m.Q_int + Q_int_U * m.stream_conc['CSTR1', c] - C_U * Q_int_U
                @m.Constraint(m.COMPONENTS)
                def mc_int_as_3(m, c): return m.mass_flow_AS_recycle[c] <= C_U * m.Q_int + Q_int_L * m.stream_conc['CSTR1', c] - C_U * Q_int_L
                @m.Constraint(m.COMPONENTS)
                def mc_int_as_4(m, c): return m.mass_flow_AS_recycle[c] <= C_L * m.Q_int + Q_int_U * m.stream_conc['CSTR1', c] - C_L * Q_int_U
        
        m.lambda_o = pyo.Var(initialize=0.5, within=pyo.Reals)
        m.lambda_g = pyo.Var(m.GOALS, initialize=0.5, within=pyo.Reals)
        m.objective = pyo.Objective(expr=m.lambda_o, sense=pyo.maximize)
        m.satisfaction_rule = pyo.Constraint(m.GOALS, rule=lambda m, g: m.lambda_o <= m.lambda_g[g])

        self._add_unit_blocks()

        effluent_source, effluent_unit_id = None, None
        if optimization_target in ['aao', 'clarifier']:
            effluent_source = m.C1_effluent_conc
        elif optimization_target == 'cstr':
            effluent_source = m.stream_conc
            effluent_unit_id = 'CSTR1'
        
        self._add_goal_constraints(effluent_source, effluent_unit_id)
        print("...Pyomo model build complete.")

    def _add_unit_blocks(self):
        """Creates a Pyomo Block for each active process unit."""
        for unit in self.active_cstr_units: self._build_unit_submodel(unit, 'cstr')
        for unit in self.active_clarifier_units: self._build_unit_submodel(unit, 'clarifier')
    
    def _build_unit_submodel(self, unit_name, unit_type):
        """Helper to build the surrogate model for a single unit."""
        m = self.model
        b = pyo.Block()
        setattr(m, unit_name, b)

        model_data = self.surrogate_models[unit_type]
        x_scaler, y_scaler, coeffs = model_data['x_scaler'], model_data['y_scaler'], model_data['coeffs']

        b.M_names, b.K_names = list(x_scaler.feature_names_in_), list(y_scaler.feature_names_in_)
        b.M, b.K = pyo.Set(initialize=b.M_names), pyo.Set(initialize=b.K_names)
        b.L = pyo.Set(initialize=itertools.combinations(b.M_names, 2), dimen=2)
        m_map, k_map = {n: i for i, n in enumerate(b.M_names)}, {n: i for i, n in enumerate(b.K_names)}
        l_map = {n: i for i, n in enumerate(list(itertools.combinations(b.M_names, 2)))}
        
        w_bounds = (SCALED_VAR_BOUNDS[0] * SCALED_VAR_BOUNDS[1], SCALED_VAR_BOUNDS[1]**2)
        
        b.X = pyo.Var(b.M)
        b.X_s = pyo.Var(b.M, bounds=SCALED_VAR_BOUNDS)
        b.Y_s = pyo.Var(b.K, bounds=SCALED_VAR_BOUNDS)
        b.w_yx = pyo.Var(b.K, b.M, bounds=w_bounds)
        b.w_xx = pyo.Var(b.L, bounds=w_bounds)

        # Variables for piecewise linearization of exponential
        y_log_min = np.log(max(1e-6, self.UNSCALED_VAR_BOUNDS[0])) # Avoid log(0)
        y_log_max = np.log(self.UNSCALED_VAR_BOUNDS[1])
        b.Y_log = pyo.Var(b.K, bounds=(y_log_min, y_log_max), initialize=0)
        b.unscaled_Y = pyo.Var(b.K, bounds=self.UNSCALED_VAR_BOUNDS, initialize=10)

        is_mixing_point = (self.optimization_target == 'aao' and unit_name == 'A1') or \
                          (self.optimization_target == 'clarifier' and unit_name == 'CSTR1')

        if is_mixing_point:
            b.mass_in_raw = pyo.Var(b.M)
            b.mass_in_int = pyo.Var(b.M)
            b.mass_in_ext = pyo.Var(b.M)
            
            C_L, C_U = self.UNSCALED_VAR_BOUNDS
            Q_raw_L, Q_raw_U = self.bounds['Q_raw_inf']
            Q_int_L, Q_int_U = self.bounds['Q_int']
            Q_ext_L, Q_ext_U = self.bounds['Q_ext']

            @b.Constraint(b.M)
            def mc_local_raw_1(b, M): return b.mass_in_raw[M] >= C_L * m.Q_raw_inf + Q_raw_L * b.X[M] - C_L * Q_raw_L if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_raw_2(b, M): return b.mass_in_raw[M] >= C_U * m.Q_raw_inf + Q_raw_U * b.X[M] - C_U * Q_raw_U if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_raw_3(b, M): return b.mass_in_raw[M] <= C_U * m.Q_raw_inf + Q_raw_L * b.X[M] - C_U * Q_raw_L if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_raw_4(b, M): return b.mass_in_raw[M] <= C_L * m.Q_raw_inf + Q_raw_U * b.X[M] - C_L * Q_raw_U if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_int_1(b, M): return b.mass_in_int[M] >= C_L * m.Q_int + Q_int_L * b.X[M] - C_L * Q_int_L if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_int_2(b, M): return b.mass_in_int[M] >= C_U * m.Q_int + Q_int_U * b.X[M] - C_U * Q_int_U if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_int_3(b, M): return b.mass_in_int[M] <= C_U * m.Q_int + Q_int_L * b.X[M] - C_U * Q_int_L if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_int_4(b, M): return b.mass_in_int[M] <= C_L * m.Q_int + Q_int_U * b.X[M] - C_L * Q_int_U if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_ext_1(b, M): return b.mass_in_ext[M] >= C_L * m.Q_ext + Q_ext_L * b.X[M] - C_L * Q_ext_L if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_ext_2(b, M): return b.mass_in_ext[M] >= C_U * m.Q_ext + Q_ext_U * b.X[M] - C_U * Q_ext_U if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_ext_3(b, M): return b.mass_in_ext[M] <= C_U * m.Q_ext + Q_ext_L * b.X[M] - C_U * Q_ext_L if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip
            @b.Constraint(b.M)
            def mc_local_ext_4(b, M): return b.mass_in_ext[M] <= C_L * m.Q_ext + Q_ext_U * b.X[M] - C_L * Q_ext_U if self._unify_comp_name(M) in m.COMPONENTS else pyo.Constraint.Skip

        @b.Constraint(b.M)
        def input_assembly_rule(b, M_name):
            if unit_type == 'cstr' and M_name in self.cstr_dvars:
                return b.X[M_name] == m.cstr_dvars[unit_name, M_name]
            if unit_type == 'clarifier' and M_name in self.clarifier_dvars:
                return b.X[M_name] == m.clarifier_dvars[unit_name, M_name]
            if M_name in self.active_model_level_dvars:
                return b.X[M_name] == getattr(m, M_name)
            
            unified_name = self._unify_comp_name(M_name)
            
            if self.optimization_target == 'aao':
                if unit_name == 'A1':
                    if unified_name not in m.COMPONENTS: return pyo.Constraint.Skip
                    C_raw = self.influent_params.get(f"inf_{unified_name}", 0)
                    lhs = b.mass_in_raw[M_name] + b.mass_in_int[M_name] + b.mass_in_ext[M_name]
                    rhs = (C_raw * m.Q_raw_inf) + m.mass_flow_int_recycle[unified_name] + m.mass_flow_ext_recycle[unified_name]
                    return lhs == rhs
                else: 
                    if unified_name not in m.COMPONENTS: return pyo.Constraint.Skip
                    prev_unit = self.full_flow_units[self.full_flow_units.index(unit_name) - 1]
                    return b.X[M_name] == m.stream_conc[prev_unit, unified_name]
            
            elif self.optimization_target == 'clarifier': # AS Plant Logic
                if unit_name == 'CSTR1':
                    if unified_name not in m.COMPONENTS: return pyo.Constraint.Skip
                    C_raw = self.influent_params.get(f"inf_{unified_name}", 0)
                    lhs = b.mass_in_raw[M_name] + b.mass_in_int[M_name] + b.mass_in_ext[M_name]
                    rhs = (C_raw * m.Q_raw_inf) + m.mass_flow_AS_recycle[unified_name] + m.mass_flow_ext_recycle[unified_name]
                    return lhs == rhs
                elif unit_name == 'C1':
                    if unified_name not in m.COMPONENTS: return pyo.Constraint.Skip
                    return b.X[M_name] == m.stream_conc['CSTR1', unified_name]

            elif self.optimization_target == 'cstr': # Isolated CSTR Logic
                if M_name in self.plant_wide_dvars:
                    if M_name in ['Q_was', 'Q_ext', 'Q_int']: return b.X[M_name] == 0.0
                    if M_name in self.bounds: return b.X[M_name] == sum(self.bounds[M_name]) / 2.0
                    else: raise ValueError(f"Cannot fix parameter '{M_name}': no bounds or specific rule defined.")
                C_raw = self.influent_params.get(f"inf_{unified_name}", 0)
                return b.X[M_name] == C_raw

            return pyo.Constraint.Skip

        @b.Constraint(b.M)
        def x_scaling_rule(b, M_name):
            idx = m_map[M_name]
            mu, sigma = x_scaler.mean_[idx], x_scaler.scale_[idx]
            return b.X_s[M_name] == (b.X[M_name] - mu) / sigma
        
        x_L, x_U = SCALED_VAR_BOUNDS
        @b.Constraint(b.K, b.M)
        def mccormick_yx_1(b, k, m_): return b.w_yx[k, m_] >= x_L * b.X_s[m_] + x_L * b.Y_s[k] - x_L * x_L
        @b.Constraint(b.K, b.M)
        def mccormick_yx_2(b, k, m_): return b.w_yx[k, m_] >= x_U * b.X_s[m_] + x_U * b.Y_s[k] - x_U * x_U
        @b.Constraint(b.K, b.M)
        def mccormick_yx_3(b, k, m_): return b.w_yx[k, m_] <= x_U * b.X_s[m_] + x_L * b.Y_s[k] - x_U * x_L
        @b.Constraint(b.K, b.M)
        def mccormick_yx_4(b, k, m_): return b.w_yx[k, m_] <= x_L * b.X_s[m_] + x_U * b.Y_s[k] - x_L * x_U
        @b.Constraint(b.L)
        def mccormick_xx_1(b, m1, m2): return b.w_xx[m1, m2] >= x_L * b.X_s[m2] + x_L * b.X_s[m1] - x_L * x_L
        @b.Constraint(b.L)
        def mccormick_xx_2(b, m1, m2): return b.w_xx[m1, m2] >= x_U * b.X_s[m2] + x_U * b.X_s[m1] - x_U * x_U
        @b.Constraint(b.L)
        def mccormick_xx_3(b, m1, m2): return b.w_xx[m1, m2] <= x_U * b.X_s[m2] + x_L * b.X_s[m1] - x_U * x_L
        @b.Constraint(b.L)
        def mccormick_xx_4(b, m1, m2): return b.w_xx[m1, m2] <= x_L * b.X_s[m2] + x_U * b.X_s[m1] - x_L * x_U

        @b.Constraint(b.K)
        def CBREM_model_rule(b, K_name):
            k_idx = k_map[K_name]
            term1 = b.Y_s[K_name]
            term2 = sum(coeffs['Lambda'][k_idx, m_map[m_n]] * b.w_yx[K_name, m_n] for m_n in b.M)
            term3 = sum(coeffs['Gamma'][k_idx, k_map[kp]] * b.Y_s[kp] for kp in b.K)
            rhs_const = coeffs['Upsilon'][k_idx, 0]
            rhs_B = sum(coeffs['B'][k_idx, m_map[m_n]] * b.X_s[m_n] for m_n in b.M)
            rhs_Theta = sum(coeffs['Theta'][k_idx, l_map[l_p]] * b.w_xx[l_p] for l_p in b.L)
            return term1 - term2 - term3 == rhs_const + rhs_B + rhs_Theta

        # --- Piecewise Linear Approximation of the Exponential Function ---
        # Define the relationship between the scaled variable Y_s and the log-transformed Y_log
        @b.Constraint(b.K)
        def y_log_definition_rule(b, K_name):
            k_idx = k_map[K_name]
            mu, sigma = y_scaler.mean_[k_idx], y_scaler.scale_[k_idx]
            return b.Y_log[K_name] == b.Y_s[K_name] * sigma + mu
        
        # Generate exponential breakpoints
        K = N_PW_PTS
        w_min = y_log_min
        w_max = y_log_max
        q_values = np.arange(K)
        pw_domain_pts = (w_min + (q_values / (K - 1))**2 * (w_max - w_min)).tolist()

        # Define the piecewise linear approximation for unscaled_Y = exp(Y_log)
        b.exp_approximation = pyo.Piecewise(
            b.K,                  # Indexed by K
            b.unscaled_Y,         # y-var: The resulting concentration
            b.Y_log,              # x-var: The log of the concentration
            pw_pts=pw_domain_pts,
            pw_repn=PLA_TYPE,
            pw_constr_type='EQ',  # y = f(x)
            f_rule=lambda model, k, x: pyo.exp(x) # The function to approximate
        )

        @b.Constraint(b.K)
        def y_unscaling_rule(b, K_name):
            unified_name = self._unify_comp_name(K_name)
            
            if unit_type == 'cstr':
                return m.stream_conc[unit_name, unified_name] == b.unscaled_Y[K_name]
            else: # clarifier unit
                if 'Effluent' in K_name:
                    return m.C1_effluent_conc[unified_name] == b.unscaled_Y[K_name]
                elif 'Wastage' in K_name:
                    return m.C1_wastage_conc[unified_name] == b.unscaled_Y[K_name]
            return pyo.Constraint.Skip

    def _add_goal_constraints(self, effluent_source, effluent_unit_id=None):
        """Adds the fuzzy goal constraints for effluent quality using a generic source."""
        if effluent_source is None: return
        m = self.model
        m.FuzzyRules = pyo.ConstraintList()
        epsilon = 1e-9
        for goal_name, limits in self.effluent_goals.items():
            x_min, x_max = limits['target'], limits['max']
            denominator = x_max - x_min
            comp_name = self._unify_comp_name(goal_name)
            
            x = effluent_source[effluent_unit_id, comp_name] if effluent_unit_id else effluent_source[comp_name]
            m.FuzzyRules.add(m.lambda_g[goal_name] == 1 - (x - x_min) / (denominator + epsilon))

    def solve(self, solver='highs',tee=True):
        """Solves the optimization problem."""
        if self.model is None: raise RuntimeError("Model has not been built yet. Call build_pyomo_model() first.")
        print(f"\n4. Solving the optimization problem with '{solver}'...")
        solver_instance = pyo.SolverFactory(solver)
        if not solver_instance.available(exception_flag=False):
            print(f"ERROR: Solver '{solver}' not found. Please install it.", file=sys.stderr)
            return None
        
        self.results = solver_instance.solve(self.model, tee=tee)
        return self.results
    
    def report_results(self):
        """Saves the optimization results to a standardized Excel file."""
        if self.results is None or self.model is None:
            print("No results to report."); return
        m = self.model
        term_cond = self.results.solver.termination_condition

        if term_cond not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal]:
            print("\n" + "="*80 + "\n" + "OPTIMIZATION FAILED".center(80) + "\n" + "="*80)
            print(f"Solver Status: {self.results.solver.status}, Termination: {term_cond}")
            print("="*80); return

        print("\n---> Optimal Solution Found <---")
        print(f"   - Overall Satisfaction Level (lambda_o): {pyo.value(m.lambda_o):.4f}")
        print(f"   - Saving results to data/optimization_results.xlsx")

        dvar_data = []
        if self.optimization_target in ['aao', 'clarifier']: # AAO and AS Plant have same DV structure
            for dv in self.plant_wide_dvars:
                if dv != 'Q_int': dvar_data.append({'Process Unit': 'plant-wide', 'Variable': dv, 'Optimal Value': pyo.value(getattr(m, dv))})
            for u, dv in m.cstr_dvars: dvar_data.append({'Process Unit': u, 'Variable': dv, 'Optimal Value': pyo.value(m.cstr_dvars[u, dv])})
            for u, dv in m.clarifier_dvars: dvar_data.append({'Process Unit': u, 'Variable': dv, 'Optimal Value': pyo.value(m.clarifier_dvars[u, dv])})
            
            q_int_val = pyo.value(m.Q_int)
            q_raw_inf_val = pyo.value(m.Q_raw_inf)
            q_ext_val = pyo.value(m.Q_ext)

            if self.optimization_target == 'aao':
                denominator = q_raw_inf_val + q_int_val + q_ext_val
                split_unit = 'O3'
            else: # AS Plant
                denominator = q_raw_inf_val + q_ext_val + q_int_val
                split_unit = 'CSTR1'
            split_val = q_int_val / denominator if denominator != 0 else 0
            dvar_data.append({'Process Unit': split_unit, 'Variable': f'{split_unit}_split_internal', 'Optimal Value': split_val})
        
        elif self.optimization_target == 'cstr':
            dvar_data.append({'Process Unit': 'plant-wide', 'Variable': 'Q_raw_inf', 'Optimal Value': pyo.value(m.Q_raw_inf)})
            for u, dv in m.cstr_dvars: dvar_data.append({'Process Unit': u, 'Variable': dv, 'Optimal Value': pyo.value(m.cstr_dvars[u, dv])})
        df_dvars = pd.DataFrame(dvar_data)

        influent_data = [{'Variable': key.replace('inf_', ''), 'Value (mg/L)': value} for key, value in self.influent_params.items()]
        df_influent = pd.DataFrame(influent_data)

        predicted_data = []
        if self.optimization_target in ['aao', 'clarifier']:
            for u in self.active_cstr_units:
                for c in self.all_components: predicted_data.append({'Component': f"{c}_{u}", 'Predicted Value (mg/L)': pyo.value(m.stream_conc[u, c])})
            for c in self.all_components:
                predicted_data.append({'Component': f"{c}_Effluent", 'Predicted Value (mg/L)': pyo.value(m.C1_effluent_conc[c])})
                predicted_data.append({'Component': f"{c}_Wastage", 'Predicted Value (mg/L)': pyo.value(m.C1_wastage_conc[c])})
        elif self.optimization_target == 'cstr':
            for c in self.all_components: predicted_data.append({'Component': f"{c}_Effluent", 'Predicted Value (mg/L)': pyo.value(m.stream_conc['CSTR1', c])})
        df_predicted = pd.DataFrame(predicted_data)

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

    SCALED_VAR_BOUNDS = (-0.5, 0.5)
    UNSCALED_VAR_BOUNDS = (0.0, 12000.0)
    N_PW_PTS = 50  # Number of breakpoints for the approximation
    PLA_TYPE = 'MC'

    try:
        print("\nWhich model would you like to optimize?")
        print("1. cstr (Single CSTR unit)")
        print("2. clarifier (Single CSTR + Clarifier Plant)")
        print("3. aao (Full AAO plant)")
        
        choice_map = {'1': 'cstr', '2': 'clarifier', '3': 'aao'}
        user_choice = input("Enter your choice (1, 2, or 3): ").strip()

        if user_choice not in choice_map:
            print(f"Invalid choice '{user_choice}'. Please run the script again and select 1, 2, or 3.", file=sys.stderr)
            sys.exit(1)
        
        optimization_target = choice_map[user_choice]
        
        config_file = os.path.join('data', 'optimization_config.xlsx')
        optimizer = WWTPPlantOptimizer(config_path=config_file)
        optimizer.build_pyomo_model(optimization_target=optimization_target)
        optimizer.solve(solver='highs', tee=True)
        optimizer.report_results()
        
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: A required file was not found. Details: {e}", file=sys.stderr)
        print("Please ensure the 'data/optimization_config.xlsx' file exists.", file=sys.stderr)
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}", file=sys.stderr)
        traceback.print_exc()