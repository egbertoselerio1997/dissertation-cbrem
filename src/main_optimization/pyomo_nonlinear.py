import os
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import joblib
import itertools
import sys

# --- PyTorch Class Definition (Required for loading the joblib file) ---
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
    class CoupledCLEFOModel: pass

# --- Analytical Reconstruction Functions (from independent script) ---
def create_interaction_features_np(X: np.ndarray, m: int):
    """Generates pairwise interaction features (Z) from independent variables (X) using NumPy."""
    col_pairs = list(itertools.combinations(range(m), 2))
    q = len(col_pairs)
    N = X.shape[0]

    Z = np.zeros((N, q), dtype=np.float64)
    for i, (col1_idx, col2_idx) in enumerate(col_pairs):
        Z[:, i] = X[:, col1_idx] * X[:, col2_idx]
    return Z

def predict_analytical(X_scaled: np.ndarray, Z_scaled: np.ndarray, coeffs: dict) -> np.ndarray:
    """
    Reconstructs the CLEFO model analytically and makes predictions using NumPy.
    Equation: Y = (I - Γ - diag(ΛX))⁻¹ * (Υ + BX + ΘZ)
    """
    Upsilon = coeffs['Upsilon']
    B = coeffs['B']
    Theta = coeffs['Theta']
    Gamma = coeffs['Gamma']
    Lambda = coeffs['Lambda']

    n_samples = X_scaled.shape[0]
    n_dep = Upsilon.shape[0]
    predictions = np.zeros((n_samples, n_dep))
    I = np.eye(n_dep)
    
    for i in range(n_samples):
        x_sample = X_scaled[i].reshape(-1, 1)
        z_sample = Z_scaled[i].reshape(-1, 1)
        
        RHS = Upsilon + (B @ x_sample) + (Theta @ z_sample)
        
        lambda_x = Lambda @ x_sample
        diag_lambda_x = np.diag(lambda_x.flatten())
        LHS = I - Gamma - diag_lambda_x
        
        try:
            y_pred_solved = np.linalg.solve(LHS, RHS)
            predictions[i, :] = y_pred_solved.flatten()
        except np.linalg.LinAlgError:
            predictions[i, :] = np.nan

    return predictions

def load_data_and_params():
    """
    Loads all model coefficients from the joblib file and all other parameters
    from the Excel configuration file.
    """
    print("1. Loading data and parameters from file...")
    params = {}

    config_path = os.path.join('data', 'optimization_config.xlsx')

    xls = pd.read_excel(config_path, sheet_name=None)
    
    df_config = xls['config'].set_index('Parameter')
    model_path = df_config.loc['model_path', 'Value']
    params['M_penalty'] = float(df_config.loc['M_penalty', 'Value'])

    params['I'] = xls['decision_var']['I_variables'].tolist()

    df_bounds = xls['decision_var_bound']
    params['x_bounds'] = {
        row.Variable: (row.LowerBound, row.UpperBound) 
        for _, row in df_bounds.iterrows()
    }

    df_defaults = xls['influent_compound_conc']
    params['defaults'] = pd.Series(df_defaults.Value.values, index=df_defaults.Variable).to_dict()

    df_cost = xls['cost_param']
    params['cost_params'] = pd.Series(df_cost.Value.values, index=df_cost.Parameter).to_dict()

    df_goals = xls['fuzzy_goal']
    params['fuzzy_goals'] = {
        row.Goal: {'target': row.Target, 'max': row.Max} 
        for _, row in df_goals.iterrows()
    }
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Please ensure the path in '{config_path}' is correct."
        )

    model_bundle = joblib.load(model_path)
    
    params['coeffs'] = {k: v for k, v in model_bundle.items() if k not in ['model', 'x_scaler', 'y_scaler']}
    params['x_scaler'] = model_bundle['x_scaler']
    params['y_scaler'] = model_bundle['y_scaler']

    raw_m_names = list(params['x_scaler'].feature_names_in_)
    params['M'] = [name.split(' (')[0].strip() for name in raw_m_names]
    
    raw_k_names = list(params['y_scaler'].feature_names_in_)
    params['K'] = [name.split(' (')[0].strip() for name in raw_k_names]
    
    params['J'] = [v for v in params['M'] if v not in params['I']]
    params['L'] = list(itertools.combinations(params['M'], 2))

    params['G'] = list(params['fuzzy_goals'].keys())

    params['scaled_fuzzy_goals'] = {}
    y_scaler = params['y_scaler']
    for goal, limits in params['fuzzy_goals'].items():
        if goal.startswith('Effluent_'):
            try:
                idx = params['K'].index(goal)
                scaled_target = (limits['target'] - y_scaler.mean_[idx]) / y_scaler.scale_[idx]
                scaled_max = (limits['max'] - y_scaler.mean_[idx]) / y_scaler.scale_[idx]
                params['scaled_fuzzy_goals'][goal] = {'target_s': scaled_target, 'max_s': scaled_max}
            except (ValueError, IndexError):
                print(f"Warning: Fuzzy goal '{goal}' not found in model outputs. Skipping.")
                pass

    print("...Data loading complete.")
    return params

def build_nlp_model(params):
    """
    Builds the Pyomo NLP model based on the mathematical formulation.
    """
    print("2. Building Pyomo NLP model...")
    model = pyo.ConcreteModel("WWTP_Optimal_Operation_NLP")

    model.I = pyo.Set(initialize=params['I'])
    model.K = pyo.Set(initialize=params['K'])
    model.M = pyo.Set(initialize=params['M'])
    model.L = pyo.Set(initialize=params['L'], dimen=2)
    model.G = pyo.Set(initialize=params['G'])
    model.K_goals = pyo.Set(initialize=[k for k in params['K'] if k in params['G']])

    k_map = {name: i for i, name in enumerate(params['K'])}
    m_map = {name: i for i, name in enumerate(params['M'])}
    l_map = {name: i for i, name in enumerate(params['L'])}

    model.x = pyo.Var(model.I, bounds=lambda m, i: params['x_bounds'][i], initialize=lambda m, i: np.mean(params['x_bounds'][i]))
    
    # REVISION: lambda_g and lambda_o have no bounds as requested
    model.lambda_g = pyo.Var(model.G, initialize=0.5)
    model.lambda_o = pyo.Var(initialize=0.5)
    
    model.X = pyo.Var(model.M)
    model.X_s = pyo.Var(model.M)
    model.Y = pyo.Var(model.K, within=pyo.NonNegativeReals)
    model.Y_s = pyo.Var(model.K)
    model.V_reactor = pyo.Var(within=pyo.NonNegativeReals)
    model.C_purch = pyo.Var(within=pyo.NonNegativeReals)
    model.CAPEX = pyo.Var(within=pyo.NonNegativeReals)
    model.AOC = pyo.Var(within=pyo.NonNegativeReals)

    model.objective = pyo.Objective(expr=model.lambda_o, sense=pyo.maximize)

    @model.Constraint(model.G)
    def overall_satisfaction_rule(m, g):
        return m.lambda_o <= m.lambda_g[g]

    @model.Constraint(model.M)
    def x_assembly_rule(m, M_name):
        return m.X[M_name] == (m.x[M_name] if M_name in m.I else params['defaults'][M_name])

    @model.Constraint(model.M)
    def x_scaling_rule(m, M_name):
        idx = m_map[M_name]
        mu = params['x_scaler'].mean_[idx]
        sigma = params['x_scaler'].scale_[idx]
        return m.X_s[M_name] == (m.X[M_name] - mu) / sigma

    @model.Constraint(model.K)
    def y_unscaling_rule(m, K_name):
        idx = k_map[K_name]
        mu = params['y_scaler'].mean_[idx]
        sigma = params['y_scaler'].scale_[idx]
        return m.Y[K_name] == m.Y_s[K_name] * sigma + mu

    @model.Constraint(model.K)
    def clefo_model_rule(m, k):
        k_idx, c = k_map[k], params['coeffs']
        term1 = m.Y_s[k]
        term2 = sum(c['Lambda'][k_idx, m_map[m_n]] * m.Y_s[k] * m.X_s[m_n] for m_n in m.M)
        term3 = sum(c['Gamma'][k_idx, k_map[kp]] * m.Y_s[kp] for kp in m.K)
        rhs_const = c['Upsilon'][k_idx, 0]
        rhs_B = sum(c['B'][k_idx, m_map[m_n]] * m.X_s[m_n] for m_n in m.M)
        rhs_Theta = sum(c['Theta'][k_idx, l_map[l_p]] * m.X_s[l_p[0]] * m.X_s[l_p[1]] for l_p in m.L)
        return term1 - term2 - term3 - (rhs_const + rhs_B + rhs_Theta) == 0

    @model.Constraint()
    def reactor_volume_rule(m):
        return m.V_reactor == m.x['flow_rate'] * (m.x['HRT'] / 24.0)

    @model.Constraint()
    def purchase_cost_rule(m):
        cp = params['cost_params']
        return m.C_purch == cp['C_base'] * (m.V_reactor / cp['V_base'])**cp['n']

    @model.Constraint()
    def capex_rule(m):
        return m.CAPEX == m.C_purch * params['cost_params']['F_BM']

    @model.Constraint()
    def aoc_rule(m):
        cp = params['cost_params']
        aeration_power_kW = (m.V_reactor * cp['rho_O2_bio'] / 24) / cp['eta_aerator']
        mixing_power_kW = cp['P_mix_specific'] * m.V_reactor
        annual_elec_cost = (aeration_power_kW + mixing_power_kW) * cp['p_elec'] * cp['H_op']
        annual_maint_cost = m.CAPEX * cp['f_maint']
        return m.AOC == annual_elec_cost + annual_maint_cost + cp['C_labor']
    
    # REVISION: Fuzzy constraints are now defined as EQUALITIES.
    # This correctly DEFINES lambda_g based on the outcome variable x.
    # The formula is lambda_g = 1 - (x - x_min) / (x_max - x_min)
    model.FuzzyConstraints = pyo.ConstraintList()
    epsilon = 1e-9 # To prevent division by zero if max == target

    # Effluent goals (using scaled values for better numerical stability)
    for k in model.K_goals:
        limits = params['scaled_fuzzy_goals'][k]
        x_s = model.Y_s[k]
        x_min_s = limits['target_s']
        x_max_s = limits['max_s']
        denominator_s = x_max_s - x_min_s
        model.FuzzyConstraints.add(
            model.lambda_g[k] == 1 - (x_s - x_min_s) / (denominator_s + epsilon)
        )

    # AOC goal (using unscaled values)
    aoc_goal = params['fuzzy_goals']['AOC']
    x_aoc = model.AOC
    x_min_aoc = aoc_goal['target']
    x_max_aoc = aoc_goal['max']
    denominator_aoc = x_max_aoc - x_min_aoc
    model.FuzzyConstraints.add(
        model.lambda_g['AOC'] == 1 - (x_aoc - x_min_aoc) / (denominator_aoc + epsilon)
    )

    # CAPEX goal (using unscaled values)
    capex_goal = params['fuzzy_goals']['CAPEX']
    x_capex = model.CAPEX
    x_min_capex = capex_goal['target']
    x_max_capex = capex_goal['max']
    denominator_capex = x_max_capex - x_min_capex
    model.FuzzyConstraints.add(
        model.lambda_g['CAPEX'] == 1 - (x_capex - x_min_capex) / (denominator_capex + epsilon)
    )
    
    print("...NLP model build complete.")
    return model

def solve_and_display_results(model, params):
    """
    Solves the Pyomo model using ipopt and prints a detailed report.
    """
    print("3. Solving with ipopt...")
    solver = pyo.SolverFactory('ipopt')
    if not solver.available(exception_flag=False):
        print("ERROR: ipopt solver not found. Please install it and ensure it's in your system's PATH.", file=sys.stderr)
        return

    results = solver.solve(model, tee=True)
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS REPORT".center(80))
    print("="*80)
    
    term_cond = results.solver.termination_condition
    print(f"Solver Status: {term_cond}")

    if term_cond in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal]:
        v_reactor, c_purch, capex, aoc = (pyo.value(v) for v in [model.V_reactor, model.C_purch, model.CAPEX, model.AOC])
        cp = params['cost_params']
        f_bm = cp['F_BM']
        
        aeration_power_kW = (v_reactor * cp['rho_O2_bio'] / 24) / cp['eta_aerator']
        mixing_power_kW = cp['P_mix_specific'] * v_reactor
        
        annual_aeration_cost = aeration_power_kW * cp['p_elec'] * cp['H_op']
        annual_mixing_cost = mixing_power_kW * cp['p_elec'] * cp['H_op']
        total_elec_cost = annual_aeration_cost + annual_mixing_cost
        
        annual_maint_cost = capex * cp['f_maint']
        annual_labor_cost = cp['C_labor']

        print("\n---> Optimal Solution Found <---\n")
        lambda_o_val = pyo.value(model.lambda_o)
        print(f"Overall Satisfaction Level (lambda_o): {lambda_o_val:.4f}")

        print("\n--- Individual Goal Satisfaction (lambda_g) ---")
        for g in sorted(model.G):
            print(f"{g:<20}: {pyo.value(model.lambda_g[g]):.4f}")
        
        print("\n--- Optimal Decision Variables ---")
        for i in model.I:
            print(f"{i:<12}: {pyo.value(model.x[i]):10.2f} (Bounds: {params['x_bounds'][i]})")

        print("\n--- CAPEX Breakdown ---")
        capex_goal = params['fuzzy_goals']['CAPEX']
        capex_satisfaction = pyo.value(model.lambda_g['CAPEX'])
        print(f"{'Reactor Volume (V_reactor)':<30}: {v_reactor:12,.2f} m^3")
        print(f"{'Reactor Purchase Cost (C_purch)':<30}: ${c_purch:11,.2f}")
        print(f"{'Bare-Module Factor (F_BM)':<30}: {f_bm:12.2f} x")
        print("-" * 45)
        print(f"{'Total CAPEX':<30}: ${capex:11,.2f} (Target: ${capex_goal['target']:,.0f}, Max: ${capex_goal['max']:,.0f}, Satisfaction: {capex_satisfaction:.3f})")

        print("\n--- AOC Breakdown ---")
        aoc_goal = params['fuzzy_goals']['AOC']
        aoc_satisfaction = pyo.value(model.lambda_g['AOC'])
        print(f"  {'1. Electricity Cost':<28}: ${total_elec_cost:11,.2f} / yr")
        print(f"     {'Aeration':<25}: ${annual_aeration_cost:11,.2f} / yr")
        print(f"     {'Mixing':<25}: ${annual_mixing_cost:11,.2f} / yr")
        print(f"  {'2. Maintenance Cost':<28}: ${annual_maint_cost:11,.2f} / yr")
        print(f"  {'3. Labor Cost':<28}: ${annual_labor_cost:11,.2f} / yr")
        print("-" * 45)
        print(f"{'Total AOC':<30}: ${aoc:11,.2f} / yr (Target: ${aoc_goal['target']:,.0f}, Max: ${aoc_goal['max']:,.0f}, Satisfaction: {aoc_satisfaction:.3f})")

        print("\n--- All Predicted Effluent Quality (mg/L) ---")
        for k in sorted(model.K):
            val = pyo.value(model.Y[k])
            if k in params['fuzzy_goals']:
                goal = params['fuzzy_goals'][k]
                satisfaction = pyo.value(model.lambda_g[k])
                print(f"{k:<20}: {val:8.3f} (Target: {goal['target']:<4.1f}, Max: {goal['max']:<4.1f}, Satisfaction: {satisfaction:.3f})")
            else:
                print(f"{k:<20}: {val:8.3f} (Not a primary goal)")
        
        # --- [START] ADDED CODE FOR SAVING RESULTS TO EXCEL ---
        print("\n--- Saving results to data/data.xlsx ---")
        try:
            output_folder = 'data'
            output_filename = 'data.xlsx'
            output_path = os.path.join(output_folder, output_filename)

            # Ensure the output directory exists
            os.makedirs(output_folder, exist_ok=True)

            # 1. Prepare DataFrames for each sheet
            # Sheet: optimal_satistfaction
            satisfaction_data = {'Metric': ['Overall Satisfaction Level (lambda_o)'], 'Value': [pyo.value(model.lambda_o)]}
            for g in sorted(model.G):
                satisfaction_data['Metric'].append(f'Individual Goal Satisfaction ({g})')
                satisfaction_data['Value'].append(pyo.value(model.lambda_g[g]))
            df_satisfaction = pd.DataFrame(satisfaction_data)

            # Sheet: optimal_decision_variables
            dec_vars_data = []
            for i in model.I:
                dec_vars_data.append({
                    'Variable': i,
                    'Optimal Value': pyo.value(model.x[i]),
                    'notes': f"Bounds: {params['x_bounds'][i]}"
                })
            df_decision_vars = pd.DataFrame(dec_vars_data)

            # Sheet: optimal_capex_breakdown
            capex_data = [
                {'Component': 'Reactor Volume (V_reactor)', 'Value': v_reactor, 'Unit': 'm^3'},
                {'Component': 'Reactor Purchase Cost (C_purch)', 'Value': c_purch, 'Unit': '$'},
                {'Component': 'Bare-Module Factor (F_BM)', 'Value': f_bm, 'Unit': 'x'},
                {'Component': 'Total CAPEX', 'Value': capex, 'Unit': '$'}
            ]
            df_capex = pd.DataFrame(capex_data)

            # Sheet: optimal_aoc_breakdown
            aoc_data = [
                {'Component': 'Annual Aeration Cost', 'Value': annual_aeration_cost, 'Unit': '$/yr'},
                {'Component': 'Annual Mixing Cost', 'Value': annual_mixing_cost, 'Unit': '$/yr'},
                {'Component': 'Total Electricity Cost', 'Value': total_elec_cost, 'Unit': '$/yr'},
                {'Component': 'Annual Maintenance Cost', 'Value': annual_maint_cost, 'Unit': '$/yr'},
                {'Component': 'Annual Labor Cost', 'Value': annual_labor_cost, 'Unit': '$/yr'},
                {'Component': 'Total AOC', 'Value': aoc, 'Unit': '$/yr'}
            ]
            df_aoc = pd.DataFrame(aoc_data)

            # Sheet: optimal_predicted_effluent_quality
            effluent_data = []
            for k in sorted(model.K):
                val = pyo.value(model.Y[k])
                note = "Not a primary goal"
                if k in params['fuzzy_goals']:
                    goal = params['fuzzy_goals'][k]
                    satisfaction = pyo.value(model.lambda_g[k])
                    note = f"Target: {goal['target']:.1f}, Max: {goal['max']:.1f}, Satisfaction: {satisfaction:.3f}"
                effluent_data.append({
                    'Component': k,
                    'Predicted Value (mg/L)': val,
                    'Notes': note
                })
            df_effluent = pd.DataFrame(effluent_data)
            
            # Sheet: default_influent_quality
            df_influent = pd.DataFrame(
                list(params['defaults'].items()),
                columns=['Variable', 'Value (mg/L)']
            )

            # 2. Write DataFrames to Excel file using append mode to avoid overwriting other sheets
            # 'if_sheet_exists="replace"' will update our specific sheets but leave others untouched.
            # This requires the 'openpyxl' engine.
            mode = 'a' if os.path.exists(output_path) else 'w'
            
            with pd.ExcelWriter(
                output_path,
                engine='openpyxl',
                mode=mode,
                if_sheet_exists='replace' if mode == 'a' else None
            ) as writer:
                df_satisfaction.to_excel(writer, sheet_name='optimal_satistfaction', index=False)
                df_decision_vars.to_excel(writer, sheet_name='optimal_decision_variables', index=False)
                df_capex.to_excel(writer, sheet_name='optimal_capex_breakdown', index=False)
                df_aoc.to_excel(writer, sheet_name='optimal_aoc_breakdown', index=False)
                df_effluent.to_excel(writer, sheet_name='optimal_predicted_effluent', index=False)
                df_influent.to_excel(writer, sheet_name='default_influent_quality', index=False)

            print(f"...Successfully saved results to '{output_path}'.")

        except Exception as e:
            print(f"\nERROR: Failed to save results to Excel file. Reason: {e}", file=sys.stderr)
        # --- [END] ADDED CODE FOR SAVING RESULTS TO EXCEL ---
        
        # --- Verification against Analytical Reconstruction ---
        print("\n" + "="*80)
        print("VERIFICATION: PYOMO vs. ANALYTICAL RECONSTRUCTION".center(80))
        print("="*80)

        m_order = params['M']
        x_s_optimal = np.array([pyo.value(model.X_s[m]) for m in m_order]).reshape(1, -1)
        z_s_optimal = create_interaction_features_np(x_s_optimal, len(m_order))
        y_s_analytical = predict_analytical(x_s_optimal, z_s_optimal, params['coeffs'])
        y_analytical = params['y_scaler'].inverse_transform(y_s_analytical).flatten()
        
        k_order = params['K']
        y_pyomo = np.array([pyo.value(model.Y[k]) for k in k_order])
        
        df_compare = pd.DataFrame({
            'Component': k_order,
            'Pyomo_Prediction': y_pyomo,
            'Analytical_Reconstruction': y_analytical
        })
        df_compare['Difference'] = df_compare['Pyomo_Prediction'] - df_compare['Analytical_Reconstruction']
        df_compare['Abs_Difference'] = df_compare['Difference'].abs()
        
        mae = df_compare['Abs_Difference'].mean()
        
        print("Comparison of effluent predictions from the Pyomo NLP model and the independent analytical formula:")
        print(df_compare.to_string(index=False))
        print("-" * 80)
        print(f"Mean Absolute Error (MAE): {mae:.6g}")
        if mae < 1e-5:
            print("Conclusion: The Pyomo model reconstruction is correct and matches the analytical formula.")
        else:
            print("Conclusion: A significant discrepancy exists. The Pyomo model reconstruction is incorrect.")

    else:
        print("\n---> No optimal solution found. <---")
        print("This could be due to an infeasible problem formulation or solver issues.")
    print("="*80)

def main():
    """
    Main execution function.
    """
    try:
        params = load_data_and_params()
        model = build_nlp_model(params)
        solve_and_display_results(model, params)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    os.environ["PATH"] = os.environ["PATH"] + ";C:\\Users\\eggy\\miniforge3\\Library\\bin"
    main()