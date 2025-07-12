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

    params['I'] = xls['decision_var']['I_variables'].tolist()

    df_bounds = xls['decision_var_bound']
    params['x_bounds'] = {
        row.Variable: (row.LowerBound, row.UpperBound) 
        for _, row in df_bounds.iterrows()
    }

    df_defaults = xls['influent_compound_conc']
    params['defaults'] = pd.Series(df_defaults.Value.values, index=df_defaults.Variable).to_dict()

    try:
        params['df_cost_vars'] = xls['cost_var']
        params['df_capex_calc'] = xls['capex_calc']
    except KeyError as e:
        raise KeyError(f"Missing required cost sheet in '{config_path}': {e}. Please add 'cost_var' and 'capex_calc' worksheets.")

    cost_var_names = set(params['df_cost_vars']['Variable'].dropna())
    decision_var_names = set(params['I'])
    overlap = cost_var_names.intersection(decision_var_names)
    
    if overlap:
        print("\nINFO: The following variables are defined as both decision variables and cost parameters.")
        print("      The decision variable bounds will be used for optimization, and the values in the")
        print("      'cost_var' sheet for these variables will be IGNORED.")
        print(f"      Overlapping variables: {', '.join(sorted(list(overlap)))}\n")

    df_goals = xls['fuzzy_goal']
    
    # --- CRITICAL FIX: Clean the goal names at the point of creation ---
    params['fuzzy_goals'] = {
        row.Goal.split(' (')[0].strip(): {'target': row.Target, 'max': row.Max} 
        for _, row in df_goals.iterrows()
    }
    # --- END CRITICAL FIX ---
    
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
        if goal.startswith('Target_Effluent_'): # Using a more specific prefix
            try:
                # This will now succeed because 'goal' is a cleaned name
                idx = params['K'].index(goal)
                scaled_target = (limits['target'] - y_scaler.mean_[idx]) / y_scaler.scale_[idx]
                scaled_max = (limits['max'] - y_scaler.mean_[idx]) / y_scaler.scale_[idx]
                params['scaled_fuzzy_goals'][goal] = {'target_s': scaled_target, 'max_s': scaled_max}
            except (ValueError, IndexError):
                print(f"Warning: Fuzzy goal '{goal}' not found in model outputs. Skipping.")
                pass

    print("...Data loading complete.")
    return params

def build_cost_model(model, params):
    """
    Dynamically builds the cost calculation part of the model from Excel sheets.
    Gives precedence to decision variables over fixed cost parameters.
    """
    print("2a. Building cost model from 'cost_var' and 'capex_calc' sheets...")
    
    cost_context = {}

    for i in model.I:
        cost_context[i] = model.x[i]

    df_cost_vars = params['df_cost_vars'].dropna(subset=['Variable', 'Value']).set_index('Variable')

    for var_name, row in df_cost_vars.iterrows():
        if var_name in cost_context:
            continue
        param = pyo.Param(initialize=row['Value'], within=pyo.Reals)
        setattr(model, f"param_{var_name}", param)
        cost_context[var_name] = param

    df_capex_calc = params['df_capex_calc'].dropna(subset=['Output Variable', 'Calculation'])
    
    model.cost_calc_order = df_capex_calc['Output Variable'].tolist()
    
    for _, row in df_capex_calc.iterrows():
        var_name = row['Output Variable']
        var = pyo.Var(within=pyo.NonNegativeReals, initialize=0)
        setattr(model, var_name, var)
        cost_context[var_name] = var

    model.CostCalculationRules = pyo.ConstraintList()
    for _, row in df_capex_calc.iterrows():
        lhs_var_name = row['Output Variable']
        calc_string = row['Calculation']
        lhs_var = cost_context[lhs_var_name]
        rhs_expr = eval(calc_string, {"__builtins__": None}, cost_context)
        model.CostCalculationRules.add(lhs_var == rhs_expr)

    print("...Cost model build complete.")
    return model

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
    
    model.lambda_g = pyo.Var(model.G, initialize=0.5)
    model.lambda_o = pyo.Var(initialize=0.5)
    
    model.X = pyo.Var(model.M)
    model.X_s = pyo.Var(model.M)
    model.Y = pyo.Var(model.K, within=pyo.NonNegativeReals)
    model.Y_s = pyo.Var(model.K)
    
    model = build_cost_model(model, params)
    if not hasattr(model, 'CAPEX'): model.CAPEX = pyo.Var(initialize=0)
    if not hasattr(model, 'AOC'): model.AOC = pyo.Var(initialize=0)

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

    model.FuzzyConstraints = pyo.ConstraintList()
    epsilon = 1e-9

    for k in model.K_goals:
        limits = params['scaled_fuzzy_goals'][k]
        x_s = model.Y_s[k]
        x_min_s = limits['target_s']
        x_max_s = limits['max_s']
        denominator_s = x_max_s - x_min_s
        model.FuzzyConstraints.add(
            model.lambda_g[k] == 1 - (x_s - x_min_s) / (denominator_s + epsilon)
        )

    if 'AOC' in model.G:
        aoc_goal = params['fuzzy_goals']['AOC']
        x_aoc = model.AOC
        x_min_aoc = aoc_goal['target']
        x_max_aoc = aoc_goal['max']
        denominator_aoc = x_max_aoc - x_min_aoc
        model.FuzzyConstraints.add(
            model.lambda_g['AOC'] == 1 - (x_aoc - x_min_aoc) / (denominator_aoc + epsilon)
        )

    if 'CAPEX' in model.G:
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
        print("\n---> Optimal Solution Found <---\n")
        lambda_o_val = pyo.value(model.lambda_o)
        print(f"Overall Satisfaction Level (lambda_o): {lambda_o_val:.4f}")

        print("\n--- Individual Goal Satisfaction (lambda_g) ---")
        for g in sorted(model.G):
            print(f"{g:<20}: {pyo.value(model.lambda_g[g]):.4f}")
        
        print("\n--- Optimal Decision Variables ---")
        for i in model.I:
            print(f"{i:<12}: {pyo.value(model.x[i]):10.2f} (Bounds: {params['x_bounds'][i]})")

        df_capex_calc = params['df_capex_calc'].dropna(subset=['Output Variable', 'Calculation'])
        
        capex = pyo.value(model.CAPEX) if hasattr(model, 'CAPEX') else 0
        aoc = pyo.value(model.AOC) if hasattr(model, 'AOC') else 0
        capex_goal = params['fuzzy_goals'].get('CAPEX', {'target': 0, 'max': 1})
        aoc_goal = params['fuzzy_goals'].get('AOC', {'target': 0, 'max': 1})
        capex_satisfaction = pyo.value(model.lambda_g['CAPEX']) if 'CAPEX' in model.G else 'N/A'
        aoc_satisfaction = pyo.value(model.lambda_g['AOC']) if 'AOC' in model.G else 'N/A'

        print("\n--- CAPEX Breakdown ---")
        for var_name in model.cost_calc_order:
             if any(keyword in var_name for keyword in ['CAPEX', 'purch', 'V_reactor', 'C_const', 'C_equip']):
                val = pyo.value(getattr(model, var_name))
                desc = df_capex_calc[df_capex_calc['Output Variable'] == var_name]['Description'].iloc[0]
                print(f"{desc:<35}: ${val:12,.2f}")
        print("-" * 50)
        print(f"{'Total CAPEX':<35}: ${capex:12,.2f} (Target: ${capex_goal['target']:,.0f}, Max: ${capex_goal['max']:,.0f}, Satisfaction: {capex_satisfaction:.3f})")

        print("\n--- AOC Breakdown ---")
        for var_name in model.cost_calc_order:
            if any(keyword in var_name for keyword in ['AOC', 'cost', 'power']):
                 val = pyo.value(getattr(model, var_name))
                 desc = df_capex_calc[df_capex_calc['Output Variable'] == var_name]['Description'].iloc[0]
                 print(f"{desc:<35}: ${val:12,.2f} / yr")
        print("-" * 50)
        print(f"{'Total AOC':<35}: ${aoc:12,.2f} / yr (Target: ${aoc_goal['target']:,.0f}, Max: ${aoc_goal['max']:,.0f}, Satisfaction: {aoc_satisfaction:.3f})")

        print("\n--- All Predicted Effluent Quality (mg/L) ---")
        for k in sorted(model.K):
            val = pyo.value(model.Y[k])
            if k in params['fuzzy_goals']:
                goal = params['fuzzy_goals'][k]
                satisfaction = pyo.value(model.lambda_g[k])
                print(f"{k:<20}: {val:8.3f} (Target: {goal['target']:<4.1f}, Max: {goal['max']:<4.1f}, Satisfaction: {satisfaction:.3f})")
            else:
                print(f"{k:<20}: {val:8.3f} (Not a primary goal)")
        
        print("\n--- Saving results to data/data.xlsx ---")
        try:
            output_folder = 'data'
            output_filename = 'data.xlsx'
            output_path = os.path.join(output_folder, output_filename)
            os.makedirs(output_folder, exist_ok=True)

            satisfaction_data = {'Metric': ['Overall Satisfaction Level (lambda_o)'], 'Value': [pyo.value(model.lambda_o)]}
            for g in sorted(model.G):
                satisfaction_data['Metric'].append(f'Individual Goal Satisfaction ({g})')
                satisfaction_data['Value'].append(pyo.value(model.lambda_g[g]))
            df_satisfaction = pd.DataFrame(satisfaction_data)

            dec_vars_data = [{'Variable': i, 'Optimal Value': pyo.value(model.x[i]), 'notes': f"Bounds: {params['x_bounds'][i]}"} for i in model.I]
            df_decision_vars = pd.DataFrame(dec_vars_data)
            
            capex_data = [{'Component': df_capex_calc[df_capex_calc['Output Variable'] == v]['Description'].iloc[0], 'Value': pyo.value(getattr(model, v)), 'Unit': '$'} for v in model.cost_calc_order if any(k in v for k in ['CAPEX', 'purch', 'C_const', 'C_equip'])]
            df_capex = pd.DataFrame(capex_data)

            aoc_data = [{'Component': df_capex_calc[df_capex_calc['Output Variable'] == v]['Description'].iloc[0], 'Value': pyo.value(getattr(model, v)), 'Unit': '$/yr'} for v in model.cost_calc_order if any(k in v for k in ['AOC', 'cost', 'power'])]
            df_aoc = pd.DataFrame(aoc_data)
            
            effluent_data = []
            for k in sorted(model.K):
                val = pyo.value(model.Y[k])
                note = "Not a primary goal"
                if k in params['fuzzy_goals']:
                    goal = params['fuzzy_goals'][k]
                    satisfaction = pyo.value(model.lambda_g[k])
                    note = f"Target: {goal['target']:.1f}, Max: {goal['max']:.1f}, Satisfaction: {satisfaction:.3f}"
                effluent_data.append({'Component': k, 'Predicted Value (mg/L)': val, 'Notes': note})
            df_effluent = pd.DataFrame(effluent_data)
            
            df_influent = pd.DataFrame(list(params['defaults'].items()), columns=['Variable', 'Value (mg/L)'])

            mode = 'a' if os.path.exists(output_path) else 'w'
            with pd.ExcelWriter(output_path, engine='openpyxl', mode=mode, if_sheet_exists='replace' if mode == 'a' else None) as writer:
                df_satisfaction.to_excel(writer, sheet_name='optimal_satisfaction', index=False)
                df_decision_vars.to_excel(writer, sheet_name='optimal_decision_variables', index=False)
                df_capex.to_excel(writer, sheet_name='optimal_capex_breakdown', index=False)
                df_aoc.to_excel(writer, sheet_name='optimal_aoc_breakdown', index=False)
                df_effluent.to_excel(writer, sheet_name='optimal_predicted_effluent', index=False)
                df_influent.to_excel(writer, sheet_name='default_influent_quality', index=False)
            print(f"...Successfully saved results to '{output_path}'.")
        except Exception as e:
            print(f"\nERROR: Failed to save results to Excel file. Reason: {e}", file=sys.stderr)
        
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
        df_compare = pd.DataFrame({'Component': k_order, 'Pyomo_Prediction': y_pyomo, 'Analytical_Reconstruction': y_analytical})
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