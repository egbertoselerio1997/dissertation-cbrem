import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import traceback
import warnings
import itertools

# --- Model Library Class Definitions (Required for loading joblib files) ---
# These classes must be defined so that 'joblib.load' can unpickle the model objects.

# Suppress library warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, module='openpyxl')
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.base import BaseEstimator, RegressorMixin
    import lightgbm as lgb
    import xgboost as xgb

    # --- Placeholders for unpickling models ---
    class ANNModel(nn.Module):
        """A placeholder for the trained ANN model structure."""
        def __init__(self, n_indep, n_dep):
            super().__init__()
            self.n_dep = n_dep
            self.layers = nn.Sequential(
                nn.Linear(n_indep, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_dep)
            )
        def forward(self, X):
            return self.layers(X)

    # --- Full CLEFO model definition, required for unpickling ---
    class CoupledCLEFOModel(nn.Module):
        """ Implements the coupled linear equations framework using PyTorch. """
        def __init__(self, n_dep, n_indep, n_inter):
            super().__init__()
            self.n_dep = n_dep
            self.Upsilon = nn.Parameter(torch.randn(n_dep, 1) * 0.01)
            self.B = nn.Parameter(torch.randn(n_dep, n_indep) * 0.01)
            self.Theta = nn.Parameter(torch.randn(n_dep, n_inter) * 0.01)
            self.Gamma = nn.Parameter(torch.randn(n_dep, n_dep) * 0.01)
            self.Lambda = nn.Parameter(torch.randn(n_dep, n_indep) * 0.01)
            self.Gamma.register_hook(lambda grad: grad.fill_diagonal_(0))
            self.register_buffer('identity', torch.eye(n_dep), persistent=False)

        def forward(self, X, Z):
            batch_size = X.shape[0]
            upsilon_expanded = self.Upsilon.expand(-1, batch_size)
            bx = self.B @ X.T
            theta_z = self.Theta @ Z.T
            RHS = upsilon_expanded + bx + theta_z
            lambda_x = self.Lambda @ X.T
            diag_lambda_x = torch.diag_embed(lambda_x.T)
            gamma_expanded = self.Gamma.unsqueeze(0).expand(batch_size, -1, -1)
            identity_expanded = self.identity.unsqueeze(0).expand(batch_size, -1, -1)
            LHS_matrix = identity_expanded - gamma_expanded - diag_lambda_x
            try:
                reg = torch.eye(self.n_dep, device=LHS_matrix.device).unsqueeze(0) * 1e-7
                Y_pred_solved = torch.linalg.solve(LHS_matrix + reg, RHS.T.unsqueeze(-1))
                Y_pred = Y_pred_solved.squeeze(-1)
            except torch.linalg.LinAlgError:
                Y_pred = torch.full((batch_size, self.n_dep), float('nan'), device=X.device)
            return Y_pred

    # Required for unpickling the scikit-learn-based model bundles
    class RandomForestRegressor(RandomForestRegressor): pass
    class KNeighborsRegressor(KNeighborsRegressor): pass
    class LinearRegression(LinearRegression): pass
    class SVR(SVR): pass
    class LGBMRegressor(lgb.LGBMRegressor): pass
    class XGBRegressor(xgb.XGBRegressor): pass

    # --- Placeholders for custom GAM model ---
    class SplineBasis:
        def __init__(self, n_knots=10):
            self.n_knots = n_knots
            self.knots_ = None
        def fit(self, x): return self
        def transform(self, x): return x

    class GAMRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, n_knots=20, max_iter=100, tol=1e-5):
            self.n_knots, self.max_iter, self.tol = n_knots, max_iter, tol
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(X.shape[0])

except ImportError:
    print("Warning: A required library (PyTorch, Scikit-learn, etc.) was not found. The script may fail to load models.", file=sys.stderr)
    # Define dummy classes if libraries are missing to avoid immediate import errors
    class ANNModel: pass
    class CoupledCLEFOModel: pass
    class StandardScaler: pass
    class MultiOutputRegressor: pass
    class RandomForestRegressor: pass
    class KNeighborsRegressor: pass
    class LinearRegression: pass
    class SVR: pass
    class BaseEstimator: pass
    class RegressorMixin: pass
    class SplineBasis: pass
    class GAMRegressor: pass
    class lgb:
        class LGBMRegressor: pass
    class xgb:
        class XGBRegressor: pass

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
from naming import CONCENTRATION_SUFFIX, normalize_stream_column, rename_concentration_columns, strip_prefix_and_units


class SurrogateModelPredictor:
    """
    Uses various pre-trained surrogate models to predict WWTP performance based 
    on optimal decision variables from an optimization run.
    """

    def __init__(self, config_path: str, results_path: str):
        """Initializes the predictor by loading all required models and data."""
        print("--- Initializing Surrogate Model Predictor ---")
        self.config_path = config_path
        self.results_path = results_path

        self.full_flow_units = ['A1', 'A2', 'O1', 'O2', 'O3', 'C1']
        self.full_flow_cstr_units = ['A1', 'A2', 'O1', 'O2', 'O3']
        self.full_flow_clarifier_units = ['C1']
        
        self.available_models = ['ann', 'lightgbm', 'xgboost', 'random_forest', 'knn', 'gam', 'glm', 'svr', 'clefo']
        self.models = {model_type: {} for model_type in self.available_models}
        self.feature_names = {'cstr': {}, 'clarifier': {}}

        self._load_data_and_models()

    @staticmethod
    def create_interaction_features_np(X: np.ndarray, m: int) -> np.ndarray:
        """Generates pairwise interaction features (Z) from independent variables (X) using NumPy."""
        col_pairs = list(itertools.combinations(range(m), 2))
        q = len(col_pairs)
        N = X.shape[0]
        Z = np.zeros((N, q), dtype=np.float64)
        for i, (col1_idx, col2_idx) in enumerate(col_pairs):
            Z[:, i] = X[:, col1_idx] * X[:, col2_idx]
        return Z

    def _unify_comp_name(self, name: str) -> str:
        """Return the descriptive base compound name without stream or unit tags."""
        return strip_prefix_and_units(name)

    def _load_data_and_models(self):
        """Loads surrogate models, scalers, and data from optimization outputs."""
        print("1. Loading required files...")
        initial_model_list = self.available_models.copy()

        for model_type in initial_model_list:
            print(f"   - Loading {model_type.upper()} models...")
            is_valid_model = True
            for unit_type in ['cstr', 'clarifier']:
                model_path = os.path.join('data', 'results', 'training', model_type, unit_type, f'{unit_type}.joblib')
                if not os.path.exists(model_path):
                    print(f"     - WARNING: Model file not found for '{model_type}/{unit_type}' at '{model_path}'. Skipping this model.")
                    is_valid_model = False
                    break
                try:
                    model_bundle = joblib.load(model_path)
                    x_features = []
                    y_features = []
                    if hasattr(model_bundle.get('x_scaler'), 'feature_names_in_'):
                        raw_x = list(model_bundle['x_scaler'].feature_names_in_)
                        x_features = [normalize_stream_column(name) for name in raw_x]
                        model_bundle['x_scaler'].feature_names_in_ = np.array(x_features)
                    if hasattr(model_bundle.get('y_scaler'), 'feature_names_in_'):
                        raw_y = list(model_bundle['y_scaler'].feature_names_in_)
                        y_features = [normalize_stream_column(name) for name in raw_y]
                        model_bundle['y_scaler'].feature_names_in_ = np.array(y_features)

                    self.models[model_type][unit_type] = model_bundle
                    print(f"     - Loaded '{unit_type}' model from '{model_path}'")
                    if not self.feature_names[unit_type].get('x') and x_features:
                        self.feature_names[unit_type]['x'] = x_features
                        print(f"     - Captured X feature names for '{unit_type}' from '{model_type}' model.")
                    if not self.feature_names[unit_type].get('y') and y_features:
                        self.feature_names[unit_type]['y'] = y_features
                        print(f"     - Captured Y feature names for '{unit_type}' from '{model_type}' model.")
                except Exception as e:
                    print(f"     - ERROR: Failed to load model file '{model_path}'. Reason: {e}. Skipping this model.")
                    is_valid_model = False
                    break
            
            if not is_valid_model:
                self.models.pop(model_type, None)
        
        self.available_models = [m for m in self.available_models if m in self.models]

        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Optimization results file not found at '{self.results_path}'")
        self.df_dvars = pd.read_excel(self.results_path, sheet_name='optimal_decision_variables')

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at '{self.config_path}'")
        xls_config = pd.read_excel(self.config_path, sheet_name=None)
        df_influent = xls_config['raw_influent_compound_conc']
        raw_influent = pd.Series(df_influent.Value.values, index=df_influent.Variable).to_dict()
        self.influent_params = {}
        for key, value in raw_influent.items():
            prefixed = key if key.startswith(('inf_', 'influent_')) else f"inf_{key}"
            normalized = normalize_stream_column(prefixed)
            self.influent_params[normalized] = value
        
        df_dvar_config = xls_config['decision_var']
        self.cstr_dvars_list = df_dvar_config[df_dvar_config['Process Unit'] == 'cstr']['I_variables'].tolist()
        self.clarifier_dvars_list = df_dvar_config[df_dvar_config['Process Unit'] == 'clarifier']['I_variables'].tolist()

        print("   - Loaded optimization results and influent configuration.")

    def _get_dvar_value(self, variable_name: str, unit: str = 'plant-wide') -> float:
        """Retrieves a specific decision variable's optimal value."""
        val = self.df_dvars[(self.df_dvars['Variable'] == variable_name) & (self.df_dvars['Process Unit'] == unit)]['Optimal Value']
        if val.empty:
            raise ValueError(f"Could not find DVar '{variable_name}' for unit '{unit}' in results file.")
        return val.iloc[0]

    def _predict_with_model(self, model_type: str, unit_type: str, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generic prediction function that handles different model types.
        It scales inputs, runs prediction, un-scales outputs, and reverses log-transformation.
        """
        model_bundle = self.models[model_type][unit_type]
        model, x_scaler, y_scaler = model_bundle['model'], model_bundle['x_scaler'], model_bundle['y_scaler']

        x_features, y_features = self.feature_names[unit_type]['x'], self.feature_names[unit_type]['y']
        
        X_df_ordered = X_df[x_features]
        
        # --- Sanitization Step ---
        # Replace NaNs with 0 and Infs with a safe large number or 0 to prevent scaler crashes
        X_values = X_df_ordered.values
        if not np.all(np.isfinite(X_values)):
            X_values = np.nan_to_num(X_values, nan=0.0, posinf=1e6, neginf=-1e6)
            
        X_scaled = x_scaler.transform(X_values)

        if model_type in ['ann', 'clefo']:
            model.eval()
            # --- Device Handling Fix ---
            # Detect the device the model is on (CPU or CUDA)
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            
            with torch.no_grad():
                # Move input tensor to the same device as the model
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                
                if model_type == 'clefo':
                    m = X_scaled.shape[1]
                    Z_scaled = self.create_interaction_features_np(X_scaled, m)
                    Z_tensor = torch.tensor(Z_scaled, dtype=torch.float32).to(device)
                    Y_pred_scaled_tensor = model(X_tensor, Z_tensor)
                else: # 'ann'
                    Y_pred_scaled_tensor = model(X_tensor)
                
                # Move result back to CPU for numpy conversion
                Y_pred_scaled = Y_pred_scaled_tensor.cpu().numpy()
        elif model_type in ['pls', 'lightgbm', 'xgboost', 'random_forest', 'knn', 'gam']:
            Y_pred_scaled = model.predict(X_scaled)
        else:
            raise NotImplementedError(f"Prediction logic for model type '{model_type}' is not implemented.")

        Y_pred_log = y_scaler.inverse_transform(Y_pred_scaled)

        # --- Overflow Prevention ---
        # Clip values to avoid overflow in exp() (exp(709) is approx max float64)
        Y_pred_log = np.clip(Y_pred_log, -700, 700)

        inverse_transform_func = model_bundle.get('inverse_transform_func', 'exp')
        Y_pred_final = np.expm1(Y_pred_log) if inverse_transform_func == 'expm1' else np.exp(Y_pred_log)

        pred_df = pd.DataFrame(Y_pred_final, columns=y_features)
        pred_df = rename_concentration_columns(pred_df)
        return pred_df

    def run_predictions(self, optimization_target: str):
        """
        Orchestrates the prediction process for each available model type,
        gracefully skipping any model that fails.
        """
        print(f"\n2. Running predictions for '{optimization_target.upper()}' scenario...")
        self.optimization_target = optimization_target
        all_predictions = {}
        
        for model_type in self.available_models:
            try:
                print(f"--- Using {model_type.upper()} models ---")
                if optimization_target == 'cstr':
                    results = self._predict_isolated_cstr(model_type)
                elif optimization_target == 'clarifier':
                    results = self._predict_as_plant(model_type)
                elif optimization_target == 'aao':
                    results = self._predict_aao_plant(model_type)
                else:
                    raise ValueError(f"Unknown optimization target: {optimization_target}")
                all_predictions[model_type] = results
                print(f"--- Successfully completed predictions for {model_type.upper()} ---")
            except Exception as e:
                print(f"\nERROR: Prediction failed for model '{model_type.upper()}'. Skipping this model.", file=sys.stderr)
                print(f"REASON: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print("-" * 50, file=sys.stderr)
                continue
        
        print("\n...All prediction runs complete.")
        return all_predictions

    def _assemble_input_df(self, model_type: str, unit_type: str, influent_source: dict, extra_vars: dict = None, current_unit: str = None) -> pd.DataFrame:
        """
        Assembles the input DataFrame for a model based on its required features.
        """
        if extra_vars is None: extra_vars = {}
        scaler_features = self.feature_names[unit_type]['x']
        input_data = {feat: 0.0 for feat in scaler_features}
        for feature in scaler_features:
            if feature in extra_vars:
                input_data[feature] = extra_vars[feature]
            elif feature in influent_source:
                input_data[feature] = influent_source[feature]
            else:
                try:
                    if current_unit and (feature in self.cstr_dvars_list or feature in self.clarifier_dvars_list):
                        input_data[feature] = self._get_dvar_value(feature, unit=current_unit)
                    else:
                        input_data[feature] = self._get_dvar_value(feature, unit='plant-wide')
                except ValueError:
                    if self.optimization_target == 'cstr' and feature in ['Q_was', 'Q_ext', 'Q_int']:
                         input_data[feature] = 0.0
                    else:
                        raise ValueError(f"Could not find a value for required input feature: '{feature}' (context unit: {current_unit})")
        
        for k, v in input_data.items():
            if not np.isfinite(v):
                input_data[k] = 0.0

        return pd.DataFrame([input_data])[scaler_features]

    def _predict_isolated_cstr(self, model_type: str) -> pd.DataFrame:
        """Handles the single CSTR unit scenario for a given model type."""
        X_df = self._assemble_input_df(model_type, 'cstr', self.influent_params, current_unit='CSTR1')
        predictions = self._predict_with_model(model_type, 'cstr', X_df)
        output_data = [{'Component': f"{self._unify_comp_name(col)}_Effluent", 'Predicted Value (mg/L)': predictions[col].iloc[0]} for col in predictions.columns]
        return pd.DataFrame(output_data)

    def _predict_as_plant(self, model_type: str) -> pd.DataFrame:
        """Handles the CSTR + Clarifier (AS) scenario for a given model type."""
        q_raw_inf, q_ext = self._get_dvar_value('Q_raw_inf'), self._get_dvar_value('Q_ext')
        split_unit, split_var_name = 'CSTR1', 'CSTR1_split_internal'
        split_ratio = self._get_dvar_value(split_var_name, unit=split_unit)
        
        # Safety check for division by zero
        denom = 1 - split_ratio
        if abs(denom) < 1e-6: denom = 1e-6
        q_int = (split_ratio * (q_raw_inf + q_ext)) / denom
        
        extra_vars = {'Q_int': q_int}
        X_cstr = self._assemble_input_df(model_type, 'cstr', self.influent_params, extra_vars=extra_vars, current_unit='CSTR1')
        cstr_predictions = self._predict_with_model(model_type, 'cstr', X_cstr)
        clarifier_influent = {f"influent_{self._unify_comp_name(c)}{CONCENTRATION_SUFFIX}": v for c, v in cstr_predictions.iloc[0].to_dict().items()}
        X_clarifier = self._assemble_input_df(model_type, 'clarifier', clarifier_influent, extra_vars=extra_vars, current_unit='C1')
        clarifier_predictions = self._predict_with_model(model_type, 'clarifier', X_clarifier)
        output_data = [{'Component': f"{self._unify_comp_name(c)}_CSTR1", 'Predicted Value (mg/L)': v} for c, v in cstr_predictions.iloc[0].to_dict().items()]
        for col, val in clarifier_predictions.iloc[0].to_dict().items():
            stream = 'Effluent' if col.startswith('effluent_') else 'Wastage'
            output_data.append({'Component': f"{self._unify_comp_name(col)}_{stream}",'Predicted Value (mg/L)': val})
        return pd.DataFrame(output_data)

    def _predict_aao_plant(self, model_type: str) -> pd.DataFrame:
        """Handles the full AAO plant scenario by sequential prediction for a given model type."""
        stream_concs = {strip_prefix_and_units(r['Component']): r['Predicted Value (mg/L)'] for _, r in pd.read_excel(self.results_path, sheet_name='optimal_predicted_effluent').iterrows()}
        q_raw_inf, q_ext = self._get_dvar_value('Q_raw_inf'), self._get_dvar_value('Q_ext')
        split_unit, split_var_name = 'O3', 'O3_split_internal'
        split_ratio = self._get_dvar_value(split_var_name, unit=split_unit)
        
        # Safety check for division by zero
        denom = 1 - split_ratio
        if abs(denom) < 1e-6: denom = 1e-6
        q_int = (split_ratio * (q_raw_inf + q_ext)) / denom
        
        extra_vars = {'Q_int': q_int}
        unit_predictions = {}
        for unit in self.full_flow_cstr_units:
            if unit == 'A1':
                q_total = q_raw_inf + q_int + q_ext
                cstr_x_features = self.feature_names['cstr']['x']
                current_influent = {feat: (self.influent_params.get(feat, 0) * q_raw_inf + stream_concs.get(f"{self._unify_comp_name(feat)}_O3", 0) * q_int + stream_concs.get(f"{self._unify_comp_name(feat)}_Wastage", 0) * q_ext) / q_total if q_total > 0 else 0
                                  for feat in cstr_x_features if feat.startswith('influent_')}
            else:
                prev_unit = self.full_flow_cstr_units[self.full_flow_cstr_units.index(unit) - 1]
                current_influent = {f"influent_{self._unify_comp_name(c)}{CONCENTRATION_SUFFIX}": v for c, v in unit_predictions[prev_unit].iloc[0].to_dict().items()}
            X_unit = self._assemble_input_df(model_type, 'cstr', current_influent, extra_vars=extra_vars, current_unit=unit)
            unit_predictions[unit] = self._predict_with_model(model_type, 'cstr', X_unit)
        clarifier_influent = {f"influent_{self._unify_comp_name(c)}{CONCENTRATION_SUFFIX}": v for c, v in unit_predictions['O3'].iloc[0].to_dict().items()}
        X_clarifier = self._assemble_input_df(model_type, 'clarifier', clarifier_influent, extra_vars=extra_vars, current_unit='C1')
        unit_predictions['C1'] = self._predict_with_model(model_type, 'clarifier', X_clarifier)
        output_data = []
        for unit, preds in unit_predictions.items():
            if unit != 'C1':
                output_data.extend([{'Component': f"{self._unify_comp_name(c)}_{unit}", 'Predicted Value (mg/L)': v} for c, v in preds.iloc[0].to_dict().items()])
            else:
                for col, val in preds.iloc[0].to_dict().items():
                    output_data.append({'Component': f"{self._unify_comp_name(col)}_{'Effluent' if 'Effluent' in col else 'Wastage'}", 'Predicted Value (mg/L)': val})
        return pd.DataFrame(output_data)

    def save_results(self, predictions_dict: dict, output_path: str):
        """Saves the prediction DataFrames to a specified Excel file, each in a new sheet."""
        if not predictions_dict:
            print("\nNo successful predictions to save.")
            return
        print(f"\n3. Saving successful predictions...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            mode = 'a' if os.path.exists(output_path) else 'w'
            engine_kwargs = {'if_sheet_exists': 'replace'} if mode == 'a' else {}
            with pd.ExcelWriter(output_path, mode=mode, engine='openpyxl', **engine_kwargs) as writer:
                for model_type, predictions_df in predictions_dict.items():
                    sheet_name = model_type
                    predictions_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"   - Saved {model_type.upper()} predictions to sheet '{sheet_name}'.")
            print(f"   - All successful predictions saved to '{output_path}'.")
        except Exception as e:
            print(f"ERROR: Failed to save results to Excel. Reason: {e}", file=sys.stderr)
            traceback.print_exc()

if __name__ == "__main__":
    try:
        print("\nWhich optimization scenario's results should be used for prediction?")
        print("1. cstr (Single CSTR unit)")
        print("2. clarifier (Single CSTR + Clarifier Plant)")
        print("3. aao (Full AAO plant)")
        choice_map = {'1': 'cstr', '2': 'clarifier', '3': 'aao'}
        user_choice = input("Enter your choice (1, 2, or 3): ").strip()
        if user_choice not in choice_map:
            print(f"Invalid choice '{user_choice}'. Please run again.", file=sys.stderr)
            sys.exit(1)
        target = choice_map[user_choice]
        CONFIG_FILE = os.path.join('data', 'config', 'optimization_config.xlsx')
        RESULTS_FILE = os.path.join('data', 'results', 'optimization', target, 'optimization_results.xlsx')
        PREDICTION_OUTPUT_FILE = os.path.join('data', 'results', 'analysis', 'generate_predictions', target, 'surrogate_model_predictions.xlsx')
        predictor = SurrogateModelPredictor(config_path=CONFIG_FILE, results_path=RESULTS_FILE)
        final_predictions_dict = predictor.run_predictions(optimization_target=target)
        predictor.save_results(final_predictions_dict, output_path=PREDICTION_OUTPUT_FILE)
        print("\n" + "="*80)
        print("PROCESS COMPLETE".center(80))
        print(f"All model predictions are in '{PREDICTION_OUTPUT_FILE}'".center(80))
        print("="*80)
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: A required file was not found. Details: {e}", file=sys.stderr)
        print("Please ensure you have run the optimization and all model training scripts first.", file=sys.stderr)
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}", file=sys.stderr)
        traceback.print_exc()
