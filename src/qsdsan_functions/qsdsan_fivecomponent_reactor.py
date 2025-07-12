import qsdsan as qs
import qsdsan.processes as pc
import qsdsan.sanunits as su
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# Ignore Pandas future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Starting QSDsan simulation setup...")

# --- 2. System Setup ---

# 2.1. Component
# Create components for ASM2d, which are already compiled.
cmps = pc.create_asm2d_cmps()
# Set the thermodynamic properties for the components in the system.
qs.set_thermo(cmps)
print(f'Successfully loaded and set thermo for components: {cmps.IDs}')

# 2.2. WasteStream
# Define system parameters: flow rates and temperature
Q_inf = 18446  # influent flowrate [m3/d]
Q_was = 385    # sludge wastage flowrate [m3/d]
Q_ext = 18446  # external recycle flowrate [m3/d]
Temp = 273.15 + 20  # temperature [K], equivalent to 20°C

# Create the main WasteStream objects for the system
# These streams will carry mass and energy flows between units
influent = qs.WasteStream('influent', T=Temp)
effluent = qs.WasteStream('effluent', T=Temp)
int_recycle = qs.WasteStream('internal_recycle', T=Temp) # For internal recycle from O3 to A1
ext_recycle = qs.WasteStream('external_recycle', T=Temp) # For external recycle from C1 to A1
wastage = qs.WasteStream('wastage', T=Temp)             # For sludge wastage from C1

# Define the default influent composition based on concentrations
default_inf_kwargs = {
    'concentrations': {
      'S_I': 14, 'X_I': 26.5, 'S_F': 20.1, 'S_A': 94.3, 'X_S': 409.75,
      'S_NH4': 31, 'S_N2': 0, 'S_NO3': 0.266, 'S_PO4': 2.8, 'X_PP': 0.05,
      'X_PHA': 0.5, 'X_H': 0.15, 'X_AUT': 0, 'X_PAO': 0, 'S_ALK': 7*12,
    },
    'units': ('m3/d', 'mg/L'), # Specify units for flowrate and concentrations
}
# Set the influent flowrate and composition
influent.set_flow_by_concentration(Q_inf, **default_inf_kwargs)
print("\nInfluent Stream details:")
influent.show()
print(f"Influent VSS: {influent.get_VSS():.2f} mg/L")

# 2.3. Process
# Define volumes for anoxic and aerated zones
V_an = 1000  # anoxic zone tank volume [m3]
V_ae = 1333  # aerated zone tank volume [m3]

# Create aeration models using DiffusedAeration process
# aer1 for Tank 3 & Tank 4 (O1, O2), aer2 for Tank 5 (O3)
aer1 = pc.DiffusedAeration('aer1', DO_ID='S_O2', KLa=240, DOsat=8.0, V=V_ae)
aer2 = pc.DiffusedAeration('aer2', DO_ID='S_O2', KLa=84, DOsat=8.0, V=V_ae)
print("\nAeration Model 1 details:")
aer1.show()

# Create the main ASM2d biokinetic process model
asm2d = pc.ASM2d()
print("\nASM2d Process Model details:")
asm2d.show()

# 2.4. SanUnit
# Define the unit operations that form the activated sludge reactor system.

# Anoxic reactors (Tank 1 & Tank 2) - CSTRs with no aeration
# A1: Takes influent, internal_recycle, and external_recycle
A1 = su.CSTR('A1', ins=[influent, int_recycle, ext_recycle], V_max=V_an,
             aeration=None, suspended_growth_model=asm2d)
# A2: Takes effluent from A1 (A1-0 notation connects A1's first outlet to A2's inlet)
A2 = su.CSTR('A2', ins=A1-0, V_max=V_an,
             aeration=None, suspended_growth_model=asm2d)

# Aerated reactors (Tank 3, Tank 4, Tank 5) - CSTRs with specified aeration models
O1 = su.CSTR('O1', ins=A2-0, V_max=V_ae, aeration=aer1,
             DO_ID='S_O2', suspended_growth_model=asm2d)
O2 = su.CSTR('O2', ins=O1-0, V_max=V_ae, aeration=aer1,
             DO_ID='S_O2', suspended_growth_model=asm2d)
# O3: Splits its effluent into an internal recycle and a treated stream going to the clarifier
O3 = su.CSTR('O3', ins=O2-0, outs=[int_recycle, 'treated'], split=[0.6, 0.4],
             V_max=V_ae, aeration=aer2,
             DO_ID='S_O2', suspended_growth_model=asm2d)

# Clarifier (FlatBottomCircularClarifier) for solids-liquid separation
# C1: Takes the treated stream from O3 (O3-1) and produces effluent, external recycle, and wastage
C1 = su.FlatBottomCircularClarifier('C1', ins=O3-1, outs=[effluent, ext_recycle, wastage],
                                    underflow=Q_ext, wastage=Q_was, surface_area=1500,
                                    height=4, N_layer=10, feed_layer=5,
                                    X_threshold=3000, v_max=474, v_max_practical=250,
                                    rh=5.76e-4, rp=2.86e-3, fns=2.28e-3)

# 2.5. System
# Create the overall system by defining the path of units and recycle streams.
sys = qs.System('example_system', path=(A1, A2, O1, O2, O3, C1), recycle=(int_recycle, ext_recycle))
print("\nSystem created. Attempting to display system diagram (requires Graphviz).")
try:
    sys.diagram()
    print("System diagram generated. Check output/image viewer for 'example_system.png'.")
except Exception as e:
    print(f"Could not generate system diagram: {e}. Ensure Graphviz is installed and configured on your system PATH.")

# 2.5.2. Set initial conditions of reactors
# The original tutorial uses an Excel file. For a self-contained executable code,
# the data is embedded directly into a Pandas DataFrame.
initial_conditions_data = {
    'S_O2':    {'A1': 0.00213, 'A2': 0.001, 'O2': 2, 'O3': 2, 'O1': 2, 'C1_s': 2, 'C1_x': np.nan, 'C1_tss': 17.8},
    'S_NH4':   {'A1': 7.23, 'A2': 22.4, 'O2': 16.5, 'O3': 10.9, 'O1': 0.111, 'C1_s': 0.114, 'C1_x': np.nan, 'C1_tss': 27.9},
    'S_NO3':   {'A1': 10.2, 'A2': 2.4, 'O2': 4.31, 'O3': 9.31, 'O1': 26.1, 'C1_s': 20.9, 'C1_x': np.nan, 'C1_tss': 44.9},
    'S_PO4':   {'A1': 4.45, 'A2': 4.24, 'O2': 5.48, 'O3': 2.62, 'O1': 2.32, 'C1_s': 0.356, 'C1_x': np.nan, 'C1_tss': 90.5},
    'S_F':     {'A1': 0.211, 'A2': 6.68, 'O2': 1.9, 'O3': 0.649, 'O1': 0.276, 'C1_s': 0.307, 'C1_x': np.nan, 'C1_tss': 305},
    'S_A':     {'A1': 0.0265, 'A2': 53.8, 'O2': 2.73, 'O3': 0.163, 'O1': 0.00407, 'C1_s': 0.00537, 'C1_x': np.nan, 'C1_tss': 304},
    'S_I':     {'A1': 15.9, 'A2': 14.5, 'O2': 13.7, 'O3': 14.1, 'O1': 18.2, 'C1_s': 20.1, 'C1_x': np.nan, 'C1_tss': 306},
    'S_ALK':   {'A1': 67, 'A2': 79, 'O2': 82.6, 'O3': 74.2, 'O1': 46.1, 'C1_s': 49.6, 'C1_x': np.nan, 'C1_tss': 304},
    'X_I':     {'A1': 2.28e+03, 'A2': 0, 'O2': 611, 'O3': 662, 'O1': 2.24e+03, 'C1_s': np.nan, 'C1_x': 2.24e+03, 'C1_tss': 304},
    'X_S':     {'A1': 84.4, 'A2': 84.1, 'O2': 77.3, 'O3': 59.3, 'O1': 61.1, 'C1_s': np.nan, 'C1_x': 61.1, 'C1_tss': 5.83e+03},
    'X_H':     {'A1': 3.78e+03, 'A2': 207, 'O2': 1.04e+03, 'O3': 1.14e+03, 'O1': 3.79e+03, 'C1_s': np.nan, 'C1_x': 3.79e+03, 'C1_tss': np.nan},
    'X_PAO':   {'A1': 322, 'A2': 18.2, 'O2': 86.4, 'O3': 95.7, 'O1': 322, 'C1_s': np.nan, 'C1_x': 322, 'C1_tss': np.nan},
    'X_PP':    {'A1': 37.2, 'A2': 4.25, 'O2': 6.45, 'O3': 9.99, 'O1': 38.4, 'C1_s': np.nan, 'C1_x': 38.4, 'C1_tss': np.nan},
    'X_PHA':   {'A1': 0.0517, 'A2': 3.59, 'O2': 11, 'O3': 7.24, 'O1': 0.00852, 'C1_s': np.nan, 'C1_x': 0.00852, 'C1_tss': np.nan},
    'X_AUT':   {'A1': 218, 'A2': 11.9, 'O2': 58, 'O3': 64, 'O1': 218, 'C1_s': np.nan, 'C1_x': 218, 'C1_tss': np.nan},
}
# Create DataFrame, transpose, and adjust column names/index to match tutorial's usage
df_init_cond = pd.DataFrame(initial_conditions_data).T
df_init_cond.columns = ['A1', 'A2', 'O2', 'O3', 'O1', 'C1_s', 'C1_x', 'C1_tss']
df_init_cond = df_init_cond.reset_index().rename(columns={'index': 'ID'}).set_index('ID').T

# Define the function to set initial conditions for all reactors.
def batch_init(system_obj, dataframe_with_init_cond):
    dct = dataframe_with_init_cond.to_dict('index')
    u = system_obj.flowsheet.unit # Access units from the system's flowsheet

    # Set initial concentrations for CSTRs (A1, A2, O1, O2, O3)
    for unit_id in ['A1', 'A2', 'O1', 'O2', 'O3']:
        unit = getattr(u, unit_id) # Get unit object by ID
        # Filter out NaN values from the dictionary before passing to set_init_conc
        valid_concs = {comp: conc for comp, conc in dct[unit_id].items() if not pd.isna(conc)}
        unit.set_init_conc(**valid_concs)

    # Set initial concentrations for the clarifier (C1)
    # Separate soluble, solids, and TSS for C1 as per its specific methods
    c1s = {k: v for k, v in dct['C1_s'].items() if not pd.isna(v) and v > 0}
    c1x = {k: v for k, v in dct['C1_x'].items() if not pd.isna(v) and v > 0}
    tss = [v for v in dct['C1_tss'].values() if not pd.isna(v) and v > 0]
    
    u.C1.set_init_solubles(**c1s)
    u.C1.set_init_sludge_solids(**c1x)
    u.C1.set_init_TSS(tss)

# Apply initial conditions to the system
batch_init(sys, df_init_cond)
print("\nInitial conditions have been set for all reactors based on the embedded data.")

# --- 3. System Simulation ---
# Set which streams/units to track dynamic changes during simulation
sys.set_dynamic_tracker(influent, effluent, A1, A2, O1, O2, O3, C1, wastage)
# Set convergence tolerance for the simulation
sys.set_tolerance(rmol=1e-6)

# Define biomass IDs for SRT calculation
biomass_IDs = ('X_H', 'X_PAO', 'X_AUT')

# Simulation settings
t = 50          # total simulation time [days]
t_step = 1      # time step for storing computed solution [days]
method = 'BDF'  # ODE integration method, 'BDF' is a good choice for stiff systems like this

print(f"\nRunning dynamic simulation for {t} days using '{method}' method...")
# Execute the dynamic simulation
sys.simulate(state_reset_hook='reset_cache', # Resets cached states before each simulation run
             t_span=(0, t),                  # Time span for the simulation (start, end)
             t_eval=np.arange(0, t + t_step, t_step), # Specific time points to evaluate and store results
             method=method)

# Calculate and print the estimated Sludge Retention Time (SRT)
# This uses a utility function from qsdsan.utils
try:
    srt = qs.utils.get_SRT(sys, biomass_IDs)
    print(f'Estimated SRT assuming at steady state is {round(srt, 2)} days')
except Exception as e:
    print(f"Could not calculate SRT: {e}. Ensure all units are correctly configured for mass balance and 'get_retained_mass()'.")

print("\nSystem state after simulation:")
sys.show()

# 3.2. Check simulation results
print("\nPlotting influent time series (expected to be constant)...")
try:
    influent.scope.plot_time_series(('S_I','X_I','S_F','S_A','X_S','S_NH4','S_N2','S_NO3','S_PO4','X_PP','X_PHA',
                                     'X_H','X_AUT','X_PAO','S_ALK'))
    plt.suptitle("Influent Composition Over Time (All Components)")
    plt.xlabel("Time [d]")
    plt.ylabel("Concentration [mg/L]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error plotting influent time series: {e}")

print("\nPlotting effluent time series (all components)...")
try:
    effluent.scope.plot_time_series((
        'S_I','X_I','S_F','S_A','X_S','S_NH4','S_N2','S_NO3','S_PO4','X_PP','X_PHA',
        'X_H','X_AUT','X_PAO','S_ALK'
    ))
    plt.suptitle("Effluent Composition Over Time (All Components)")
    plt.xlabel("Time [d]")
    plt.ylabel("Concentration [mg/L]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error plotting effluent (all) time series: {e}")

print("\nPlotting effluent S_NH4 and S_NO3 time series...")
try:
    effluent.scope.plot_time_series(('S_NH4', 'S_NO3'))
    plt.suptitle("Effluent NH4 and NO3 Concentrations Over Time")
    plt.xlabel("Time [d]")
    plt.ylabel("Concentration [mg/L]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error plotting effluent NH4/NO3 time series: {e}")

print("\nPlotting effluent S_O2 time series...")
try:
    effluent.scope.plot_time_series(('S_O2'))
    plt.suptitle("Effluent Dissolved Oxygen (S_O2) Over Time")
    plt.xlabel("Time [d]")
    plt.ylabel("Concentration [mg/L]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error plotting effluent O2 time series: {e}")

print("\nSimulation and plotting complete.")