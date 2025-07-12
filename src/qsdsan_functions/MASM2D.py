import qsdsan as qs
import qsdsan.processes as pc
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Set up components
print("--- Step 1: Setting up components ---")
cmps = pc.create_masm2d_cmps()
qs.set_thermo(cmps)
print("Components for mASM2d loaded and set.")


# Step 2: Define the biokinetic model
print("\n--- Step 2: Defining the mASM2d model ---")
asm = pc.mASM2d()
print("mASM2d model instance created.")


# Step 3: Define influent and reactor
print("\n--- Step 3: Defining influent and CSTR unit ---")
influent = qs.WasteStream('influent_ww', T=293.15)
influent_concentrations = {
    'S_F': 30.0, 'S_A': 20.0, 'S_I': 30.0,
    'S_NH4': 35.0, 'S_PO4': 4.0,
    # 'S_ALK': 7 * 12, # CORRECTED: Removed S_ALK as it's not a state variable in mASM2d
    'X_S': 100.0, 'X_I': 50.0, 'X_H': 25.0, 'X_PAO': 5.0, 'X_AUT': 5.0
}
influent.set_flow_by_concentration(
    flow_tot=18446,
    concentrations=influent_concentrations,
    units=('m3/d', 'mg/L')
)
R1 = qs.sanunits.CSTR('R1',
                      ins=influent,
                      outs=['effluent', 'waste_sludge'],
                      V_max=1500,
                      aeration=2.0,
                      DO_ID='S_O2',
                      suspended_growth_model=asm,
                      split=[0.98, 0.02])
print("Influent and CSTR unit created.")


# Step 4: Set initial conditions and create system
print("\n--- Step 4: Setting initial conditions and creating system ---")
R1.set_init_conc(
    S_F=5.0, S_A=5.0, S_NH4=10.0, S_PO4=2.0, S_O2=2.0,
    # S_ALK=6*12, # CORRECTED: Removed S_ALK from initial conditions
    X_S=50.0, X_I=50.0, X_H=200.0, X_PAO=50.0, X_AUT=20.0,
    X_PHA=1.0, X_PP=5.0
)
sys = qs.System('WWTP_R1', path=(R1,))
sys.set_dynamic_tracker(R1, R1.outs[0])
print("System created and dynamic tracker set.")


# Step 5: Run the dynamic simulation
sim_time = 50 # days
print(f"\n--- Step 5: Running dynamic simulation for {sim_time} days... ---")
sys.simulate(t_span=(0, sim_time), method='BDF', state_reset_hook='reset_cache')
print("Simulation complete.")


# Step 6: Visualize results
print("\n--- Step 6: Visualizing results ---")
R1 = sys.flowsheet.unit.R1
effluent = sys.flowsheet.stream.effluent

fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
plt.sca(axes[0])
R1.scope.plot_time_series(('S_NH4', 'S_NO3', 'S_PO4'))
axes[0].set_title('Nutrient Concentrations in Reactor')
axes[0].set_ylabel('Concentration [mg/L]')
axes[0].legend()

plt.sca(axes[1])
R1.scope.plot_time_series(('X_H', 'X_PAO', 'X_AUT'))
axes[1].set_title('Biomass Concentrations in Reactor')
axes[1].set_ylabel('Concentration [mg/L]')
axes[1].legend()

plt.sca(axes[2])
effluent.scope.plot_time_series(('S_F', 'S_A'))
axes[2].set_title('Substrate Concentrations in Effluent')
axes[2].set_ylabel('Concentration [mg/L]')
axes[2].legend()

fig.tight_layout()
plt.show()