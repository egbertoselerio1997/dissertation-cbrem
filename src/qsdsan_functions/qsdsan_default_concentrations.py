import qsdsan as qs
import pandas as pd
import numpy as np

# --- 1. Create the Influent WasteStream ---
print(f'This script was run with qsdsan v{qs.__version__}.')
cmps = qs.Components.load_default()
qs.set_thermo(cmps)
ww = qs.WasteStream.codbased_inf_model('ww', flow_tot=1000, units=('L/hr', 'mg/L'))
print("\nWasteStream 'ww' has been created successfully.")


# --- 2. Extract Data for Each Sheet ---

# Sheet 1: General Properties
general_data = {
    'Property': [
        'Temperature', 'Pressure', 'pH', 'Alkalinity', 
        'Total Mass Flow', 'Total Molar Flow', 'Total Volumetric Flow'
    ],
    'Value': [
        ww.T, ww.P, ww.pH, ww.SAlk,
        ww.F_mass, ww.F_mol, ww.F_vol
    ],
    'Unit': [
        'K', 'Pa', '-', 'meq/L',
        'kg/hr', 'kmol/hr', 'm3/hr'
    ]
}
df_general = pd.DataFrame(general_data)


# Sheet 2: Component-specific Data
component_data = []
for cmp in ww.components:
    # Use a small tolerance to only include components with non-zero flow
    if ww.imass[cmp.ID] > 1e-9:
        component_data.append({
            'Component ID': cmp.ID,
            'Mass Flow (g/hr)': ww.imass[cmp.ID] * 1000, # convert from kg/hr
            'Molar Flow (mol/hr)': ww.imol[cmp.ID] * 1000, # convert from kmol/hr
            'Concentration (mg/L)': ww.iconc[cmp.ID]
        })
df_components = pd.DataFrame(component_data)


# Sheet 3: Composite Wastewater Properties (The Results)
composite_properties = {
    'Property': [
        'COD', 'BOD5', 'uBOD', 'cnBOD', 'ThOD',
        'TC', 'TOC', 'TN', 'TKN', 'TP', 'TK', 'TCa', 'TMg',
        'TSS (excluding colloidal)', 'TSS (including colloidal)',
        'VSS (excluding colloidal)', 'VSS (including colloidal)',
        'ISS',
        'TDS (excluding colloidal)', 'TDS (including colloidal)'
    ],
    'Value': [
        ww.COD, ww.BOD, ww.uBOD, ww.cnBOD, ww.ThOD,
        ww.TC, ww.TOC, ww.TN, ww.TKN, ww.TP, ww.TK, ww.TCa, ww.TMg,
        ww.get_TSS(include_colloidal=False), ww.get_TSS(include_colloidal=True),
        ww.get_VSS(include_colloidal=False), ww.get_VSS(include_colloidal=True),
        ww.get_ISS(),
        ww.get_TDS(include_colloidal=False), ww.get_TDS(include_colloidal=True)
    ],
    'Unit': ['mg/L'] * 20
}
df_composites = pd.DataFrame(composite_properties)


# Sheet 4: Composite Calculation Parameters (The Coefficients)
params_data = []
for cmp in ww.components:
    # We include all components here, even those with zero flow,
    # to create a complete reference table.
    params_data.append({
        'Component ID': cmp.ID,
        'Measured As': cmp.measured_as or 'mass',
        'Particle Size': cmp.particle_size,
        'i_COD (g COD/g)': cmp.i_COD,
        'i_N (g N/g)': cmp.i_N,
        'i_P (g P/g)': cmp.i_P,
        'i_K (g K/g)': cmp.i_K,
        'i_Ca (g Ca/g)': cmp.i_Ca,
        'i_Mg (g Mg/g)': cmp.i_Mg,
        'f_BOD5_COD (-)': cmp.f_BOD5_COD,
        'f_uBOD_COD (-)': cmp.f_uBOD_COD,
        'f_Vmass_Totmass (-)': cmp.f_Vmass_Totmass
    })
df_params = pd.DataFrame(params_data)


# --- 3. Write all DataFrames to an Excel file ---
output_filename = 'data.xlsx'
with pd.ExcelWriter(output_filename) as writer:
    df_general.to_excel(writer, sheet_name='General_Properties', index=False)
    df_components.to_excel(writer, sheet_name='Component_Data', index=False)
    df_composites.to_excel(writer, sheet_name='Composite_Properties', index=False)
    df_params.to_excel(writer, sheet_name='Composite_Parameters', index=False)

print(f"\nSuccessfully exported all WasteStream data and parameters to '{output_filename}'.")