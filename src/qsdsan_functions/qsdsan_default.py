import pandas as pd
import qsdsan as qs

# Load default components and set thermodynamic properties
cmps = qs.Components.load_default()
qs.set_thermo(cmps)

# Generate a default influent WasteStream to get baseline values
default_influent = qs.WasteStream.codbased_inf_model('default_influent', flow_tot=1000)

# Prepare the new rows to be added
new_rows_data = []
# Iterate through all components in the loaded set
for cmp in cmps:
    cmp_id = cmp.ID
    # Get the concentration value, which will be 0.0 for many components
    value = default_influent.iconc[cmp_id]
    
    # Add a row for every component, regardless of its value
    new_rows_data.append({
        'variable': f'inf_{cmp_id}',
        'baseline': value,
        'default': None,
        'std. dev.': None,
        'units': 'mg/L',
        'randomizable': None
    })

df_new_rows = pd.DataFrame(new_rows_data)

# Define the path to the Excel file
file_path = 'data.xlsx'

# Read all existing sheets from the Excel file
try:
    with pd.ExcelFile(file_path) as xls:
        all_sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
except FileNotFoundError:
    # If the file doesn't exist, create the main sheet to append to
    all_sheets = {
        'input_config': pd.DataFrame(columns=['variable', 'baseline', 'default', 'std. dev.', 'units', 'randomizable'])
    }

# Get the specific DataFrame for the 'input_config' sheet
df_input_config = all_sheets.get('input_config')

# Append the new data to the existing DataFrame
df_updated_config = pd.concat([df_input_config, df_new_rows], ignore_index=True)

# Update the dictionary of sheets with the modified DataFrame
all_sheets['input_config'] = df_updated_config

# Write all sheets back to the Excel file, preserving other sheets
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    for sheet_name, df in all_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)