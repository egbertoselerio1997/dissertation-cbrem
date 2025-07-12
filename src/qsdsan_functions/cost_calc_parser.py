import pandas as pd
import os

def calculate_clarifier_costs(excel_filepath: str):
    """
    Calculates clarifier CAPEX and AOC from a user-provided Excel file.

    Args:
        excel_filepath (str): The path to the Excel file containing 'cost_var'
                                and 'capex_calc' worksheets.

    Returns:
        dict: A dictionary containing all calculated variables, including
              the final CAPEX and AOC, or None if an error occurs.
    """
    if not os.path.exists(excel_filepath):
        print(f"Error: The file '{excel_filepath}' was not found.")
        print("Please ensure the file exists and the path is correct.")
        return None
        
    try:
        # Read the variables from the 'cost_var' sheet
        # Use the 'Variable' column as the index for easy lookup
        cost_vars_df = pd.read_excel(excel_filepath, sheet_name='cost_var', index_col=0)
        
        # Filter out rows that are just for comments/headers (where 'Value' is NaN)
        cost_vars_df = cost_vars_df.dropna(subset=['Value'])
        
        # Convert the 'Value' column to a dictionary for easy access
        # e.g., {'C1_surface_area': 750, 'C1_height': 4, ...}
        variables = cost_vars_df['Value'].to_dict()

        # Read the calculations from the 'capex_calc' sheet
        calc_steps_df = pd.read_excel(excel_filepath, sheet_name='capex_calc')
        
        # Filter out comment rows
        calc_steps_df = calc_steps_df.dropna(subset=['Calculation'])

        print("--- Input Variables ---")
        for key, value in variables.items():
            print(f"{key}: {value}")
        print("-" * 25)

        # Dictionary to store the results of the calculations
        calculated_results = {}

        print("\n--- Performing Calculations ---")
        # Iterate through each row in the calculation sheet and execute the formula
        for index, row in calc_steps_df.iterrows():
            output_var = row['Output Variable']
            calculation_str = row['Calculation']

            # Combine the initial variables with any intermediate results
            # This allows calculations to use results from previous steps
            eval_context = {**variables, **calculated_results}

            try:
                # Execute the Python expression from the 'Calculation' column
                # Using pandas.eval for safe evaluation of numerical expressions
                result = pd.eval(calculation_str, local_dict=eval_context, engine='python')
                
                # Store the result
                calculated_results[output_var] = result
                
                print(f"Calculated: {output_var:<22} | Result: {result:,.2f}")

            except Exception as e:
                print(f"Error calculating '{output_var}': {e}")
                print(f"Failed expression: {calculation_str}")
                return None
        
        print("-" * 25)
        
        return calculated_results

    except KeyError as e:
        print(f"Error: A required sheet or column is missing from the Excel file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing the Excel file: {e}")
        return None

if __name__ == '__main__':
    # Define the path to the configuration file.
    # Using os.path.join is a robust way to create file paths that work on any OS.
    config_file_path = os.path.join('data', 'optimization_config.xlsx')

    # --- Execute the calculation using the user-provided file ---
    final_costs = calculate_clarifier_costs(config_file_path)

    if final_costs:
        print("\n--- Final Results ---")
        capex = final_costs.get('CAPEX_C1', 0)
        aoc = final_costs.get('AOC_C1', 0)
        print(f"Total Clarifier CAPEX: ${capex:,.2f}")
        print(f"Total Clarifier AOC:  ${aoc:,.2f} per year")