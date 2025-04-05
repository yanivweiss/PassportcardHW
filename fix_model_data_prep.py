import json
import os

# The path to the notebook we want to fix
notebook_path = 'notebooks/2_PassportCard_Model_Development.ipynb'

# Read the notebook file
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the prepare_for_modeling function cell
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'prepare_for_modeling' in ''.join(cell['source']):
        # This is the function we want to modify
        source_lines = cell['source']
        
        # Find where to insert our fix - right after the feature selection
        for i, line in enumerate(source_lines):
            if 'X = data[feature_cols]' in line:
                insert_index = i + 1
                break
        
        # Insert our fix to filter out non-numeric columns
        fix_code = [
            "    # Filter out non-numeric columns to avoid conversion errors\n",
            "    X = X.select_dtypes(include=['int64', 'float64'])\n",
            "    \n"
        ]
        
        # Insert the fix at the appropriate position
        source_lines[insert_index:insert_index] = fix_code
        
        # Update the cell source
        cell['source'] = source_lines
        
        print("Successfully added fix to filter non-numeric columns!")
        break

# Write the modified notebook back to the file
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Updated {notebook_path}") 