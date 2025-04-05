import json
import os

# Path to the notebook file
notebook_path = 'notebooks/2_PassportCard_Model_Development.ipynb'

# Read the notebook content
with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Parse the JSON content
try:
    notebook = json.loads(content)
    
    # Fix the first markdown cell
    first_cell = notebook['cells'][0]
    if first_cell['cell_type'] == 'markdown':
        old_text = first_cell['source'][0]
        if 'DevelopmentThis' in old_text:
            # Replace the problematic text
            new_text = "# 2. PassportCard Model Development\n\nThis notebook builds on the data exploration from notebook 1, focusing on feature engineering and predictive model development."
            first_cell['source'] = [new_text]
    
    # Write the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Successfully fixed {notebook_path}")
    
except json.JSONDecodeError as e:
    # If JSON parsing fails, try a direct string replacement
    if 'DevelopmentThis' in content:
        fixed_content = content.replace(
            "# 2. PassportCard Model DevelopmentThis", 
            "# 2. PassportCard Model Development\n\nThis"
        )
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed {notebook_path} using string replacement")
    else:
        print(f"Error: Failed to parse notebook JSON: {e}") 