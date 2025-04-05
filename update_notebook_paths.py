import json
import os

# Define notebooks to update
notebooks = [
    "notebooks/PassportCard_Insurance_Claims_Prediction.ipynb",
    "notebooks/PassportCard_Model_Development.ipynb",
    "notebooks/PassportCard_Business_Applications.ipynb"
]

# Define patterns to replace in code cells
replacements = [
    ("'claims_data_clean.csv'", "'../claims_data_clean.csv'"),
    ("\"claims_data_clean.csv\"", "\"../claims_data_clean.csv\""),
    ("'members_data_clean.csv'", "'../members_data_clean.csv'"),
    ("\"members_data_clean.csv\"", "\"../members_data_clean.csv\""),
    # Add any other files that might need path updates
    ("'best_xgboost_model.pkl'", "'../models/best_xgboost_model.pkl'")
]

# Process each notebook
for notebook_path in notebooks:
    print(f"Processing {notebook_path}...")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_data = json.load(f)
    
    # Go through all cells
    modified = False
    for cell in notebook_data["cells"]:
        if cell["cell_type"] == "code":
            # Process each source line in code cells
            for i, line in enumerate(cell["source"]):
                for old_text, new_text in replacements:
                    if old_text in line:
                        cell["source"][i] = line.replace(old_text, new_text)
                        modified = True
                        print(f"  Updated: {old_text} -> {new_text}")
    
    # Write the updated notebook
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=1)
        print(f"  Saved updates to {notebook_path}")
    else:
        print(f"  No changes needed in {notebook_path}")

print("All notebooks updated successfully!") 