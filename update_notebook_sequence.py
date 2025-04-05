import json
import os
import shutil

# Define notebooks with their sequence
notebooks = [
    {
        "old_name": "notebooks/PassportCard_Insurance_Claims_Prediction.ipynb",
        "new_name": "notebooks/1_PassportCard_Insurance_Claims_Prediction.ipynb",
        "header": "# 1. PassportCard Insurance Claims Prediction\n\nThis project develops a machine learning system to predict future insurance claims for PassportCard policyholders."
    },
    {
        "old_name": "notebooks/PassportCard_Model_Development.ipynb",
        "new_name": "notebooks/2_PassportCard_Model_Development.ipynb",
        "header": "# 2. PassportCard Model Development\n\nThis notebook builds on the data exploration from notebook 1, focusing on feature engineering and predictive model development."
    },
    {
        "old_name": "notebooks/PassportCard_Business_Applications.ipynb",
        "new_name": "notebooks/3_PassportCard_Business_Applications.ipynb",
        "header": "# 3. PassportCard Business Applications\n\nThis notebook applies the models developed in notebook 2 to derive business insights and actionable recommendations."
    }
]

# Process each notebook
for notebook in notebooks:
    old_name = notebook["old_name"]
    new_name = notebook["new_name"]
    new_header = notebook["header"]
    
    print(f"Processing {old_name}...")
    
    # Check if file exists
    if not os.path.exists(old_name):
        print(f"  ERROR: File {old_name} not found, skipping")
        continue
    
    # Read the notebook
    with open(old_name, 'r', encoding='utf-8') as f:
        notebook_data = json.load(f)
    
    # Update the header (assuming it's the first markdown cell)
    for cell in notebook_data["cells"]:
        if cell["cell_type"] == "markdown" and len(cell["source"]) > 0 and cell["source"][0].startswith("# "):
            cell["source"] = new_header.split('\n')
            print(f"  Updated header in {old_name}")
            break
    
    # Save with the new content
    with open(old_name, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=1)
    
    # Rename the file
    shutil.move(old_name, new_name)
    print(f"  Renamed to {new_name}")

print("All notebooks updated successfully!") 