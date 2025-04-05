#!/usr/bin/env python

def fix_notebook():
    print("Fixing the notebook using direct string replacement...")
    
    # Read the notebook file
    notebook_path = "notebooks/2_PassportCard_Model_Development.ipynb"
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Make a backup
    with open(notebook_path + ".manual_backup", 'w', encoding='utf-8') as f:
        f.write(content)
    print("Created backup at", notebook_path + ".manual_backup")
    
    # Add the fix to model.fit line
    if "model.fit(X_train, y_train)" in content:
        # Replace the model.fit line with a version that includes the safety check
        content = content.replace(
            "model.fit(X_train, y_train)",
            "# Ensure data types are numeric only (fail-safe check)\n    print(f\"Checking {name} input data types...\")\n    X_train = X_train.select_dtypes(include=['number'])\n    X_test = X_test.select_dtypes(include=['number'])\n    print(f\"X_train shape after type checking: {X_train.shape}\")\n    \n    # Train the model\n    model.fit(X_train, y_train)"
        )
        print("Added safety check before model.fit")
    else:
        print("WARNING: Could not find model.fit line")
    
    # Add fix to the data preparation
    if "X = data[feature_cols]" in content:
        # Add the data type filtering after feature selection
        content = content.replace(
            "X = data[feature_cols]",
            "X = data[feature_cols]\n    \n    # Critical fix: Filter out non-numeric columns before model training\n    print('Original X shape:', X.shape)\n    date_cols = X.select_dtypes(include=['object', 'datetime64']).columns.tolist()\n    print('Removing non-numeric columns:', date_cols)\n    X = X.select_dtypes(include=['int64', 'float64'])\n    print('New X shape after removing non-numeric columns:', X.shape)"
        )
        print("Added data type filtering after feature selection")
    else:
        print("WARNING: Could not find feature selection line")
    
    # Write the modified content back to the file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Successfully fixed the notebook!")
    print("\nInstructions:")
    print("1. Restart the Jupyter kernel")
    print("2. Run all cells from the beginning")

if __name__ == "__main__":
    fix_notebook() 