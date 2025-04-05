#!/usr/bin/env python
import json
import os

def fix_model_training_error():
    """Fix the date conversion error in model training by updating the notebook"""
    notebook_path = "notebooks/2_PassportCard_Model_Development.ipynb"
    backup_path = notebook_path + ".bak"
    
    print(f"Creating backup of {notebook_path} to {backup_path}")
    # Create a backup of the original file
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
            
        print("Backup created successfully")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the data preparation cell
        prep_cell_found = False
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'prepare_for_modeling' in ''.join(cell['source']):
                prep_cell_found = True
                print(f"Found prepare_for_modeling function in cell {i}")
                
                # Create new source code with the fix
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    
                    # Add fix after the line that selects features
                    if 'X = data[feature_cols]' in line:
                        # Add the fix to filter out non-numeric columns
                        new_source.append("\n    # Filter out non-numeric columns to avoid conversion errors\n")
                        new_source.append("    print('Original X columns:', X.columns.tolist())\n")
                        new_source.append("    # Filter by data type\n")
                        new_source.append("    object_cols = X.select_dtypes(include=['object', 'datetime64']).columns.tolist()\n")
                        new_source.append("    print('Dropping date/object columns:', object_cols)\n")
                        new_source.append("    X = X.select_dtypes(include=['int64', 'float64'])\n")
                        new_source.append("    print('Remaining X columns:', X.columns.tolist())\n")
                
                # Replace the original source with the fixed version
                cell['source'] = new_source
                break
                
        if not prep_cell_found:
            print("Could not find prepare_for_modeling function in the notebook!")
        
        # Add a safety check in the evaluate_model function
        model_cell_found = False
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'def evaluate_model' in ''.join(cell['source']):
                model_cell_found = True
                print(f"Found evaluate_model function in cell {i}")
                
                # Create new source code with additional safety check
                new_source = []
                for line in cell['source']:
                    # Add safety check before the model fitting line
                    if 'model.fit(X_train, y_train)' in line:
                        new_source.append("    # Add a final safety check to ensure all columns are numeric\n")
                        new_source.append("    print('Final check - X_train dtypes:', X_train.dtypes)\n")
                        new_source.append("    # Force to numeric types only\n")
                        new_source.append("    X_train = X_train.select_dtypes(include=['number'])\n")
                        new_source.append("    X_test = X_test.select_dtypes(include=['number'])\n")
                        new_source.append("    print(f'X_train shape after final check: {X_train.shape}')\n")
                    
                    new_source.append(line)
                
                # Replace the original source with the fixed version
                cell['source'] = new_source
                break
        
        if not model_cell_found:
            print("Could not find evaluate_model function in the notebook!")
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
            
        print(f"Successfully fixed {notebook_path}")
        return True
    
    except Exception as e:
        print(f"Error fixing notebook: {e}")
        # Restore from backup if there was an error
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
                
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
                
            print("Restored from backup due to error")
        except Exception as restore_error:
            print(f"Error restoring from backup: {restore_error}")
        
        return False

if __name__ == "__main__":
    print("🔧 Fixing Model Training Error 🔧")
    print("================================")
    success = fix_model_training_error()
    
    if success:
        print("\n✅ Fix applied successfully!")
        print("To use the fix:")
        print("1. Restart the Jupyter kernel")
        print("2. Run all cells again")
    else:
        print("\n❌ Failed to apply fix")
        print("Try manual fix: Edit the notebook and add the following code after 'X = data[feature_cols]':")
        print("    X = X.select_dtypes(include=['int64', 'float64'])") 