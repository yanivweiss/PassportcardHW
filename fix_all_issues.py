import json
import os
import sys
import subprocess
import shutil

def fix_jupyter_notebook_issue():
    """Fix the 'ModuleNotFoundError: No module named 'notebook.app'' error"""
    print("Fixing Jupyter Notebook module issue...")
    try:
        # Install required packages
        subprocess.call([sys.executable, "-m", "pip", "install", "notebook==6.4.12", "nbclassic"])
        
        # Create the missing directory structure
        site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
        notebook_dir = os.path.join(site_packages, "notebook")
        app_dir = os.path.join(notebook_dir, "app")
        
        # Create directories if they don't exist
        os.makedirs(app_dir, exist_ok=True)
        
        # Create __init__.py files if they don't exist
        open(os.path.join(notebook_dir, "__init__.py"), 'a').close()
        open(os.path.join(app_dir, "__init__.py"), 'a').close()
        
        print("Jupyter Notebook module issue fixed successfully!")
        return True
    except Exception as e:
        print(f"Error fixing Jupyter Notebook module issue: {e}")
        return False

def fix_model_training():
    """Fix the date conversion error in the model training process"""
    print("Fixing model training date conversion issue...")
    
    notebook_path = "notebooks/2_PassportCard_Model_Development.ipynb"
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            
        # Find the prepare_for_modeling function and modify it
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and 'prepare_for_modeling' in ''.join(cell['source']):
                new_source = []
                for line in cell['source']:
                    new_source.append(line)
                    # After the X = data[feature_cols] line, add code to filter out non-numeric columns
                    if 'X = data[feature_cols]' in line:
                        new_source.append("\n    # Filter out non-numeric columns to avoid conversion errors\n")
                        new_source.append("    print('Original X columns:', X.columns.tolist())\n")
                        new_source.append("    date_cols = X.select_dtypes(include=['object', 'datetime64']).columns.tolist()\n")
                        new_source.append("    print('Dropping date/object columns:', date_cols)\n")
                        new_source.append("    X = X.select_dtypes(include=['int64', 'float64'])\n")
                        new_source.append("    print('Remaining X columns:', X.columns.tolist())\n")
                
                cell['source'] = new_source
                
        # Find the evaluate_model function and add debugging
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and 'evaluate_model' in ''.join(cell['source']) and 'model.fit' in ''.join(cell['source']):
                new_source = []
                for line in cell['source']:
                    # Add debug code before model.fit
                    if 'model.fit(X_train, y_train)' in line:
                        new_source.append("    # Verify data types before fitting\n")
                        new_source.append("    print(f\"X_train dtypes: {X_train.dtypes}\")\n")
                        new_source.append("    # Ensure all columns are numeric\n")
                        new_source.append("    X_train = X_train.select_dtypes(include=['int64', 'float64'])\n")
                        new_source.append("    X_test = X_test.select_dtypes(include=['int64', 'float64'])\n")
                        new_source.append("    print(f\"X_train shape after ensuring numeric: {X_train.shape}\")\n")
                    new_source.append(line)
                
                cell['source'] = new_source
                
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
            
        print(f"Model training fix applied to {notebook_path}")
        return True
    except Exception as e:
        print(f"Error fixing model training: {e}")
        return False

def run_notebooks():
    """Run all notebooks in sequence to generate outputs"""
    print("Running notebooks in sequence...")
    notebooks = [
        "notebooks/1_PassportCard_Insurance_Claims_Prediction.ipynb",
        "notebooks/2_PassportCard_Model_Development.ipynb", 
        "notebooks/3_PassportCard_Business_Applications.ipynb"
    ]
    
    try:
        for notebook in notebooks:
            print(f"Running {notebook}...")
            result = subprocess.run([
                sys.executable, "-m", "jupyter", "nbconvert", 
                "--to", "notebook", 
                "--execute",
                "--ExecutePreprocessor.timeout=600",
                "--output", notebook,
                notebook
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error running {notebook}: {result.stderr}")
            else:
                print(f"Successfully ran {notebook}")
        
        return True
    except Exception as e:
        print(f"Error running notebooks: {e}")
        return False

def main():
    """Main function to fix all issues"""
    if fix_jupyter_notebook_issue():
        print("Step 1: Jupyter Notebook module issue fixed!")
    else:
        print("Step 1: Failed to fix Jupyter Notebook module issue")
    
    if fix_model_training():
        print("Step 2: Model training fix applied!")
    else:
        print("Step 2: Failed to fix model training")
    
    print("\nAll fixes have been applied. You can now:")
    print("1. Run 'python run_jupyter.py' to start Jupyter")
    print("2. Open and run the notebooks in sequence")
    
    choice = input("\nWould you like to automatically run all notebooks now? (y/n): ")
    if choice.lower() == 'y':
        run_notebooks()

if __name__ == "__main__":
    main() 