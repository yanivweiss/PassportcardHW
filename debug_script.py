import os
import json

def check_notebook_exists():
    notebook_path = "notebooks/2_PassportCard_Model_Development.ipynb"
    
    print(f"Checking if notebook exists at path: {notebook_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    if os.path.exists(notebook_path):
        print(f"✅ Notebook exists at {notebook_path}")
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
            print(f"✅ Successfully loaded notebook JSON structure")
            
            code_cells = [cell for cell in notebook_data['cells'] if cell['cell_type'] == 'code']
            print(f"Found {len(code_cells)} code cells in the notebook")
            
            for i, cell in enumerate(code_cells):
                source = ''.join(cell['source'])
                if 'prepare_for_modeling' in source:
                    print(f"Found prepare_for_modeling in code cell {i}")
                if 'evaluate_model' in source:
                    print(f"Found evaluate_model in code cell {i}")
        except Exception as e:
            print(f"❌ Error reading notebook: {e}")
    else:
        print(f"❌ Notebook not found at {notebook_path}")
        
        # Check for notebooks in the current directory
        notebooks_in_cwd = [f for f in os.listdir('.') if f.endswith('.ipynb')]
        print(f"Notebooks in current directory: {notebooks_in_cwd}")
        
        # Check if notebooks directory exists
        if os.path.exists('notebooks'):
            print("'notebooks' directory exists")
            print(f"Files in notebooks directory: {os.listdir('notebooks')}")
        else:
            print("'notebooks' directory does not exist")
            
            # Look for notebooks everywhere
            all_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.ipynb'):
                        all_files.append(os.path.join(root, file))
            print(f"All notebooks found in workspace: {all_files}")

if __name__ == "__main__":
    check_notebook_exists() 