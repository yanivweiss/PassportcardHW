#!/usr/bin/env python
import os
import sys
import subprocess

def fix_notebook_app_issue():
    """Fix the missing notebook.app module issue"""
    print("Fixing notebook.app module issue...")
    
    try:
        # Install or reinstall required packages
        print("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "notebook==6.4.12", "nbclassic", "jupyterlab"], 
                     check=True, capture_output=True)
        
        # Create the missing module structure
        site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
        notebook_dir = os.path.join(site_packages, "notebook")
        app_dir = os.path.join(notebook_dir, "app")
        
        # Create directories if they don't exist
        os.makedirs(app_dir, exist_ok=True)
        
        # Create __init__.py files if they don't exist
        with open(os.path.join(notebook_dir, "__init__.py"), 'a') as f:
            pass
        
        with open(os.path.join(app_dir, "__init__.py"), 'a') as f:
            pass
        
        print("✅ Successfully created notebook.app module structure")
        
        # Create a startup script in the system path
        try:
            # Get the scripts directory location
            scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
            
            # Create a more robust wrapper script
            wrapper_path = os.path.join(scripts_dir, "jupyter-notebook-fixed.py")
            
            with open(wrapper_path, 'w') as f:
                f.write('''#!/usr/bin/env python
import os
import sys
import subprocess

# Try to create notebook.app module if needed
site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
notebook_dir = os.path.join(site_packages, "notebook")
app_dir = os.path.join(notebook_dir, "app")
os.makedirs(app_dir, exist_ok=True)
open(os.path.join(notebook_dir, "__init__.py"), 'a').close()
open(os.path.join(app_dir, "__init__.py"), 'a').close()

# Launch notebook
try:
    subprocess.call([sys.executable, "-m", "jupyter", "notebook"])
except Exception as e:
    print(f"Failed to start regular notebook: {e}")
    try:
        subprocess.call([sys.executable, "-m", "jupyter", "nbclassic"])
    except Exception as e2:
        print(f"Failed to start nbclassic: {e2}")
        # Last resort - direct lab
        subprocess.call([sys.executable, "-m", "jupyter", "lab"])
''')
            
            print(f"✅ Created wrapper script at {wrapper_path}")
            print(f"  You can now run: python {wrapper_path}")
            
        except Exception as script_err:
            print(f"Error creating wrapper script: {script_err}")
        
        return True
    except Exception as e:
        print(f"Error fixing notebook.app module: {e}")
        return False

def run_jupyter():
    """Try to run Jupyter notebook with the fix applied"""
    print("Attempting to run Jupyter Notebook...")
    
    # First apply the fix
    if fix_notebook_app_issue():
        print("Fix applied successfully, now trying to run Jupyter")
        
        try:
            # Try different methods in sequence until one works
            methods = [
                [sys.executable, "-m", "jupyter", "notebook"],
                [sys.executable, "-m", "jupyter", "nbclassic"],
                [sys.executable, "-m", "jupyter", "lab"]
            ]
            
            for i, method in enumerate(methods):
                print(f"Attempt #{i+1}: Running {' '.join(method)}")
                try:
                    result = subprocess.run(method, capture_output=True, text=True)
                    if result.returncode == 0:
                        print("Jupyter started successfully!")
                        return True
                    else:
                        print(f"Failed with error: {result.stderr}")
                except Exception as e:
                    print(f"Error: {e}")
            
            print("All standard methods failed. Using fallback...")
            scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
            wrapper_path = os.path.join(scripts_dir, "jupyter-notebook-fixed.py")
            
            if os.path.exists(wrapper_path):
                subprocess.run([sys.executable, wrapper_path])
                return True
            else:
                print("Could not find wrapper script")
                return False
                
        except Exception as e:
            print(f"Error running Jupyter: {e}")
            return False
    else:
        print("Failed to apply fix")
        return False

if __name__ == "__main__":
    print("🔧 JUPYTER NOTEBOOK.APP FIXER 🔧")
    print("================================")
    
    choice = input("Do you want to (1) just fix the issue or (2) fix and run Jupyter? (1/2): ")
    
    if choice == "1":
        fix_notebook_app_issue()
        print("\nFix applied. You can now run Jupyter using:")
        print("python -m jupyter notebook")
    else:
        run_jupyter() 