import sys
import os
import subprocess

def fix_jupyterlab_path():
    """Fix the JupyterLab path issue by adding required directories to sys.path"""
    try:
        # Install nbclassic if not already installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nbclassic"])
        
        # Create the missing module structure
        notebook_app_dir = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'notebook', 'app')
        os.makedirs(notebook_app_dir, exist_ok=True)
        
        # Create an empty __init__.py file in the notebook directory if it doesn't exist
        notebook_init = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'notebook', '__init__.py')
        if not os.path.exists(notebook_init):
            with open(notebook_init, 'w') as f:
                f.write('# Created by fix_jupyterlab_path.py\n')
        
        # Create an empty __init__.py file in the app directory
        app_init = os.path.join(notebook_app_dir, '__init__.py')
        with open(app_init, 'w') as f:
            f.write('# Created by fix_jupyterlab_path.py\n')
        
        print("JupyterLab path fixed successfully!")
        return True
    except Exception as e:
        print(f"Error fixing JupyterLab path: {e}")
        return False

if __name__ == "__main__":
    if fix_jupyterlab_path():
        # Run jupyter lab
        try:
            print("Starting Jupyter Lab...")
            subprocess.call(["jupyter", "lab"])
        except Exception as e:
            print(f"Error starting Jupyter Lab: {e}")
            # Try running jupyter notebook as fallback
            try:
                print("Starting Jupyter Notebook as fallback...")
                subprocess.call(["jupyter", "notebook"])
            except Exception as e:
                print(f"Error starting Jupyter Notebook: {e}")
    else:
        print("Failed to fix JupyterLab path. Please reinstall Jupyter.") 