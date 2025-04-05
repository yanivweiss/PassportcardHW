#!/usr/bin/env python
import subprocess
import sys
import os

def run_jupyter():
    """Run Jupyter notebook or lab"""
    try:
        # Try running jupyter lab first
        print("Starting Jupyter Lab...")
        result = subprocess.run(["jupyter-lab"], capture_output=True, text=True)
        
        if "ModuleNotFoundError: No module named 'notebook.app'" in result.stderr:
            print("Detected missing notebook.app module, trying workaround...")
            # Create a simple wrapper to run jupyter nbclassic
            subprocess.run([sys.executable, "-m", "jupyter", "nbclassic"], check=True)
        elif result.returncode != 0:
            # If jupyter-lab fails for another reason, try jupyter notebook
            print("Jupyter Lab failed, trying Jupyter Notebook...")
            subprocess.run([sys.executable, "-m", "jupyter", "notebook"], check=True)
    except Exception as e:
        print(f"Error running Jupyter: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_jupyter() 