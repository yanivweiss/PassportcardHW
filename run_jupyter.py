#!/usr/bin/env python
import subprocess
import sys
import os

def fix_notebook_app_issue():
    """Fix the missing notebook.app module issue"""
    try:
        # Create the missing module structure
        site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
        notebook_dir = os.path.join(site_packages, "notebook")
        app_dir = os.path.join(notebook_dir, "app")
        
        # Create directories if they don't exist
        os.makedirs(app_dir, exist_ok=True)
        
        # Create __init__.py files if they don't exist
        open(os.path.join(notebook_dir, "__init__.py"), 'a').close()
        open(os.path.join(app_dir, "__init__.py"), 'a').close()
        
        print("✅ Fixed notebook.app module issue")
        return True
    except Exception as e:
        print(f"❌ Error fixing notebook.app module: {e}")
        return False

def run_jupyter():
    """Run Jupyter notebook or lab with automatic fixes"""
    # First fix the notebook.app issue
    fix_notebook_app_issue()
    
    try:
        # First try running jupyter lab
        print("🚀 Starting Jupyter Lab...")
        lab_process = subprocess.run([sys.executable, "-m", "jupyter", "lab"], 
                                    capture_output=True, text=True)
        
        # If lab fails, try notebook
        if lab_process.returncode != 0:
            print("⚠️ Jupyter Lab failed, trying Jupyter Notebook...")
            try:
                notebook_process = subprocess.run([sys.executable, "-m", "jupyter", "notebook"], 
                                                check=True)
            except subprocess.CalledProcessError:
                # If notebook fails, try nbclassic as a last resort
                print("⚠️ Trying nbclassic as fallback...")
                subprocess.run([sys.executable, "-m", "jupyter", "nbclassic"], check=True)
    except Exception as e:
        print(f"❌ Error running Jupyter: {e}")
        
        # Last resort - try to launch using a direct system call
        print("🔄 Trying alternative method to start Jupyter...")
        try:
            if sys.platform.startswith('win'):
                os.system('jupyter notebook')
            else:
                os.system('jupyter notebook &')
        except Exception as last_e:
            print(f"❌ All attempts to start Jupyter failed: {last_e}")
            print("\n📋 Troubleshooting steps:")
            print("1. Run: pip install -U notebook==6.4.12 jupyter ipykernel nbclassic")
            print("2. Try running: python -m jupyter notebook")
            sys.exit(1)

if __name__ == "__main__":
    print("🛠️ PassportCard Jupyter Launcher")
    print("================================")
    run_jupyter() 