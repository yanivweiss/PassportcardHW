#!/usr/bin/env python
import os
import sys
import subprocess
import platform
import time

def fix_notebook_app():
    """Create notebook.app module structure if missing"""
    print("🔧 Fixing notebook.app module structure...")
    
    # Get site-packages directory
    site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
    print(f"Site packages directory: {site_packages}")
    
    notebook_dir = os.path.join(site_packages, "notebook")
    app_dir = os.path.join(notebook_dir, "app")
    
    # Create directories if they don't exist
    os.makedirs(app_dir, exist_ok=True)
    
    # Create __init__.py files if they don't exist
    init_path1 = os.path.join(notebook_dir, "__init__.py")
    init_path2 = os.path.join(app_dir, "__init__.py")
    
    with open(init_path1, 'a') as f:
        pass
    
    with open(init_path2, 'a') as f:
        pass
    
    print(f"✅ Created directories and files:")
    print(f"  - {notebook_dir}")
    print(f"  - {app_dir}")
    print(f"  - {init_path1}")
    print(f"  - {init_path2}")

def install_required_packages():
    """Install required packages for Jupyter"""
    print("📦 Installing required packages...")
    try:
        # Install required packages - retry with --user if first attempt fails
        try:
            print("Attempting to install notebook==6.4.12...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "notebook==6.4.12", "nbclassic"])
        except subprocess.CalledProcessError:
            print("Regular install failed, trying with --user flag...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--user", "notebook==6.4.12", "nbclassic"])

        print("✅ Successfully installed required packages")
        return True
    except Exception as e:
        print(f"❌ Error installing packages: {str(e)}")
        print("Continuing anyway as the structure fix might still work...")
        return False

def run_jupyter():
    """Try to run Jupyter with multiple fallbacks"""
    # First try to install required packages
    install_required_packages()
    
    # Fix the module issue
    fix_notebook_app()
    
    print("\n🚀 Launching Jupyter...")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    # Try different methods to launch Jupyter
    methods = [
        [sys.executable, "-m", "jupyter", "lab"],
        [sys.executable, "-m", "jupyter", "notebook"],
        [sys.executable, "-m", "jupyter", "nbclassic"]
    ]
    
    for method in methods:
        try:
            print(f"\n🚀 Trying: {' '.join(method)}")
            # Run the method without capturing output so user can see the full output
            return subprocess.call(method)
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("❌ All standard methods failed! Last resort: direct execution")
    
    # Direct execution attempt
    try:
        jupyter_path = os.path.join(os.path.dirname(sys.executable), "Scripts", "jupyter.exe")
        if os.path.exists(jupyter_path):
            print(f"Found jupyter.exe at: {jupyter_path}")
            print(f"Trying direct execution: {jupyter_path} notebook")
            return subprocess.call([jupyter_path, "notebook"])
        else:
            print(f"❌ Could not find jupyter.exe at {jupyter_path}")
    except Exception as e:
        print(f"❌ Direct execution failed: {e}")

    print("\n❌ All methods failed")
    print("Please try these manual steps:")
    print("1. pip install -U notebook==6.4.12 nbclassic jupyterlab")
    print("2. python -m notebook.app")
    return 1

if __name__ == "__main__":
    print("🚀 PassportCard Jupyter Launcher")
    print("=============================")
    
    try:
        result = run_jupyter()
        if result != 0:
            print("\n❌ Jupyter launch failed with exit code", result)
            print("Waiting 5 seconds before exiting...")
            time.sleep(5)
        sys.exit(result)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Waiting 5 seconds before exiting...")
        time.sleep(5)
        sys.exit(1)
