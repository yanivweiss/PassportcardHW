#!/usr/bin/env python
import os
import sys
import subprocess
import importlib
import platform

def check_jupyter_installation():
    """Check if Jupyter is installed correctly and diagnose any issues"""
    print("🔍 Checking Jupyter installation...")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable path: {sys.executable}")
    
    # Check if jupyter module is installed
    try:
        import jupyter
        print(f"✅ Jupyter is installed (version: {jupyter.__version__})")
    except ImportError:
        print("❌ Jupyter is not installed.")
        return False
    
    # Check if notebook module is installed
    try:
        import notebook
        print(f"✅ Notebook is installed (version: {notebook.__version__})")
    except ImportError:
        print("❌ Notebook is not installed.")
        return False
    
    # Check if notebook.app exists
    try:
        import notebook.app
        print("✅ notebook.app module exists")
    except ImportError as e:
        print(f"❌ notebook.app module is missing: {e}")
        print("This is the error we need to fix.")
        
        # Fix notebook.app
        site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
        notebook_dir = os.path.join(site_packages, "notebook")
        app_dir = os.path.join(notebook_dir, "app")
        
        print(f"Creating fix at: {app_dir}")
        
        # Create directories if they don't exist
        os.makedirs(app_dir, exist_ok=True)
        
        # Create __init__.py files if they don't exist
        with open(os.path.join(notebook_dir, "__init__.py"), 'a') as f:
            pass
        
        with open(os.path.join(app_dir, "__init__.py"), 'a') as f:
            pass
        
        print("✅ Fix applied. Try importing notebook.app again...")
        
        try:
            # Try to import again after fix
            importlib.invalidate_caches()
            import notebook.app
            print("✅ notebook.app module now exists after fix")
        except ImportError as e2:
            print(f"❌ Fix didn't work: {e2}")
            return False
    
    # Try to run jupyter command
    try:
        result = subprocess.run([sys.executable, "-m", "jupyter", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Jupyter command works: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Jupyter command failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    print("\n✅ All checks passed! You should be able to run jupyter now.")
    return True

def diagnose_pip_packages():
    """Check installed pip packages"""
    print("\n🔍 Checking installed packages...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                             capture_output=True, text=True, check=True)
        packages = result.stdout.strip()
        
        # Check for key packages
        important_packages = ['jupyter', 'notebook', 'nbclassic', 'jupyterlab']
        for package in important_packages:
            if package in packages:
                print(f"✅ {package} is installed")
            else:
                print(f"❌ {package} is NOT installed")
        
        # Find notebook version specifically
        for line in packages.split('\n'):
            if 'notebook ' in line:
                print(f"📦 {line.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list packages: {e}")

def fix_and_install():
    """Fix installation issues by installing required packages"""
    print("\n🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", 
                             "notebook==6.4.12", "nbclassic", "jupyterlab", "jupyter"])
        print("✅ Packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        try:
            print("Trying with --user flag...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--user",
                                 "notebook==6.4.12", "nbclassic", "jupyterlab", "jupyter"])
            print("✅ Packages installed successfully with --user flag")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Failed to install packages with --user flag: {e2}")
            return False
    
    # Recheck after installation
    print("\n🔍 Rechecking installation...")
    return check_jupyter_installation()

if __name__ == "__main__":
    print("🔎 JUPYTER INSTALLATION DIAGNOSTIC 🔎")
    print("====================================")
    
    # First check the installation
    result = check_jupyter_installation()
    
    # Show pip package information
    diagnose_pip_packages()
    
    # If checks failed, offer to fix
    if not result:
        print("\n❌ Issues detected with Jupyter installation.")
        choice = input("Do you want to try fixing the installation? (y/n): ")
        if choice.lower() == 'y':
            success = fix_and_install()
            if success:
                print("\n✅ Installation fixed. You can now run Jupyter with:")
                print("python launch_jupyter.py")
            else:
                print("\n❌ Couldn't fix installation automatically.")
                print("Try manually running:")
                print("pip install -U notebook==6.4.12 nbclassic jupyterlab jupyter")
        else:
            print("No changes made. You can try fixing manually with:")
            print("pip install -U notebook==6.4.12 nbclassic jupyterlab jupyter")
    else:
        print("\n✅ Jupyter appears to be installed correctly.")
        print("You can run Jupyter with: python launch_jupyter.py") 