#!/usr/bin/env python
import os
import sys
import subprocess
import platform

def install_requirements():
    """Install all required packages from requirements.txt"""
    print("\n📦 Installing requirements from requirements.txt...")
    
    try:
        # Try with regular install
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Regular install failed, trying with --user flag...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    
    print("✅ Requirements installed successfully")
    return True

def fix_notebook_app_module():
    """Create the missing notebook.app module structure"""
    try:
        # Get site-packages directory
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
        
        print("✅ Fixed notebook.app module structure")
        return True
    except Exception as e:
        print(f"❌ Failed to fix notebook.app module: {e}")
        return False

def check_environment():
    """Check the Python environment and installed packages"""
    print("\n🔍 Checking environment...")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}")
    
    # Check if virtual environment is active
    in_virtualenv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_virtualenv:
        print("✅ Running in a virtual environment")
    else:
        print("⚠️ Not running in a virtual environment")
    
    # Check for notebook module
    try:
        import notebook
        print(f"✅ Notebook module found (version: {notebook.__version__})")
    except ImportError:
        print("❌ Notebook module not found")
    
    # Check for notebook.app module
    try:
        import notebook.app
        print("✅ notebook.app module exists")
    except ImportError:
        print("❌ notebook.app module is missing")
    
    return True

def setup():
    """Run the full setup process"""
    print("🔧 PASSPORTCARD SETUP 🔧")
    print("=======================")
    
    # Check environment
    check_environment()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Fix notebook.app module
    if not fix_notebook_app_module():
        print("❌ Failed to fix notebook.app module")
        return False
    
    # Final check
    print("\n🔍 Checking final setup...")
    check_environment()
    
    print("\n✅ Setup completed successfully!")
    print("\nTo run Jupyter, use one of these options:")
    print("1. Double-click run_jupyter.bat (Windows)")
    print("2. Run: python launch_jupyter.py")
    print("3. Run: python jupyter_check.py (for diagnostics)")
    
    return True

if __name__ == "__main__":
    setup() 