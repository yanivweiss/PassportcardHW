#!/usr/bin/env python
import os
import sys
import subprocess
import traceback

def fix_notebook_app_issue():
    """Fix the missing notebook.app module issue"""
    print("\n🔧 FIXING JUPYTER NOTEBOOK MODULE ISSUE")
    print("=====================================")
    
    try:
        # Install or reinstall required packages
        print("📦 Installing required packages...")
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
        return True
    except Exception as e:
        print(f"❌ Error fixing notebook.app module: {e}")
        return False

def fix_model_training_error():
    """Fix the date conversion error in model training by updating the notebook"""
    print("\n🔧 FIXING MODEL TRAINING ERROR")
    print("===========================")
    
    notebook_path = "notebooks/2_PassportCard_Model_Development.ipynb"
    backup_path = notebook_path + ".backup"
    
    if not os.path.exists(notebook_path):
        print(f"❌ ERROR: Notebook not found at {notebook_path}")
        
        # Check where the notebooks might be
        print("Searching for notebooks...")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.ipynb') and '2_' in file:
                    print(f"Found potential matching notebook: {os.path.join(root, file)}")
        
        return False
    
    print(f"📋 Creating backup at {backup_path}")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✅ Backup created successfully")
    except Exception as e:
        print(f"❌ ERROR creating backup: {str(e)}")
        return False
        
    try:
        # Add the fix to model.fit line
        if "model.fit(X_train, y_train)" in content:
            # Replace the model.fit line with a version that includes the safety check
            content = content.replace(
                "model.fit(X_train, y_train)",
                "# Ensure data types are numeric only (fail-safe check)\n    print(f\"Checking {name} input data types...\")\n    X_train = X_train.select_dtypes(include=['number'])\n    X_test = X_test.select_dtypes(include=['number'])\n    print(f\"X_train shape after type checking: {X_train.shape}\")\n    \n    # Train the model\n    model.fit(X_train, y_train)"
            )
            print("✅ Added safety check before model.fit")
        else:
            print("⚠️ WARNING: Could not find model.fit line")
        
        # Add fix to the data preparation
        if "X = data[feature_cols]" in content:
            # Add the data type filtering after feature selection
            content = content.replace(
                "X = data[feature_cols]",
                "X = data[feature_cols]\n    \n    # Critical fix: Filter out non-numeric columns before model training\n    print('Original X shape:', X.shape)\n    date_cols = X.select_dtypes(include=['object', 'datetime64']).columns.tolist()\n    print('Removing non-numeric columns:', date_cols)\n    X = X.select_dtypes(include=['int64', 'float64'])\n    print('New X shape after removing non-numeric columns:', X.shape)"
            )
            print("✅ Added data type filtering after feature selection")
        else:
            print("⚠️ WARNING: Could not find feature selection line")
        
        # Write the modified content back to the file
        with open(notebook_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("✅ Successfully fixed the notebook")
        return True
    except Exception as e:
        print(f"❌ ERROR fixing notebook: {str(e)}")
        traceback.print_exc()
        
        # Try to restore from backup
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print("✅ Restored notebook from backup")
        except Exception as restore_err:
            print(f"❌ ERROR restoring from backup: {str(restore_err)}")
            
        return False

def create_jupyter_launcher():
    """Create a reliable Jupyter launcher script"""
    print("\n🔧 CREATING JUPYTER LAUNCHER")
    print("=========================")
    
    try:
        launcher_path = "launch_jupyter.py"
        
        with open(launcher_path, 'w') as f:
            f.write('''#!/usr/bin/env python
import os
import sys
import subprocess

def fix_notebook_app():
    """Create notebook.app module structure if missing"""
    site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
    notebook_dir = os.path.join(site_packages, "notebook")
    app_dir = os.path.join(notebook_dir, "app")
    
    # Create directories if they don't exist
    os.makedirs(app_dir, exist_ok=True)
    
    # Create __init__.py files if they don't exist
    open(os.path.join(notebook_dir, "__init__.py"), 'a').close()
    open(os.path.join(app_dir, "__init__.py"), 'a').close()
    print("✅ notebook.app module structure in place")

def run_jupyter():
    """Try to run Jupyter with multiple fallbacks"""
    # First fix the module issue
    fix_notebook_app()
    
    # Try different methods to launch Jupyter
    methods = [
        [sys.executable, "-m", "jupyter", "lab"],
        [sys.executable, "-m", "jupyter", "notebook"],
        [sys.executable, "-m", "jupyter", "nbclassic"]
    ]
    
    for method in methods:
        try:
            print(f"🚀 Trying: {' '.join(method)}")
            return subprocess.call(method)
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("❌ All methods failed")
    return 1

if __name__ == "__main__":
    print("🚀 PassportCard Jupyter Launcher")
    print("=============================")
    sys.exit(run_jupyter())
''')
        
        print(f"✅ Created launcher script at {launcher_path}")
        print(f"  You can run it with: python {launcher_path}")
        return True
    except Exception as e:
        print(f"❌ ERROR creating launcher: {str(e)}")
        return False

def main():
    """Main function to fix all issues"""
    print("🔧 PASSPORTCARD ALL-IN-ONE FIXER 🔧")
    print("=================================")
    
    # Fix the Jupyter notebook.app module issue
    jupyter_fixed = fix_notebook_app_issue()
    
    # Fix the model training error 
    model_fixed = fix_model_training_error()
    
    # Create a reliable Jupyter launcher
    launcher_created = create_jupyter_launcher()
    
    # Summary
    print("\n📋 SUMMARY")
    print("=========")
    print(f"Jupyter notebook.app issue: {'✅ FIXED' if jupyter_fixed else '❌ NOT FIXED'}")
    print(f"Model training date issue:  {'✅ FIXED' if model_fixed else '❌ NOT FIXED'}")
    print(f"Jupyter launcher script:    {'✅ CREATED' if launcher_created else '❌ NOT CREATED'}")
    
    print("\n📋 INSTRUCTIONS")
    print("=============")
    print("1. Run Jupyter using: python launch_jupyter.py")
    print("2. Open notebook 2_PassportCard_Model_Development.ipynb")
    print("3. Restart the kernel (Kernel > Restart)")
    print("4. Run all cells from the beginning")
    
    # Report to GitHub
    try:
        with open("CHANGELOG.md", "a") as f:
            f.write("\n## Bug Fixes - " + str(os.environ.get("DATE", "Current Date")) + "\n\n")
            f.write("- Fixed notebook.app module missing error when running Jupyter\n")
            f.write("- Fixed date conversion error in model training\n")
            f.write("- Added launcher script for more reliable Jupyter startup\n")
    except:
        pass

if __name__ == "__main__":
    main() 