# Jupyter Installation and Troubleshooting Guide

This guide will help you fix Jupyter installation issues and run the notebooks properly.

## Quick Start (Windows)

1. **Setup (one-time)**:
   - Double-click `setup.bat` to install required packages
   - This will create a virtual environment and install all dependencies

2. **Run Jupyter**:
   - Double-click `run_jupyter.bat` to start Jupyter
   - Or run `python launch_jupyter.py` in your terminal

## Quick Start (Mac/Linux)

1. **Setup (one-time)**:
   ```bash
   # Create virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac

   # Install requirements and fix issues
   python setup.py
   ```

2. **Run Jupyter**:
   ```bash
   python launch_jupyter.py
   ```

## Troubleshooting Common Issues

### 1. "No module named 'notebook.app'" Error

This is a known issue with newer versions of Jupyter. We've provided several ways to fix it:

**Automatic Fix**:
- Run `python jupyter_check.py` to diagnose and fix the issue
- Or run `python setup.py` to install all requirements and fix the issue

**Manual Fix**:
1. Install the correct version of the notebook package:
   ```
   pip install notebook==6.4.12 nbclassic
   ```
2. Create the missing module structure:
   ```python
   import os, sys
   site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
   notebook_dir = os.path.join(site_packages, "notebook")
   app_dir = os.path.join(notebook_dir, "app")
   os.makedirs(app_dir, exist_ok=True)
   open(os.path.join(notebook_dir, "__init__.py"), 'a').close()
   open(os.path.join(app_dir, "__init__.py"), 'a').close()
   ```

### 2. Jupyter Not Starting

If Jupyter fails to start:

1. Try different Jupyter commands:
   ```
   jupyter lab
   jupyter notebook
   jupyter nbclassic
   ```

2. Check if packages are installed correctly:
   ```
   pip list | grep jupyter
   pip list | grep notebook
   ```

3. Run our diagnostic tool:
   ```
   python jupyter_check.py
   ```

### 3. Model Training Error

If you encounter a model training error related to date conversions:

1. Run our fix script:
   ```
   python fix_manual.py
   ```

2. Or run the comprehensive fix script:
   ```
   python passportcard_fix_all.py
   ```

## Required Packages

The main packages required are:
- notebook==6.4.12 (specific version needed for our fix)
- nbclassic
- jupyterlab
- pandas, numpy, matplotlib, scikit-learn, etc.

These are all included in our `requirements.txt` file.

## Support

If you continue to experience issues:

1. Run the diagnostic script to collect information:
   ```
   python jupyter_check.py > diagnostic_output.txt
   ```

2. Share the diagnostic output with us for further assistance. 