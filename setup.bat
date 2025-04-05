@echo off
echo ===================================
echo PassportCard Setup
echo ===================================

echo Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH!
    echo Please install Python and ensure it's in your PATH
    pause
    exit /b 1
)

echo Checking virtual environment...
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Creating one...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment!
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
)

echo Running setup script...
python setup.py

if %ERRORLEVEL% NEQ 0 (
    echo Setup failed! Please check the error messages above.
) else (
    echo Setup completed successfully!
    echo.
    echo You can now run Jupyter by:
    echo 1. Double-clicking run_jupyter.bat
    echo 2. Running 'python launch_jupyter.py'
)

pause 