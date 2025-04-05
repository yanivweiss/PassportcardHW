@echo off
echo ===================================
echo PassportCard Jupyter Launcher
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
)

echo Running Jupyter diagnostic script...
python jupyter_check.py

echo Running Jupyter launcher...
python launch_jupyter.py

pause 