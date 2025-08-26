@echo off
setlocal

:: Default to demo mode
set "INSTALL_MODE=demo"

:menu
cls
echo ==================================================
echo Financial Sentiment Analysis Setup
echo ==================================================
echo.
echo Choose installation type:
echo   1. Quick Demo (minimal dependencies)
echo   2. Full Setup (for running notebooks)
echo.
set /p "CHOICE=Enter your choice (1 or 2): "

if "%CHOICE%"=="1" (
    set "INSTALL_MODE=demo"
    goto start_setup
)
if "%CHOICE%"=="2" (
    set "INSTALL_MODE=full"
    goto start_setup
)

echo Invalid choice. Please enter 1 or 2.
timeout /t 2 >nul
goto menu

:start_setup
cls
echo Setting up Financial Sentiment Analysis (%INSTALL_MODE% mode)...
echo ==================================================

echo Creating virtual environment in 'nlp_env'...
python -m venv nlp_env
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment. Please ensure Python 3 is installed and in your PATH.
    pause
    exit /b 1
)

call nlp_env\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies from requirements-%INSTALL_MODE%.txt...
pip install -r requirements-%INSTALL_MODE%.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies. See the output above for details.
    goto end_setup
)

echo.
echo Setup complete!
echo ==================================================
:end_setup
echo To start the demo, run these commands:
echo 1. Activate environment: nlp_env\Scripts\activate
echo 2. Run demo:           python quick_demo.py
pause
