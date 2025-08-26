@echo off
echo 🚀 Setting up Financial Sentiment Analysis Demo...
echo ==================================================

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment in 'nlp_env'...
python -m venv nlp_env
call nlp_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📥 Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo 🎉 Setup complete!
echo ==================================================
echo To start the demo, run these commands:
echo.
echo 1. Activate environment: nlp_env\Scripts\activate.bat
echo 2. Run demo:           python quick_demo.py
echo.
pause
