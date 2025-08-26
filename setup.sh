#!/bin/bash

echo "🚀 Setting up Financial Sentiment Analysis Demo..."
echo "=================================================="

# Check for Python 3.8+
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment in 'nlp_env'..."
python3 -m venv nlp_env
source nlp_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete!"
echo "=================================================="
echo "To start the demo, run these commands:"
echo ""
echo "1. Activate environment: source nlp_env/bin/activate"
echo "2. Run demo:           python quick_demo.py"
echo ""
