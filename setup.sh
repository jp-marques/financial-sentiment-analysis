#!/bin/bash

# Function to display colored text
c_echo() {
    COLOR=$1
    TEXT=$2
    case $COLOR in
        "green") echo -e "\033[92m${TEXT}\033[0m" ;;
        "red") echo -e "\033[91m${TEXT}\033[0m" ;;
        "blue") echo -e "\033[94m${TEXT}\033[0m" ;;
        *) echo "${TEXT}" ;;
    esac
}

clear
echo "=================================================="
echo "Financial Sentiment Analysis Setup"
echo "=================================================="
echo
echo "Choose installation type:"
echo "  1. Quick Demo (minimal dependencies for CLI)"
echo "  2. Full Setup (for running notebooks)"
echo

while true; do
    read -p "Enter your choice (1 or 2): " CHOICE
    case $CHOICE in
        1)
            INSTALL_MODE="demo"
            break
            ;;
        2)
            INSTALL_MODE="full"
            break
            ;;
        *)
            echo "Invalid choice. Please enter 1 or 2."
            ;;
    esac
done

clear
echo "=================================================="
echo "Setting up Financial Sentiment Analysis (${INSTALL_MODE} mode)..."
echo "=================================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment in 'nlp_env'..."
python3 -m venv nlp_env
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

source nlp_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements-${INSTALL_MODE}.txt..."
pip install -r "requirements-${INSTALL_MODE}.txt"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. See the output above for details."
else
    echo
    echo "Setup complete!"
    echo "=================================================="
fi

echo "To start the demo, run these commands:"
echo "1. Activate environment: source nlp_env/bin/activate"
echo "2. Run demo:           python3 quick_demo.py"
echo
