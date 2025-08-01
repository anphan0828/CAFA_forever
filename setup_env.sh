#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "lafa" ]; then
    python3 -m venv lafa
fi

# Activate virtual environment
source lafa/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Environment setup complete!"
echo "To activate: source venv/bin/activate"
echo "To run app: streamlit run streamlit_plot.py"
