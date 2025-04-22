#!/bin/bash
# Setup script for visualize_knowledge FastAPI app

set -e

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Setup complete. To run the app:"
echo "source .venv/bin/activate && python app.py"
