#!/bin/bash

# Stop if anything fails
set -e

echo "🔧 Creating virtual environment in .venv..."
python3 -m venv .venv

echo "🚀 Activating environment and installing requirements..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete. To activate: source .venv/bin/activate"
