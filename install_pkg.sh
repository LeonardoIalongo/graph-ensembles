#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error 

echo "🛑 Deactivating virtual environment"
deactivate

echo "⚙️ Installing package in editable mode (PEP 660)"
pip install --editable . --config-settings editable_mode=compat

echo "✅ Installation complete!"