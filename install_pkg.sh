#!/bin/bash 

echo "🛑 Deactivating virtual environment"
deactivate

echo "⚙️ Installing package in editable mode (PEP 660)"
pip install --editable . --config-settings editable_mode=compat

echo "✅ Installation complete!"