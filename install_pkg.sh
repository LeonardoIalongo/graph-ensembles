#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error 

echo "➡️ Changing directory to /home/inghero/riccardo/"
cd /home/inghero/riccardo/

echo "📁 Unzipping graph-ensembles.zip"
if unzip -o graph-ensembles.zip; then
    echo "📁 Unzipped graph-ensembles.zip successfully."
else
    echo "❌ Unzip failed. Remove old graph-ensembles directory with mc"
    echo "❌ Please run: mc rm -rf graph-ensembles"
    exit 1
fi

echo "➡️ Changing directory to graph-ensembles"
cd graph-ensembles

echo "🛑 Deactivating virtual environment"
deactivate

echo "⚙️ Installing package in editable mode (PEP 660)"
pip install --editable . --config-settings editable_mode=compat

echo "✅ Installation complete!"