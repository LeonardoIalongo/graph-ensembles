#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error

# Define variables
BUCKET_NAME="dap"
FOLDER_NAME="corealgos/rmilocco/repo"
FILE_NAME="graph-ensembles"
ZIP_FILE="$FILE_NAME.zip"
TARGET_DIR="$HOME/data/$FOLDER_NAME"

# 2. Unzip the project
echo "📦 Unzipping ${TARGET_DIR}, $ZIP_FILE"
cd "${TARGET_DIR}"
unzip -o "${ZIP_FILE}"

# 3. Change directory to target project folder
echo "📁 Changing directory to ${FILE_NAME}"
cd "./${FILE_NAME}"

# 4. Install in editable mode using modern pip config
echo "⚙️ Installing package in editable mode (PEP 660)"
pip install --editable . --config-settings editable_mode=compat

echo "✅ Installation complete!"
