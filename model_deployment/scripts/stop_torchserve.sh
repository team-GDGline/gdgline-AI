#!/bin/bash

# Check Python version command
PYTHON_CMD=""

if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Python is not installed. Please install Python to proceed."
    exit 1
fi

# Debugging log: show which Python command is used
echo "Using Python command: $PYTHON_CMD"

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Debugging log: show the script directory and command being run
echo "Script directory: $SCRIPT_DIR"
echo "Running command: $PYTHON_CMD $SCRIPT_DIR/stop_torchserve.py"

# Run stop_torchserve.py
$PYTHON_CMD stop_torchserve.py

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "TorchServe stopped successfully."
else
    echo "Failed to stop TorchServe."
    exit 1
fi
