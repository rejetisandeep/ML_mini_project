#!/usr/bin/env bash
# macOS/Linux Shell Launcher for BLE Indoor Position Prediction App
# Works on any Unix system regardless of PATH configuration.

echo "============================================================"
echo " BLE Indoor Position Prediction - Web App Launcher"
echo "============================================================"

# Detect Python command
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python not found. Please install Python from https://python.org"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"
echo "Starting app at: http://localhost:8501"
echo "Press Ctrl+C to stop."
echo ""

$PYTHON_CMD run_app.py
