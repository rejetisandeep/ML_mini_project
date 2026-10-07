@echo off
:: Windows Batch Launcher for BLE Indoor Position Prediction App
:: Works on any Windows system regardless of PATH configuration.

echo ============================================================
echo  BLE Indoor Position Prediction - Web App Launcher
echo ============================================================

:: Find Python - check common locations
where python >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON_CMD=python
    goto :run
)

where python3 >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON_CMD=python3
    goto :run
)

echo ERROR: Python not found. Please install Python from https://python.org
pause
exit /b 1

:run
%PYTHON_CMD% run_app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo If streamlit is not installed, run: %PYTHON_CMD% -m pip install streamlit
    pause
)
