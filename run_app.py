"""
Universal Cross-Platform App Launcher for BLE Indoor Position Prediction.
Uses 'python -m streamlit' instead of the 'streamlit' command directly, so it
works on any system regardless of PATH configuration or OS.
"""

import sys
import os
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(PROJECT_DIR, "app.py")

def main():
    print("=" * 60)
    print("BLE Indoor Position Prediction - Web App Launcher")
    print("=" * 60)
    print(f"Python:  {sys.executable}")
    print(f"App:     {APP_PATH}")
    print(f"URL:     http://localhost:8501")
    print("=" * 60)
    print("Opening browser... Press Ctrl+C to stop the server.\n")

    cmd = [
        sys.executable,          # Uses the exact Python that ran THIS script
        "-m", "streamlit",
        "run", APP_PATH,
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApp server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nError: {e}")
        print("Make sure streamlit is installed: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
