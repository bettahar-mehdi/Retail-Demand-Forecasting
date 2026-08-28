"""Start both API and Dash servers — portable, no hardcoded absolute paths."""

import subprocess
import sys
import os
import time
from pathlib import Path

# Resolve repo root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
# Ensure src is on PYTHONPATH
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
os.environ["PYTHONPATH"] = src_path + os.pathsep + os.environ.get("PYTHONPATH", "")

# Use current venv python
PY = sys.executable

api = subprocess.Popen(
    [PY, "-m", "uvicorn", "retail_demand_forecasting.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=str(PROJECT_ROOT),
)
print(f"API: http://127.0.0.1:8000  (PID {api.pid})")

dash = subprocess.Popen(
    [PY, "-m", "retail_demand_forecasting.dash_app.app"],
    cwd=str(PROJECT_ROOT),
)
print(f"Dashboard: http://127.0.0.1:8050  (PID {dash.pid})")
print("\nPress Ctrl+C to stop both servers.")

try:
    api.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    for proc in (api, dash):
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    print("Stopped.")
