"""Launch both the FastAPI and Dash servers for development."""
import subprocess
import sys

PYTHON = sys.executable
CWD = r"D:\Retail Demand Forecasting"

api = subprocess.Popen(
    [PYTHON, "-m", "uvicorn",
     "retail_demand_forecasting.api.app:app",
     "--host", "127.0.0.1", "--port", "8000"],
    cwd=CWD,
)
print(f"API server started (PID {api.pid}) on http://127.0.0.1:8000")

dash = subprocess.Popen(
    [PYTHON, "-m", "retail_demand_forecasting.dash_app.app"],
    cwd=CWD,
)
print(f"Dash app started (PID {dash.pid}) on http://127.0.0.1:8050")

print("\nPress Ctrl+C to stop both servers.")
try:
    api.wait()
except KeyboardInterrupt:
    api.terminate()
    dash.terminate()
    print("\nServers stopped.")
