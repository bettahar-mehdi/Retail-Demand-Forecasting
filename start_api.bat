@echo off
REM Start FastAPI backend — portable, uses repo-relative paths
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
set PYTHONPATH=%~dp0src;%PYTHONPATH%
"%PY%" -m uvicorn retail_demand_forecasting.api.app:app --host 127.0.0.1 --port 8000 --reload
