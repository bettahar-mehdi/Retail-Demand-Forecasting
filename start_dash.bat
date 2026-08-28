@echo off
REM Start Dash frontend — portable
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
set PYTHONPATH=%~dp0src;%PYTHONPATH%
"%PY%" -m retail_demand_forecasting.dash_app.app
