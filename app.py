"""Root WSGI entrypoint for deployment (Render/Railway/HuggingFace).
Exposes `server` for Gunicorn: `gunicorn app:server --workers 2 --timeout 120`
"""

from retail_demand_forecasting.dash_app.app import app

server = app.server  # Required for Gunicorn

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
