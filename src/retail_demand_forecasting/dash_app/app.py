"""Dash frontend for the Retail Demand Forecasting API.

Run with:
    python -m retail_demand_forecasting.dash_app.app

Then open http://localhost:8050 in your browser.
"""

import requests
import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    title="Retail Demand Forecasting",
    update_title="Predicting...",
    suppress_callback_exceptions=True,
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Roboto, sans-serif", "maxWidth": "960px", "margin": "0 auto", "padding": "24px"},
    children=[
        # Header
        html.Div(
            style={"textAlign": "center", "marginBottom": "32px"},
            children=[
                html.H1("Retail Demand Forecasting", style={"marginBottom": "4px"}),
                html.P("M5 Walmart dataset  |  PySpark + MLflow + FastAPI", style={"color": "#888"}),
            ],
        ),
        # Health badge
        html.Div(id="health-badge", style={"textAlign": "center", "marginBottom": "24px"}),
        # Tabs
        dcc.Tabs(
            id="tabs",
            value="single",
            children=[
                dcc.Tab(label="Single Prediction", value="single"),
                dcc.Tab(label="Batch Prediction", value="batch"),
                dcc.Tab(label="Model Metrics", value="metrics"),
            ],
        ),
        html.Div(id="tab-content", style={"marginTop": "24px"}),
    ],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
INPUT_FIELDS = [
    ("lag_7", "Lag 7", "number", 5),
    ("lag_28", "Lag 28", "number", 3),
    ("rolling_mean_7", "Rolling Mean 7", "number", 4.5),
    ("rolling_mean_28", "Rolling Mean 28", "number", 3.2),
    ("day_of_week", "Day of Week (1-7)", "number", 6),
    ("month", "Month", "number", 4),
    ("year", "Year", "number", 2016),
    ("snap_CA", "SNAP CA", "dropdown", 0),
    ("snap_TX", "SNAP TX", "dropdown", 0),
    ("snap_WI", "SNAP WI", "dropdown", 0),
    ("has_event_1", "Event 1", "dropdown", 0),
    ("has_event_2", "Event 2", "dropdown", 0),
    ("sell_price", "Sell Price", "number", 1.25),
]


def _make_input_field(field_id, label, input_type, default):
    if input_type == "dropdown":
        return html.Div(
            style={"marginBottom": "12px"},
            children=[
                html.Label(label, style={"fontWeight": "600", "fontSize": "13px"}),
                dcc.Dropdown(
                    id=field_id,
                    options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                    value=default,
                    clearable=False,
                    style={"marginTop": "4px"},
                ),
            ],
        )
    return html.Div(
        style={"marginBottom": "12px"},
        children=[
            html.Label(label, style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Input(
                id=field_id,
                type=number,
                value=default,
                debounce=True,
                style={"width": "100%", "padding": "6px", "marginTop": "4px", "borderRadius": "4px", "border": "1px solid #ccc"},
            ),
        ],
    )


def _build_input_form(prefix=""):
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
        children=[_make_input_field(f"{prefix}{fid}", lbl, typ, default) for fid, lbl, typ, default in INPUT_FIELDS],
    )


def _collect_payload(prefix=""):
    """Build a JS-style dict string for the callback (Dash handles via ctx)."""
    return {f"{prefix}{fid}": fid if prefix else fid for fid, *_ in INPUT_FIELDS}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(Output("health-badge", "children"), Input("tabs", "value"))
def check_health(_):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        data = r.json()
        color = "#2ecc71" if data.get("model_loaded") else "#e67e22"
        text = "Model loaded" if data.get("model_loaded") else "Model NOT loaded"
    except Exception:
        color = "#e74c3c"
        text = "API unreachable"
    return html.Div(
        style={
            "display": "inline-block", "padding": "6px 16px", "borderRadius": "12px",
            "backgroundColor": color, "color": "white", "fontWeight": "600", "fontSize": "13px",
        },
        children=text,
    )


@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "single":
        return _render_single_tab()
    elif tab == "batch":
        return _render_batch_tab()
    elif tab == "metrics":
        return _render_metrics_tab()
    return html.Div()


# ---- Single prediction tab ----
def _render_single_tab():
    return html.Div([
        html.H3("Single Prediction"),
        html.P("Fill in the features and click Predict.", style={"color": "#666"}),
        _build_input_form(),
        html.Button(
            "Predict",
            id="predict-btn",
            n_clicks=0,
            style={
                "marginTop": "16px", "padding": "10px 32px", "fontSize": "15px",
                "backgroundColor": "#3498db", "color": "white", "border": "none",
                "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
            },
        ),
        html.Div(id="single-result", style={"marginTop": "24px"}),
    ])


@callback(
    Output("single-result", "children"),
    Input("predict-btn", "n_clicks"),
    [State(fid, "value") for fid, *_ in INPUT_FIELDS],
    prevent_initial_call=True,
)
def do_predict(_, *values):
    payload = {fid: val for (fid, *_), val in zip(INPUT_FIELDS, values)}
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        data = r.json()
        if "error" in data:
            return html.Div(data["error"], style={"color": "#e74c3c", "fontWeight": "600"})
        pred = data["prediction"]
        return html.Div(
            style={
                "padding": "20px", "borderRadius": "8px", "backgroundColor": "#ecf0f1",
                "textAlign": "center",
            },
            children=[
                html.H2(f"{pred:.2f}", style={"margin": "0", "color": "#2c3e50"}),
                html.P("Predicted daily unit sales", style={"margin": "4px 0 0 0", "color": "#888"}),
            ],
        )
    except Exception as e:
        return html.Div(f"Error: {e}", style={"color": "#e74c3c"})


# ---- Batch prediction tab ----
def _render_batch_tab():
    return html.Div([
        html.H3("Batch Prediction"),
        html.P("Preview multiple scenarios side-by-side.", style={"color": "#666"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px"},
            children=[
                html.Div([
                    html.H4("Scenario A", style={"marginTop": "0"}),
                    _build_input_form(prefix="a_"),
                ]),
                html.Div([
                    html.H4("Scenario B", style={"marginTop": "0"}),
                    _build_input_form(prefix="b_"),
                ]),
            ],
        ),
        html.Button(
            "Compare",
            id="batch-btn",
            n_clicks=0,
            style={
                "marginTop": "16px", "padding": "10px 32px", "fontSize": "15px",
                "backgroundColor": "#9b59b6", "color": "white", "border": "none",
                "borderRadius": "6px", "cursor": "pointer", "fontWeight": "600",
            },
        ),
        html.Div(id="batch-result", style={"marginTop": "24px"}),
    ])


ALL_BATCH_FIELDS = []
for fid, lbl, typ, default in INPUT_FIELDS:
    ALL_BATCH_FIELDS.append((f"a_{fid}", lbl, typ, default))
    ALL_BATCH_FIELDS.append((f"b_{fid}", lbl, typ, default))


@callback(
    Output("batch-result", "children"),
    Input("batch-btn", "n_clicks"),
    [State(afid, "value") for afid, *_ in ALL_BATCH_FIELDS],
    prevent_initial_call=True,
)
def do_batch(_, *values):
    a_payload = {}
    b_payload = {}
    for i, (fid, *_) in enumerate(INPUT_FIELDS):
        a_payload[fid] = values[i * 2]
        b_payload[fid] = values[i * 2 + 1]

    try:
        r = requests.post(
            f"{API_BASE}/predict/batch",
            json={"requests": [a_payload, b_payload]},
            timeout=10,
        )
        data = r.json()
        preds = data.get("predictions", [0, 0])

        fig = go.Figure(go.Bar(
            x=["Scenario A", "Scenario B"],
            y=preds,
            marker_color=["#3498db", "#9b59b6"],
            text=[f"{p:.2f}" for p in preds],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Predicted Sales",
            template="plotly_white",
            margin=dict(t=20, b=40, l=40, r=20),
            height=320,
        )

        return dcc.Graph(figure=fig)
    except Exception as e:
        return html.Div(f"Error: {e}", style={"color": "#e74c3c"})


# ---- Metrics tab ----
def _render_metrics_tab():
    return html.Div([
        html.H3("Model Metrics"),
        html.P("Results from the latest MLflow run.", style={"color": "#666"}),
        html.Button(
            "Refresh",
            id="metrics-btn",
            n_clicks=0,
            style={
                "marginBottom": "16px", "padding": "8px 20px", "fontSize": "14px",
                "backgroundColor": "#27ae60", "color": "white", "border": "none",
                "borderRadius": "6px", "cursor": "pointer",
            },
        ),
        html.Div(id="metrics-content"),
    ])


@callback(
    Output("metrics-content", "children"),
    Input("metrics-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_metrics(_):
    try:
        import mlflow
        from retail_demand_forecasting.nodes.constants import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=["0", "1"], order_by=["start_time DESC"], max_results=1)
        if not runs:
            return html.Div("No runs found.", style={"color": "#e74c3c"})
        run = runs[0]
        data = run.data
        mae = data.metrics.get("mae", "N/A")
        rmse = data.metrics.get("rmse", "N/A")
        mape = data.metrics.get("mape", "N/A")

        def _metric_card(title, value, color):
            return html.Div(
                style={
                    "padding": "20px", "borderRadius": "8px", "backgroundColor": color,
                    "color": "white", "textAlign": "center", "flex": "1",
                },
                children=[
                    html.Div(f"{value:.4f}" if isinstance(value, float) else str(value), style={"fontSize": "28px", "fontWeight": "700"}),
                    html.Div(title, style={"fontSize": "13px", "marginTop": "4px", "opacity": 0.9}),
                ],
            )

        return html.Div(
            style={"display": "flex", "gap": "16px"},
            children=[
                _metric_card("MAE", mae, "#3498db"),
                _metric_card("RMSE", rmse, "#e67e22"),
                _metric_card("MAPE", f"{mape:.1f}%", "#9b59b6"),
            ],
        )
    except Exception as e:
        return html.Div(f"Error loading metrics: {e}", style={"color": "#e74c3c"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
