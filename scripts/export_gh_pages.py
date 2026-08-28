"""Export static Plotly demo for GitHub Pages (no Python backend needed)."""

import os
import pandas as pd
import plotly.graph_objects as go


def build_static_demo():
    os.makedirs("docs", exist_ok=True)

    # Load sample historical and forecast data
    # (Replace with your actual sample data loader)
    dates_hist = pd.date_range(end=pd.Timestamp.today(), periods=90)
    dates_fc = pd.date_range(start=dates_hist[-1] + pd.Timedelta(days=1), periods=60)

    fig = go.Figure()

    # Historical trace
    fig.add_trace(
        go.Scatter(
            x=dates_hist,
            y=[0, 1, 0, 2, 0, 0, 3, 1, 0, 2] * 9,
            mode="lines",
            name="Historical Sales",
            line=dict(color="#3b82f6"),
        )
    )

    # Forecast trace
    fig.add_trace(
        go.Scatter(
            x=dates_fc,
            y=[0.85 * (0.98**i) for i in range(60)],
            mode="lines+markers",
            name="Forecast (Expected Rate/Day)",
            line=dict(color="#ef4444", dash="dash"),
        )
    )

    fig.update_layout(
        title="Retail Demand Forecasting — Interactive Demo (M5 Walmart)",
        xaxis_title="Date",
        yaxis_title="Units Sold / Expected Rate",
        template="plotly_white",
    )

    fig.write_html("docs/index.html", include_plotlyjs="cdn")
    print("Static GitHub Pages demo created at docs/index.html")


if __name__ == "__main__":
    build_static_demo()
