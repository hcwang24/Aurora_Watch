import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Import the load function
from dataLoading import load_and_clean_kpindex_3hr

# Load initial data
kp_data = load_and_clean_kpindex_3hr(end_time=(datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d"))

# Initialize the app with a dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Default date range for display (last 3 days and next 3 days)
default_start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
default_end_date = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')

def round_to_nearest_3hr():
    now = datetime.now().astimezone(pytz.UTC)
    hours_to_adjust = now.hour % 3

    if hours_to_adjust >= 1.5:
        adjusted_time = now + timedelta(hours=(3 - hours_to_adjust))
    else:
        adjusted_time = now - timedelta(hours=hours_to_adjust)

    # Zero out minutes and seconds
    adjusted_time = adjusted_time.replace(minute=0, second=0, microsecond=0)

    return adjusted_time

rounded_time = round_to_nearest_3hr().strftime('%Y-%m-%d %H:%M:%S%z')

# Timezone options
timezones = pytz.all_timezones
timezone_options = [{"label": tz, "value": tz} for tz in timezones]

# Sidebar layout
sidebar = dbc.Collapse(
    [
        dbc.Row(dbc.Label("Filter by Date Range")),
        dcc.DatePickerRange(
            id='date-picker',
            start_date=default_start_date,
            end_date=default_end_date
        ),
        dbc.Row(dbc.Label("Choose Location")),
        dcc.Dropdown(
            id='location-dropdown',
            options=[{'label': 'Location 1', 'value': 'loc1'}, {'label': 'Location 2', 'value': 'loc2'}],
            placeholder="Select Location"
        ),
        dbc.Row(dbc.Label("Select Timezone")),
        dcc.Dropdown(
            id='timezone-dropdown',
            options=timezone_options,
            value="UTC",  # Default to UTC
            placeholder="Select Timezone"
        ),
        dbc.Button("Use Current Location", id="current-location-button", n_clicks=0),
        dbc.Checklist(
            options=[{"label": "Show Highest Point", "value": 1}],
            id="peak-toggle",
            switch=True,
        )
    ],
    id="sidebar",
    is_open=True,
)

# Stats Boxes
stats_boxes = dbc.Row(
    [
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Current Kp Index"), html.P(id="current-kp")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Next Predicted Kp Index"), html.P(id="predicted-kp")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Last High Kp Date (>= 6)"), html.P(id="last-high-kp-date")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Current Weather"), html.P(id="current-weather")])), width=3)
    ]
)

# Time-Series Plot
time_series_plot = dcc.Graph(id='kp-time-series', config={'displayModeBar': False})

# Layout of the app
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.Div([sidebar]), width=3, id='sidebar-col'),
                dbc.Col(
                    [
                        stats_boxes,
                        time_series_plot,
                        html.Div(id="map-or-weather-area")
                    ],
                    width=9
                ),
            ]
        )
    ],
    fluid=True,
    style={'backgroundColor': '#1a1a1a'}  # Aurora dark theme background
)

# Callback to update stats and graph based on selected date range and timezone
@app.callback(
    [
        Output("current-kp", "children"),
        Output("predicted-kp", "children"),
        Output("last-high-kp-date", "children"),
        Output("kp-time-series", "figure"),
    ],
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("timezone-dropdown", "value"),
        Input("peak-toggle", "value")
    ]
)
def update_dashboard(start_date, end_date, timezone, show_peak):
    # Filter data based on selected date range
    filtered_data = kp_data[(kp_data['Datetime'] >= start_date) & (kp_data['Datetime'] <= end_date)]

    # Get the current Kp Index
    current_kp = kp_data[kp_data['Datetime']==rounded_time]['Kp_Index'] if not filtered_data.empty else "N/A"
    
    # Convert 'Datetime' column to the selected timezone
    filtered_data['Datetime'] = filtered_data['Datetime'].dt.tz_convert(timezone)

    # Update stats boxes
    predicted_kp = np.nan  # Placeholder - replace with prediction logic
    last_high_kp_date = filtered_data[filtered_data["Kp_Index"] >= 6]["Datetime"].max()

    # Time-series plot
    fig = px.line(filtered_data, x="Datetime", y="Kp_Index", title="Kp Index Over Time")
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Kp Index",
        yaxis_range=[0, 9],
        margin=dict(t=40, b=20)
    )

    # Highlight the highest point if toggle is active
    if show_peak:
        highest_point = filtered_data.loc[filtered_data["Kp_Index"].idxmax()]
        fig.add_scatter(
            x=[highest_point["Datetime"]],
            y=[highest_point["Kp_Index"]],
            mode="markers",
            marker=dict(size=10, color="lime", symbol="star"),
            name="Peak Kp Index"
        )

    return current_kp, predicted_kp, last_high_kp_date, fig

if __name__ == '__main__':
    app.run_server(debug=True)
