# app.py

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from config import DATABASE, TABLE_NAMES, NUM_DAYS_BACK
from database import get_connection
import pandas as pd
from logger import logger
import os
import numpy as np

# Initialize Dash app
app = dash.Dash(__name__)


def get_data(freq):
    """
    Fetch data from the specified table based on frequency.
    Ensures data integrity by sorting and removing duplicates.
    Converts relevant columns to numeric types.
    """
    engine = get_connection()
    table_name = TABLE_NAMES[freq]
    query = f"""
        SELECT timestamp, close, trend, sma, ema, predicted_close
        FROM {table_name}
        WHERE timestamp >= NOW() - INTERVAL '{NUM_DAYS_BACK} days'
        ORDER BY timestamp ASC
    """
    try:
        df = pd.read_sql(query, engine, parse_dates=['timestamp'])
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        else:
            logger.error(f"Fetched DataFrame from {table_name} does not contain 'timestamp' column.")
            return pd.DataFrame()

        # Ensure data is sorted by timestamp
        df.sort_index(inplace=True)

        # Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate timestamps in {table_name}. Removing duplicates.")
            df = df[~df.index.duplicated(keep='last')]

        # Ensure 'sma', 'ema', 'predicted_close' are numeric
        for col in ['sma', 'ema', 'predicted_close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                logger.warning(f"Column '{col}' not found in {table_name}.")

        # Log data integrity
        logger.debug(f"DataFrame for {freq}:\n{df.head()}")
        logger.debug(f"Data types for {freq}:\n{df.dtypes}")
        logger.debug(f"Number of NaNs in 'sma': {df['sma'].isna().sum()}")
        logger.debug(f"Number of NaNs in 'ema': {df['ema'].isna().sum()}")
        logger.debug(f"Number of NaNs in 'predicted_close': {df['predicted_close'].isna().sum()}")

        return df
    except Exception as e:
        logger.error(f"Error fetching data for frequency '{freq}': {e}")
        return pd.DataFrame()


# Define the layout of the Dash app
app.layout = html.Div([
    html.H1('Tesla Stock Price Trends'),

    # Trend Indicators for Each Frequency
    html.Div([
        html.Div([
            html.H3(f"{freq} Current Trend"),
            html.Div(id=f"{freq}-trend", style={'fontSize': 20, 'color': 'black'})
        ], style={
            'display': 'inline-block',
            'margin': '10px',
            'padding': '10px',
            'border': '1px solid black',
            'borderRadius': '5px',
            'width': '150px',
            'textAlign': 'center'
        })
        for freq in TABLE_NAMES.keys()
    ], style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'}),

    # Frequency Selection Dropdown
    html.Div([
        html.Label('Select Frequency:', style={'fontSize': 18, 'marginRight': '10px'}),
        dcc.Dropdown(
            id='frequency-dropdown',
            options=[{'label': freq, 'value': freq} for freq in TABLE_NAMES.keys()],
            value='1M',
            style={'width': '200px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginTop': '20px'}),

    # Stock Price Graph
    dcc.Graph(id='stock-graph'),

    # Interval Component for Live Updates
    dcc.Interval(
        id='interval-component',
        interval=1 * 60 * 1000,  # Update every minute
        n_intervals=0
    )
], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})


@app.callback(
    [Output(f"{freq}-trend", 'children') for freq in TABLE_NAMES.keys()] +
    [Output('stock-graph', 'figure')],
    [Input('frequency-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(selected_freq, n):
    """
    Callback function to update trend indicators and the stock graph based on selected frequency and interval.
    """
    engine = get_connection()
    trend_texts = []

    # Fetch and display the latest trend for each frequency
    for freq, table in TABLE_NAMES.items():
        query = f"""
            SELECT trend FROM {table}
            ORDER BY timestamp DESC LIMIT 1
        """
        try:
            df_trend = pd.read_sql(query, engine, parse_dates=['timestamp'])
            if not df_trend.empty:
                latest_trend = df_trend['trend'].iloc[0]
                if pd.isna(latest_trend):
                    trend_text = html.Div('STABLE', style={'color': 'gray'})
                elif latest_trend.upper() == 'UP':
                    trend_text = html.Div('UP', style={'color': 'green'})
                elif latest_trend.upper() == 'DOWN':
                    trend_text = html.Div('DOWN', style={'color': 'red'})
                else:
                    trend_text = html.Div('UNKNOWN', style={'color': 'black'})
            else:
                trend_text = html.Div('No Data', style={'color': 'black'})
        except Exception as e:
            logger.error(f"Error fetching latest trend for {freq}: {e}")
            trend_text = html.Div('Error', style={'color': 'black'})
        trend_texts.append(trend_text)

    # Fetch data for the selected frequency to plot
    df_plot = get_data(selected_freq)

    # Create the graph
    if not df_plot.empty:
        # Log the first few rows to verify data integrity
        logger.debug(f"First few rows of data for {selected_freq}:\n{df_plot.head()}")

        # Check if 'sma', 'ema', 'predicted_close' have sufficient non-NaN values
        sma_available = df_plot['sma'].notna().sum() > 1
        ema_available = df_plot['ema'].notna().sum() > 1
        predicted_close_available = df_plot['predicted_close'].notna().sum() > 1

        if not sma_available:
            logger.warning(f"Not enough data points to plot 'SMA' for frequency '{selected_freq}'.")
        if not ema_available:
            logger.warning(f"Not enough data points to plot 'EMA' for frequency '{selected_freq}'.")
        if not predicted_close_available:
            logger.warning(f"Not enough data points to plot 'Predicted Close' for frequency '{selected_freq}'.")

        # Always display 'close' prices
        trace_actual = go.Scatter(
            x=df_plot.index,
            y=df_plot['close'],
            mode='lines',
            name='Actual Close',
            line=dict(color='blue')
        )

        # Plot 'predicted_close' if available
        trace_predicted = go.Scatter(
            x=df_plot.index,
            y=df_plot['predicted_close'],
            mode='lines',
            name='Predicted Close',
            line=dict(color='orange'),
            visible='legendonly'  # Optional: Hide by default
        ) if predicted_close_available else None

        # Plot 'sma' if available
        trace_sma = go.Scatter(
            x=df_plot.index,
            y=df_plot['sma'],
            mode='lines',
            name='SMA',
            line=dict(color='green', dash='dash'),
            visible='legendonly'  # Optional: Hide by default
        ) if sma_available else None

        # Plot 'ema' if available
        trace_ema = go.Scatter(
            x=df_plot.index,
            y=df_plot['ema'],
            mode='lines',
            name='EMA',
            line=dict(color='purple', dash='dot'),
            visible='legendonly'  # Optional: Hide by default
        ) if ema_available else None

        # Initialize data list with mandatory 'close' trace
        data = [trace_actual]

        # Append optional traces if they exist
        if trace_predicted:
            data.append(trace_predicted)
        if trace_sma:
            data.append(trace_sma)
        if trace_ema:
            data.append(trace_ema)

        # Add trend markers only where 'trend' is 'UP' or 'DOWN'
        trace_trend_up = go.Scatter(
            x=df_plot[df_plot['trend'].str.upper() == 'UP'].index,
            y=df_plot[df_plot['trend'].str.upper() == 'UP']['close'],
            mode='markers',
            name='UP Trend',
            marker=dict(color='green', size=10, symbol='triangle-up')
        )
        trace_trend_down = go.Scatter(
            x=df_plot[df_plot['trend'].str.upper() == 'DOWN'].index,
            y=df_plot[df_plot['trend'].str.upper() == 'DOWN']['close'],
            mode='markers',
            name='DOWN Trend',
            marker=dict(color='red', size=10, symbol='triangle-down')
        )
        data.extend([trace_trend_up, trace_trend_down])

    else:
        # If no data is available, display an empty graph with a message
        data = []

    layout = go.Layout(
        title=f"Tesla Stock Price and Predicted Close ({selected_freq})",
        xaxis={'title': 'Timestamp'},
        yaxis={'title': 'Price'},
        hovermode='closest',
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40}
    )

    figure = {'data': data, 'layout': layout}

    return trend_texts + [figure]


if __name__ == '__main__':
    # Ensure model directory exists
    os.makedirs('models', exist_ok=True)
    # Run the Dash app
    app.run_server(debug=True)