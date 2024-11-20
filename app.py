# app.py

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from config import DATABASE, TABLE_NAMES, NUM_DAYS_BACK
from database import get_connection
import pandas as pd
from logger import logger

app = dash.Dash(__name__)

def get_data(freq):
    engine = get_connection()
    table_name = TABLE_NAMES[freq]
    query = f"""
        SELECT timestamp, close, trend, sma, ema, predicted_close
        FROM {table_name}
        WHERE timestamp >= NOW() - INTERVAL '{NUM_DAYS_BACK} days'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, engine)
    return df

app.layout = html.Div([
    html.H1('Tesla Stock Price Trends'),
    dcc.Dropdown(
        id='frequency-dropdown',
        options=[{'label': freq, 'value': freq} for freq in TABLE_NAMES.keys()],
        value='1M'
    ),
    dcc.Graph(id='stock-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*60*1000,  # Update every minute
        n_intervals=0
    )
])

@app.callback(
    Output('stock-graph', 'figure'),
    [Input('frequency-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graph(freq, n):
    df = get_data(freq)
    if df.empty:
        return go.Figure()
    # Convert 'timestamp' to datetime with UTC timezone, then remove timezone info
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
    df.set_index('timestamp', inplace=True)

    fig = go.Figure()

    # Actual Close Price
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Actual Close'))

    # Predicted Close Price
    if 'predicted_close' in df.columns and df['predicted_close'].notnull().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['predicted_close'], mode='lines', name='Predicted Close'))
    else:
        # Do not log a warning, proceed without 'predicted_close'
        pass

    # Simple Moving Average (SMA)
    if 'sma' in df.columns and df['sma'].notnull().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['sma'], mode='lines', name='SMA'))

    # Exponential Moving Average (EMA)
    if 'ema' in df.columns and df['ema'].notnull().any():
        fig.add_trace(go.Scatter(x=df.index, y=df['ema'], mode='lines', name='EMA'))

    # Trends as Markers
    if 'trend' in df.columns and df['trend'].notnull().any():
        trend_markers = df[df['trend'] != 'STABLE']
        if not trend_markers.empty:
            fig.add_trace(go.Scatter(
                x=trend_markers.index,
                y=trend_markers['close'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=trend_markers['trend'].map({'UP': 'green', 'DOWN': 'red'}),
                    symbol='triangle-up'  # You can customize symbols based on trend
                ),
                name='Trends'
            ))

    fig.update_layout(title=f'Tesla Stock Price and Predicted Close ({freq})')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)