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
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    else:
        logger.error(f"Fetched DataFrame from {table_name} does not contain 'timestamp' column.")
        return pd.DataFrame()
    return df

app.layout = html.Div([
    html.H1('Tesla Stock Price Trends'),
    html.Div([
        html.Div([
            html.H3(f"{freq} Current Trend"),
            html.Div(id=f"{freq}-trend", style={'fontSize': 20, 'color': 'black'})
        ], style={'display': 'inline-block', 'margin': '10px', 'padding': '10px', 'border': '1px solid black'})
        for freq in TABLE_NAMES.keys()
    ]),
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
    [Output(f"{freq}-trend", 'children') for freq in TABLE_NAMES.keys()] +
    [Output('stock-graph', 'figure')],
    [Input('frequency-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(selected_freq, n):
    engine = get_connection()
    trend_texts = []
    for freq, table in TABLE_NAMES.items():
        query = f"""
            SELECT trend FROM {table}
            ORDER BY timestamp DESC LIMIT 1
        """
        try:
            df_trend = pd.read_sql(query, engine, parse_dates=['timestamp'])
            if not df_trend.empty:
                latest_trend = df_trend['trend'].iloc[0]
                if latest_trend == 'UP':
                    trend_text = html.Div('UP', style={'color': 'green'})
                elif latest_trend == 'DOWN':
                    trend_text = html.Div('DOWN', style={'color': 'red'})
                else:
                    trend_text = html.Div('', style={'color': 'black'})  # Exclude 'STABLE' or NaN
            else:
                trend_text = html.Div('', style={'color': 'black'})
        except Exception as e:
            logger.error(f"Error fetching latest trend for {freq}: {e}")
            trend_text = html.Div('', style={'color': 'black'})
        trend_texts.append(trend_text)

    # Fetch data for the selected frequency to plot
    df_plot = get_data(selected_freq)

    # Create the graph
    if not df_plot.empty:
        # Exclude rows where 'trend' is NaN (i.e., 'STABLE' trends)
        df_plot = df_plot.dropna(subset=['trend'])

        trace_actual = go.Scatter(
            x=df_plot.index,
            y=df_plot['close'],
            mode='lines',
            name='Actual Close',
            line=dict(color='blue')
        )
        trace_predicted = go.Scatter(
            x=df_plot.index,
            y=df_plot['predicted_close'],
            mode='lines',
            name='Predicted Close',
            line=dict(color='orange')
        )
        trace_sma = go.Scatter(
            x=df_plot.index,
            y=df_plot['sma'],
            mode='lines',
            name='SMA',
            line=dict(color='green', dash='dash')
        )
        trace_ema = go.Scatter(
            x=df_plot.index,
            y=df_plot['ema'],
            mode='lines',
            name='EMA',
            line=dict(color='purple', dash='dot')
        )
        # Trends as Markers
        trace_trend_up = go.Scatter(
            x=df_plot[df_plot['trend'] == 'UP'].index,
            y=df_plot[df_plot['trend'] == 'UP']['close'],
            mode='markers',
            name='UP Trend',
            marker=dict(color='green', size=10, symbol='triangle-up')
        )
        trace_trend_down = go.Scatter(
            x=df_plot[df_plot['trend'] == 'DOWN'].index,
            y=df_plot[df_plot['trend'] == 'DOWN']['close'],
            mode='markers',
            name='DOWN Trend',
            marker=dict(color='red', size=10, symbol='triangle-down')
        )
        data = [trace_actual, trace_predicted, trace_sma, trace_ema, trace_trend_up, trace_trend_down]
    else:
        data = []

    layout = go.Layout(
        title=f"Tesla Stock Price and Predicted Close ({selected_freq})",
        xaxis={'title': 'Timestamp'},
        yaxis={'title': 'Price'},
        hovermode='closest'
    )

    figure = {'data': data, 'layout': layout}

    return trend_texts + [figure]

if __name__ == '__main__':
    app.run_server(debug=True)