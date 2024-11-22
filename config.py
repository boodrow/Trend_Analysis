# config.py

import json
import os

DATABASE = {
    'HOST': '192.168.0.102',
    'PORT': '5432',
    'USER': 'trishdbuser',
    'PASSWORD': '1R@dp1mp',
    'NAME': 'trish_tsla_ohlc',
}

TABLE_NAMES = {
    '1S': 'actual_tsla_1s',
    '1M': 'actual_tsla_1m',
    '5M': 'actual_tsla_5m',
    '15M': 'actual_tsla_15m',
    '30M': 'actual_tsla_30m',
    '1H': 'actual_tsla_1h',
    '1D': 'actual_tsla_1d',
}

WEB_APP = {
    'PORT': 8050,
}

NUM_DAYS_BACK = 180
NUM_OF_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 10
TTT_INTERVAL = 300
PERFORM_TTT = False
FLOAT_TOLERANCE = 1e-4
BATCH_UPDATE = 100 # Batch size for database updates

GLOBAL_VARIABLES = {
    'latest_logs': '',
}

TREND_PARAMETERS = {
    '1S': {'short_window': 5, 'long_window': 20, 'num_layers': 2, 'hidden_size': 64, 'num_transformer_layers': 1},
    '1M': {'short_window': 10, 'long_window': 50, 'num_layers': 2, 'hidden_size': 64, 'num_transformer_layers': 1},
    '5M': {'short_window': 10, 'long_window': 50, 'num_layers': 2, 'hidden_size': 64, 'num_transformer_layers': 1},
    '15M': {'short_window': 10, 'long_window': 50, 'num_layers': 2, 'hidden_size': 64, 'num_transformer_layers': 1},
    '30M': {'short_window': 5, 'long_window': 20, 'num_layers': 2, 'hidden_size': 64, 'num_transformer_layers': 1},
    '1H': {'short_window': 5, 'long_window': 20, 'num_layers': 2, 'hidden_size': 64, 'num_transformer_layers': 1},
    '1D': {'short_window': 5, 'long_window': 20, 'num_layers': 2, 'hidden_size': 64, 'num_transformer_layers': 1},
}

# Directory to store last processed timestamps
TIMESTAMP_DIR = 'last_processed_timestamps'
os.makedirs(TIMESTAMP_DIR, exist_ok=True)