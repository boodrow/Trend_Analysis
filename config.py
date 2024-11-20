# config.py

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

NUM_DAYS_BACK = 90

GLOBAL_VARIABLES = {
    'LAST_PROCESSED_TIMESTAMP': {
        '1S': None,
        '1M': None,
        '5M': None,
        '15M': None,
        '30M': None,
        '1H': None,
        '1D': None,
    },
    'latest_logs': '',
}

TREND_PARAMETERS = {
    '1S': {'short_window': 5, 'long_window': 20},
    '1M': {'short_window': 10, 'long_window': 50},
    '5M': {'short_window': 10, 'long_window': 50},
    '15M': {'short_window': 10, 'long_window': 50},
    '30M': {'short_window': 5, 'long_window': 20},
    '1H': {'short_window': 5, 'long_window': 20},
    '1D': {'short_window': 5, 'long_window': 20},
}
