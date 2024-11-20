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

NUM_DAYS_BACK = 1

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
    '1S': {'window_size': 20},
    '1M': {'window_size': 10},
    '5M': {'window_size': 10},
    '15M': {'window_size': 10},
    '30M': {'window_size': 5},
    '1H': {'window_size': 5},
    '1D': {'window_size': 5}
}
