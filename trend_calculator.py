# trend_calculator.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from logger import logger
from config import TREND_PARAMETERS
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModelWithAttention(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, bidirectional=True):
        """
        input_size: Number of input features per timestep.
        hidden_size: Number of features in the hidden state.
        num_layers: Number of recurrent layers.
        bidirectional: If True, use bidirectional LSTM.
        """
        super(LSTMModelWithAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=bidirectional
        )
        direction_multiplier = 2 if bidirectional else 1
        self.attention = nn.Linear(hidden_size * direction_multiplier, 1)
        self.fc = nn.Linear(hidden_size * direction_multiplier, 1)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        lstm_out, _ = self.lstm(x)  # [batch_size, sequence_length, hidden_size * direction_multiplier]
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch_size, sequence_length, 1]
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size * direction_multiplier]
        out = self.fc(context)  # [batch_size, 1]
        return out


def calculate_technical_indicators(df):
    df['sma'] = df['close'].rolling(window=5).mean()
    df['ema'] = df['close'].ewm(span=5, adjust=False).mean()
    return df


def train_model(model, train_loader, val_loader, model_path, epochs=20, device='cpu', early_stopping_patience=5):
    """
    Train the model and save the best performing model based on validation loss.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        logger.info(f"Training Epoch [{epoch + 1}/{epochs}]")
        with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] (Training)", leave=False) as t:
            for x, y in t:
                x = x.float().to(device)
                y = y.float().to(device)

                if len(y.shape) == 1:
                    y = y.unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                t.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Training Average Loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{epochs}] (Validation)", leave=False) as t_val:
                for x_val, y_val in t_val:
                    x_val = x_val.float().to(device)
                    y_val = y_val.float().to(device)

                    if len(y_val.shape) == 1:
                        y_val = y_val.unsqueeze(1)

                    outputs_val = model(x_val)
                    loss_val = criterion(outputs_val, y_val)
                    val_loss += loss_val.item()

                    t_val.set_postfix(loss=loss_val.item())

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Validation Average Loss: {avg_val_loss:.6f}")

        # Early Stopping Check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            # Save the best model whenever a new best loss is achieved
            torch.save(best_model_state, model_path)
            logger.info(f"New best model saved with validation loss {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                if 'best_model_state' in locals():
                    model.load_state_dict(best_model_state)
                break

        model.train()


def get_sequence_length(frequency):
    """
    Determine the sequence length based on the frequency.
    """
    if frequency == '1S':
        return 20
    elif frequency in ['1M', '5M', '15M']:
        return 10
    else:
        return 5  # For '30M', '1H', '1D'


def aggregate_lower_freq_data(df_main, lower_freq_df, frequency):
    """
    Aggregates lower frequency trend data to create additional features.
    """
    if lower_freq_df is None or lower_freq_df.empty:
        # Return zeros or NaNs for additional features if no lower frequency data is available
        df_main['lower_up_count'] = 0
        df_main['lower_down_count'] = 0
        df_main['lower_avg_predicted_close'] = 0.0
        return df_main

    # Define the aggregation based on frequency
    if frequency == '1M':
        # Use 1S data within the 1M period
        window = '1Min'
    elif frequency == '5M':
        # Use 1M data within the 5M period
        window = '5Min'
    elif frequency == '15M':
        # Use 5M data within the 15M period
        window = '15Min'
    elif frequency == '30M':
        # Use 5M data within the 30M period
        window = '30Min'
    elif frequency == '1H':
        # Use 15M data within the 1H period
        window = '1h'  # Changed 'H' to 'h'
    elif frequency == '1D':
        # Use 1H data within the 1D period
        window = '1d'
    else:
        window = None

    if window:
        try:
            # Resample lower_freq_df to match the main frequency's timestamp
            lower_freq_df_resampled = lower_freq_df.resample(window).agg({
                'trend': lambda x: (x == 'UP').sum(),
                'predicted_close': 'mean'
            }).rename(columns={'trend': 'lower_up_count', 'predicted_close': 'lower_avg_predicted_close'})

            # Calculate 'lower_down_count' as total minus 'lower_up_count'
            total_counts = lower_freq_df.resample(window).size()
            lower_freq_df_resampled['lower_down_count'] = total_counts - lower_freq_df_resampled['lower_up_count']

            # Merge the aggregated lower frequency data with the main df
            df_main = df_main.merge(lower_freq_df_resampled, left_index=True, right_index=True, how='left')

            # Fill NaNs with zeros
            df_main['lower_up_count'] = df_main['lower_up_count'].fillna(0).astype(int)
            df_main['lower_down_count'] = df_main['lower_down_count'].fillna(0).astype(int)
            df_main['lower_avg_predicted_close'] = df_main['lower_avg_predicted_close'].fillna(0.0)

        except Exception as e:
            logger.error(f"Error during resampling in aggregate_lower_freq_data for {frequency}: {e}")
            df_main['lower_up_count'] = 0
            df_main['lower_down_count'] = 0
            df_main['lower_avg_predicted_close'] = 0.0

    return df_main


def detect_trends(df, frequency, lower_freq_data=None, perform_ttt=True):
    """
    Detect trends and predict closing prices using an LSTM model.
    """
    logger.info("Trends being detected...")
    model_path = f"models/lstm_model_{frequency}.pt"

    # Ensure 'timestamp' is datetime and set as index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    else:
        logger.error("DataFrame does not contain 'timestamp' column.")
        return df

    # Use all available data
    logger.info(f"Data used for trend detection: {len(df)} records.")

    sequence_length = get_sequence_length(frequency)
    if len(df) < sequence_length + 1:
        logger.warning(f"Not enough data to create sequences for trend detection for {frequency}.")
        df['predicted_close'] = np.nan
        df['trend'] = 'STABLE'
        df.reset_index(inplace=True)
        return df

    # Prepare data
    df = calculate_technical_indicators(df)

    # Aggregate lower frequency data to create additional features
    df = aggregate_lower_freq_data(df, lower_freq_data, frequency)

    # Select features for the model
    feature_columns = ['close', 'sma', 'ema', 'lower_up_count', 'lower_down_count', 'lower_avg_predicted_close']
    df = df.dropna(subset=feature_columns)  # Drop rows with NaN in feature columns

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])

    sequences = []
    targets = []
    for i in range(len(scaled_features) - sequence_length):
        sequences.append(scaled_features[i:i+sequence_length])
        targets.append(scaled_features[i+sequence_length][0])  # 'close' is the target

    sequences = np.array(sequences)  # Shape: [num_samples, sequence_length, input_size]
    targets = np.array(targets)      # Shape: [num_samples, ]

    dataset = TimeSeriesDataset(torch.Tensor(sequences), torch.Tensor(targets))

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(sequences, targets, test_size=0.2, shuffle=False)

    train_dataset = TimeSeriesDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_dataset = TimeSeriesDataset(torch.Tensor(X_val), torch.Tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModelWithAttention(input_size=6).to(device)

    # Try to load the existing model
    model_loaded = False
    try:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded pre-trained model for {frequency} from {model_path}")
            model_loaded = True
    except (FileNotFoundError, RuntimeError) as e:
        # Handle size mismatch or other loading errors
        logger.warning(f"Could not load model for {frequency}: {e}")
        if isinstance(e, RuntimeError) and 'size mismatch' in str(e):
            # Delete the incompatible model file
            os.remove(model_path)
            logger.info(f"Deleted incompatible model file: {model_path}")
        model_loaded = False

    if not os.path.exists(model_path) or not model_loaded:
        logger.info(f"Training new model for {frequency}.")
        train_model(model, train_loader, val_loader, model_path, epochs=20, device=device, early_stopping_patience=5)

    # Implement Test-Time Training (TTT) always
    if perform_ttt:
        logger.info("Starting Test-Time Training (TTT)...")
        # Prepare TTT data (using the entire dataset for TTT)
        ttt_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
        train_model(model, ttt_loader, val_loader, model_path, epochs=10, device=device, early_stopping_patience=3)
        logger.info("Test-Time Training (TTT) completed.")

    # Predictions
    logger.debug("Starting prediction...")
    predictions = []
    model.eval()
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
            x = x.float().to(device)
            output = model(x)
            predictions.extend(output.squeeze().cpu().tolist())

    # Inverse transform predictions
    # Since we scaled all features, we need to invert only the 'close' feature
    scaler_inverse = MinMaxScaler()
    scaler_inverse.min_ = scaler.min_[0]
    scaler_inverse.scale_ = scaler.scale_[0]
    predictions = scaler_inverse.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    df = df.iloc[sequence_length:].copy()
    df['predicted_close'] = predictions

    # Add logging to check predictions
    logger.debug(f"Sample actual close prices: {df['close'].values[:5]}")
    logger.debug(f"Sample predicted close prices: {df['predicted_close'].values[:5]}")
    logger.debug(f"Predictions variance: {np.var(predictions)}")

    # Calculate MAE
    mae = mean_absolute_error(df['close'], df['predicted_close'])
    logger.info(f"Mean Absolute Error for {frequency}: {mae}")

    # Determine trends using SMA crossover
    df['trend'] = detect_trend_shifts(df, frequency)

    # Validate trends
    df = validate_trends(df, frequency)

    logger.info("Trends detected.")

    # Reset index to include 'timestamp' as a column before returning
    df.reset_index(inplace=True)

    return df


def detect_trend_shifts(df, frequency):
    """
    Determine the direction of trend within the time frequency and detect shifts in trend direction.
    This function implements logic to detect trend shifts based on Simple Moving Average (SMA) crossovers.
    """
    # Retrieve SMA crossover parameters from config.py
    params = TREND_PARAMETERS.get(frequency, {'short_window': 10, 'long_window': 50})
    short_window = params.get('short_window', 10)
    long_window = params.get('long_window', 50)

    logger.debug(f"Detecting trends for frequency {frequency} with short_window={short_window} and long_window={long_window}")

    # Calculate short-term and long-term SMA
    df['short_sma'] = df['sma'].rolling(window=short_window, min_periods=1).mean()
    df['long_sma'] = df['sma'].rolling(window=long_window, min_periods=1).mean()

    # Shift SMA for crossover detection
    df['prev_short_sma'] = df['short_sma'].shift(1)
    df['prev_long_sma'] = df['long_sma'].shift(1)

    # Initialize trend as 'STABLE'
    df['trend'] = 'STABLE'

    # Detect UP trend: short_sma crosses above long_sma
    df.loc[
        (df['short_sma'] > df['long_sma']) &
        (df['prev_short_sma'] <= df['prev_long_sma']),
        'trend'
    ] = 'UP'

    # Detect DOWN trend: short_sma crosses below long_sma
    df.loc[
        (df['short_sma'] < df['long_sma']) &
        (df['prev_short_sma'] >= df['prev_long_sma']),
        'trend'
    ] = 'DOWN'

    # Remove 'STABLE' trends as per requirement (Do not include in Dash charts)
    df['trend'] = df['trend'].replace('STABLE', np.nan)

    # Drop intermediate SMA columns
    df.drop(['short_sma', 'long_sma', 'prev_short_sma', 'prev_long_sma'], axis=1, inplace=True)

    return df['trend']


def validate_trends(df, frequency):
    """
    Validate the detected trends.
    Placeholder for additional validation logic.
    """
    logger.info(f"Validating {frequency} Trends...")
    # Implement your own validation logic here if needed
    invalid_trend_count = df['trend'].isna().sum()
    logger.debug(f"Total invalid trends detected for frequency {frequency}: {invalid_trend_count}")
    logger.info("Trends detected and validated.")
    return df