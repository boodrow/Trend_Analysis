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
from sqlalchemy import create_engine

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

class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, bidirectional=True):
        """
        input_size: Number of input features per timestep.
        hidden_size: Number of features in the hidden state.
        num_layers: Number of recurrent layers.
        bidirectional: If True, use bidirectional LSTM.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.3, bidirectional=bidirectional
        )
        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_multiplier, 1)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last timestep
        return out  # Shape: [batch_size, 1]

def calculate_technical_indicators(df):
    df['sma'] = df['close'].rolling(window=5).mean()
    df['ema'] = df['close'].ewm(span=5, adjust=False).mean()
    return df

def train_model(model, loader, model_path, epochs=10, device='cpu', early_stopping_patience=3):
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
        with tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False) as t:
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

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.6f}")

        # Early Stopping Check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            # Save the best model whenever a new best loss is achieved
            torch.save(best_model_state, model_path)
            logger.info(f"New best model saved with loss {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                if 'best_model_state' in locals():
                    model.load_state_dict(best_model_state)
                break

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
        window = '1H'
    elif frequency == '1D':
        # Use 1H data within the 1D period
        window = '1D'
    else:
        window = None

    if window:
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

    return df_main

def detect_trends(df, frequency, lower_freq_data=None, perform_ttt=True):
    """
    Detect trends and predict closing prices using an LSTM model.
    """
    logger.info("Trends being detected...")
    model_path = f"models/lstm_model_{frequency}.pt"

    # Use all available data
    logger.info(f"Data used for trend detection: {len(df)} records.")

    sequence_length = get_sequence_length(frequency)
    if len(df) < sequence_length + 1:
        logger.warning(f"Not enough data to create sequences for trend detection for {frequency}.")
        df['predicted_close'] = np.nan
        df['trend'] = 'STABLE'
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
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=6).to(device)

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
        train_model(model, loader, model_path, epochs=20, device=device, early_stopping_patience=5)

    # Implement Test-Time Training (TTT) always
    if perform_ttt:
        logger.info("Starting Test-Time Training (TTT)...")
        ttt_epochs = 10
        ttt_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
        train_model(model, ttt_loader, model_path, epochs=ttt_epochs, device=device, early_stopping_patience=3)
        logger.info("Test-Time Training (TTT) completed.")

    # Predictions
    logger.debug("Starting prediction...")
    predictions = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.float().to(device)
            output = model(x)
            predictions.extend(output.squeeze().cpu().tolist())

    # Inverse transform predictions
    # Since we scaled all features, we need to invert only the 'close' feature
    # Initialize a scaler for inverse transformation
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

    # Determine trends
    df['trend'] = 'STABLE'
    df.loc[df['predicted_close'] > df['close'], 'trend'] = 'UP'
    df.loc[df['predicted_close'] < df['close'], 'trend'] = 'DOWN'

    # Validate trends
    df = validate_trends(df, frequency)

    logger.info("Trends detected.")
    return df

def validate_trends(df, frequency):
    """
    Validate the detected trends.
    Placeholder for additional validation logic.
    """
    logger.info(f"Validating {frequency} Trends...")
    # Implement your own validation logic here if needed
    invalid_trend_count = 0  # Assuming no invalid trends
    logger.debug(f"Total invalid trends detected for frequency {frequency}: {invalid_trend_count}")
    logger.info("Trends detected and validated.")
    return df