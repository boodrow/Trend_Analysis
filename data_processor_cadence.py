# data_processor_cadence.py

import os
import pickle
import pandas as pd
import torch
import warnings
from trend_calculator import detect_trends
from config import TABLE_NAMES, TREND_PARAMETERS, NUM_OF_EPOCHS, EARLY_STOPPING_PATIENCE
from logger import logger
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from trend_calculator import TimeSeriesDataset, LSTMModelWithAttention, train_model
import sys

# Suppress DeprecationWarnings related to numpy.core.numeric
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_data(data_dir='data'):
    """
    Loads data for all frequencies from pickle files.
    Returns a dictionary with frequency as keys and DataFrames as values.
    """
    data = {}
    for freq, table in TABLE_NAMES.items():
        file_path = os.path.join(data_dir, f"{table}.pkl")
        if not os.path.exists(file_path):
            logger.warning(f"Data file '{file_path}' does not exist. Skipping frequency '{freq}'.")
            continue
        try:
            with open(file_path, 'rb') as file:
                df = pickle.load(file)
            if 'timestamp' not in df.columns:
                logger.warning(f"'timestamp' column missing in data for frequency '{freq}'. Attempting to reset index.")
                if df.index.name == 'timestamp':
                    df.reset_index(inplace=True)
                    logger.debug(f"Reset index to ensure 'timestamp' is a column for frequency '{freq}'.")
                else:
                    logger.error(f"'timestamp' column is missing and cannot be reset for frequency '{freq}'. Skipping.")
                    continue
            # Ensure 'timestamp' is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isnull().all():
                logger.error(f"All 'timestamp' values are NaT for frequency '{freq}'. Skipping.")
                continue
            data[freq] = df
            logger.info(f"Loaded data for frequency '{freq}' from '{file_path}'.")
        except Exception as e:
            logger.error(f"Error loading data from '{file_path}': {e}")
    return data

def process_and_train(data, model_dir='.'):
    """
    Processes the data, trains the models, and saves them.
    Models are saved to the specified model_dir.
    """
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Ensured model directory '{model_dir}' exists.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    for freq, df in data.items():
        try:
            logger.debug(f"Starting processing for frequency '{freq}'.")
            # Detect trends and prepare data
            logger.debug(f"Detecting trends for frequency '{freq}'.")
            df_trends = detect_trends(df, freq, lower_freq_data=None, perform_ttt=False)

            # If trends are empty after detection, skip training
            if df_trends.empty:
                logger.warning(f"No trends detected for frequency '{freq}'. Skipping training.")
                continue

            # Prepare features and target
            feature_columns = ['close', 'sma', 'ema']
            # Check if additional lower frequency features are present
            additional_features = ['lower_up_count', 'lower_down_count', 'lower_avg_predicted_close']
            for feature in additional_features:
                if feature in df_trends.columns:
                    feature_columns.append(feature)
            df_features = df_trends[feature_columns].dropna()
            if df_features.empty:
                logger.warning(f"No valid feature data for frequency '{freq}'. Skipping training.")
                continue

            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(df_features)

            sequence_length = TREND_PARAMETERS.get(freq, {}).get('short_window', 10)
            if len(scaled_features) < sequence_length + 1:
                logger.warning(f"Not enough data for training frequency '{freq}'. Skipping.")
                continue

            sequences = []
            targets = []
            for i in range(len(scaled_features) - sequence_length):
                sequences.append(scaled_features[i:i+sequence_length])
                targets.append(scaled_features[i+sequence_length][0])  # 'close' is the target

            sequences = torch.Tensor(sequences)
            targets = torch.Tensor(targets)

            dataset = TimeSeriesDataset(sequences, targets)
            train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
            val_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

            # Initialize model with increased complexity
            base_num_layers = TREND_PARAMETERS.get(freq, {}).get('num_layers', 2)
            base_hidden_size = TREND_PARAMETERS.get(freq, {}).get('hidden_size', 128)
            base_num_transformer_layers = TREND_PARAMETERS.get(freq, {}).get('num_transformer_layers', 2)

            model = LSTMModelWithAttention(
                input_size=len(feature_columns),
                hidden_size=base_hidden_size * 10,  # Increased by 10x
                num_layers=base_num_layers * 10,      # Increased by 10x
                num_transformer_layers=base_num_transformer_layers * 10,  # Increased by 10x
                dropout=0.3,
                bidirectional=True
            ).to(device)

            # Use multiple GPUs if available
            if torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
                model = torch.nn.DataParallel(model)
            else:
                logger.info("Using a single GPU for training." if torch.cuda.is_available() else "Using CPU for training.")

            # Save models to the root directory
            model_filename = f"lstm_model_{freq}.pt"
            model_path = os.path.join(model_dir, model_filename)

            # Train model with dynamic learning rate and increased early stopping
            logger.info(f"Training model for frequency '{freq}'.")
            train_model(
                model,
                train_loader,
                val_loader,
                model_path,
                epochs=NUM_OF_EPOCHS,
                device=device,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                dynamic_lr=True  # Enable dynamic learning rate
            )

            logger.info(f"Model training completed for frequency '{freq}'. Model saved to '{model_path}'.")

        except Exception as e:
            logger.error(f"Error processing and training for frequency '{freq}': {e}")

def main():
    """
    Main function to execute the data processing and model training.
    """
    try:
        logger.debug("Starting data_processor_cadence.py")
        data = load_data(data_dir='data')
        if not data:
            logger.warning("No data loaded. Exiting.")
            sys.exit(0)
        process_and_train(data, model_dir='.')  # Save models to root directory
        logger.info("Data processing and model training completed successfully.")
    except Exception as e:
        logger.critical(f"Fatal error in data_processor_cadence.py: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()