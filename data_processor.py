# data_processor.py

import os
import time
import pandas as pd
import sys
import torch
from sqlalchemy import text
from logger import logger
from config import (
    DATABASE,
    TABLE_NAMES,
    NUM_DAYS_BACK,
    NUM_OF_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    TTT_INTERVAL,
    PERFORM_TTT,
    TREND_PARAMETERS,
    TIMESTAMP_DIR,
    FLOAT_TOLERANCE,
    BATCH_UPDATE
)
from trend_calculator import detect_trends, calculate_technical_indicators
from database import (
    get_connection,
    fetch_data,
    update_trends,
    ensure_columns
)

import pickle

torch.set_num_threads(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_last_processed_timestamp(freq, pkl_dir=TIMESTAMP_DIR):
    """
    Loads the last processed timestamp for a given frequency from a pkl file.
    If the file does not exist, returns None.
    """
    os.makedirs(pkl_dir, exist_ok=True)
    pkl_file = os.path.join(pkl_dir, f"{freq}_last_timestamp.pkl")
    if os.path.exists(pkl_file):
        try:
            with open(pkl_file, 'rb') as f:
                last_timestamp = pickle.load(f)
                logger.debug(f"Loaded last processed timestamp for frequency '{freq}': {last_timestamp}")
                return last_timestamp
        except Exception as e:
            logger.error(f"Error loading last processed timestamp from '{pkl_file}': {e}")
            return None
    else:
        logger.debug(f"No last processed timestamp file found for frequency '{freq}'.")
        return None


def save_last_processed_timestamp(freq, timestamp, pkl_dir=TIMESTAMP_DIR):
    """
    Saves the last processed timestamp for a given frequency to a pkl file.
    """
    os.makedirs(pkl_dir, exist_ok=True)
    pkl_file = os.path.join(pkl_dir, f"{freq}_last_timestamp.pkl")
    try:
        with open(pkl_file, 'wb') as f:
            pickle.dump(timestamp, f)
        logger.debug(f"Saved last processed timestamp for frequency '{freq}': {timestamp}")
    except Exception as e:
        logger.error(f"Error saving last processed timestamp to '{pkl_file}': {e}")


def process_table(engine, table_name, frequency, model_dir='models'):
    """
    Process a table by fetching new data since the last processed timestamp,
    detecting trends, updating the database, and verifying the update.
    """
    try:
        logger.debug(f"Starting process_table for frequency: {frequency}")

        # Define the path to the existing model
        model_path = os.path.join(model_dir, f"lstm_model_{frequency}.pt")

        # Check if the model exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file '{model_path}' does not exist. Skipping frequency '{frequency}'.")
            return  # Skip processing for this frequency

        logger.info(f"Using existing model for frequency '{frequency}' from '{model_path}'.")

        # Load the last processed timestamp from pkl file
        last_timestamp = load_last_processed_timestamp(frequency)

        if last_timestamp:
            try:
                start_time = pd.to_datetime(last_timestamp) + pd.Timedelta(seconds=1)  # Start just after the last timestamp
                logger.debug(f"Last processed timestamp for '{frequency}': {last_timestamp}")
            except Exception as e:
                logger.error(f"Invalid timestamp format for frequency '{frequency}': {last_timestamp}. Processing last {NUM_DAYS_BACK} days.")
                start_time = pd.Timestamp.utcnow() - pd.Timedelta(days=NUM_DAYS_BACK)
        else:
            # If no timestamp exists, process the last NUM_DAYS_back days
            start_time = pd.Timestamp.utcnow() - pd.Timedelta(days=NUM_DAYS_BACK)
            logger.debug(f"No last processed timestamp found for '{frequency}'. Processing last {NUM_DAYS_BACK} days.")

        end_time = pd.Timestamp.utcnow()
        logger.info(f"Processing data from {start_time} to {end_time} for frequency '{frequency}'.")

        # Fetch data within the time range
        logger.debug(f"Fetching data for frequency '{frequency}' from {start_time} to {end_time}.")
        df = fetch_data(engine, table_name, start_time=start_time, end_time=end_time)
        if df.empty:
            logger.info(f"No new data found in the specified time range for frequency '{frequency}'. Skipping.")
            return

        logger.info(f"Fetched {len(df)} new records for frequency '{frequency}' from {table_name}.")

        # Ensure 'timestamp' column exists
        if 'timestamp' not in df.columns:
            logger.error(f"'timestamp' column is missing in fetched data for frequency '{frequency}'. Skipping.")
            return

        # Ensure 'timestamp's are unique
        if df['timestamp'].duplicated().any():
            logger.warning(f"Duplicate 'timestamp's found in data for frequency '{frequency}'. Removing duplicates.")
            df = df.drop_duplicates(subset=['timestamp'])

        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        logger.info("Calculated technical indicators.")

        # Detect trends and perform Test-Time Training (TTT) if enabled
        logger.debug(f"Detecting trends and performing TTT for frequency '{frequency}'.")
        df_trends = detect_trends(df, frequency, lower_freq_data=None, perform_ttt=PERFORM_TTT)

        if df_trends.empty:
            logger.warning(f"No trends detected for frequency '{frequency}'. Skipping update.")
            return

        logger.info(f"Detected trends for frequency '{frequency}'.")

        # Include 'sma' and 'ema' in the update by merging with the original DataFrame
        df_trends = df_trends.merge(df[['timestamp', 'sma', 'ema']], on='timestamp', how='left', validate='one_to_one')
        logger.debug(f"Included 'sma' and 'ema' in the trends DataFrame.")

        # Ensure 'timestamp's are unique in df_trends
        if df_trends['timestamp'].duplicated().any():
            logger.warning(f"Duplicate 'timestamp's found after merging in frequency '{frequency}'. Removing duplicates.")
            df_trends = df_trends.drop_duplicates(subset=['timestamp'])

        # Log the size before updating
        logger.debug(f"Number of records to update for frequency '{frequency}': {len(df_trends)}")

        # Drop rows with NaN in 'predicted_close' and 'trend' to prevent verification errors
        df_trends = df_trends.dropna(subset=['predicted_close', 'trend'])
        logger.debug(f"Number of records after dropping NaNs: {len(df_trends)}")

        if df_trends.empty:
            logger.warning(f"All records have NaN 'predicted_close' or 'trend' for frequency '{frequency}'. Skipping update.")
            return

        # Ensure 'timestamp' is present after merging
        if 'timestamp' not in df_trends.columns:
            logger.error("'timestamp' column missing after merging 'sma' and 'ema'.")
            raise KeyError("'timestamp' column missing after merging 'sma' and 'ema'.")
        else:
            logger.debug("'timestamp' column present after merging.")

        # Update the 'trend', 'predicted_close', 'sma', and 'ema' columns in the database with verification
        logger.debug(f"Updating 'trend', 'predicted_close', 'sma', and 'ema' in the database for frequency '{frequency}'.")
        update_trends(engine, table_name, df_trends, freq=frequency)

        # Update the last processed timestamp in pkl file
        latest_timestamp = df['timestamp'].max()
        if pd.isna(latest_timestamp):
            logger.warning(f"No valid timestamp found in the data for frequency '{frequency}'.")
        else:
            save_last_processed_timestamp(frequency, latest_timestamp)
            logger.info(f"Updated last processed timestamp for '{frequency}' to {latest_timestamp} in pkl file.")

    except Exception as e:
        logger.error(f"Error processing table {table_name} for frequency '{frequency}': {e}")
        sys.exit(1)  # Stop the program upon encountering an error as per requirement


def main():
    """
    Main function to execute the data processing and trend predictions.
    """
    try:
        logger.debug("Starting data_processor.py")
        engine = get_connection()
        # Ensure 'trend', 'predicted_close', 'sma', and 'ema' columns exist for all tables
        columns_to_ensure = {'trend': 'TEXT', 'predicted_close': 'DOUBLE PRECISION', 'sma': 'REAL', 'ema': 'REAL'}
        for freq, table in TABLE_NAMES.items():
            ensure_columns(engine, table, columns_to_ensure)
        logger.debug("Initialization complete.")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        sys.exit(1)  # Exit if initialization fails

    # Define TTT frequency (in seconds)
    POLL_INTERVAL = TTT_INTERVAL  # As per config, default 300 seconds

    # Initial run: Process new data based on last processed timestamp
    for freq, table in TABLE_NAMES.items():
        logger.debug(f"Performing initial processing for table {table} with frequency '{freq}'.")
        process_table(engine, table, freq, model_dir='models')

    # Continuous processing loop
    while True:
        logger.debug("-----------------------------------\nStarting data processing loop.")
        for freq, table in TABLE_NAMES.items():
            logger.debug(f"Processing table '{table}' for frequency '{freq}'.")
            process_table(engine, table, freq, model_dir='models')
            logger.debug(f"Finished processing table '{table}' for frequency '{freq}'.")
        logger.debug(f"Sleeping for {POLL_INTERVAL} seconds before next processing run.")
        time.sleep(POLL_INTERVAL)  # Wait for POLL_INTERVAL seconds before next run


if __name__ == '__main__':
    # Ensure model directory exists
    os.makedirs('models', exist_ok=True)
    main()