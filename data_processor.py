# data_processor.py

import os
import time
import pandas as pd
import sys
import torch
from sqlalchemy import text
from logger import logger
from config import DATABASE, TABLE_NAMES, NUM_DAYS_BACK, NUM_OF_EPOCHS, EARLY_STOPPING_PATIENCE
from trend_calculator import detect_trends, calculate_technical_indicators
from database import (
    get_connection,
    initialize_last_processed_table,
    fetch_data,
    update_trends,
    ensure_columns
)

torch.set_num_threads(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def process_table(engine, table_name, frequency, num_days):
    """
    Process a table by fetching data from NUM_DAYS_BACK days up to present,
    conducting Test-Time Training (TTT), detecting trends, and updating the database.
    """
    try:
        logger.debug(f"Starting process_table for frequency: {frequency}")

        # Determine the time range: from num_days ago to now
        end_time = pd.Timestamp.utcnow()
        start_time = end_time - pd.Timedelta(days=num_days)
        logger.info(f"Processing data from {start_time} to {end_time} for frequency '{frequency}'.")

        # Clear existing 'trend' and 'predicted_close' within the time range
        logger.debug(f"Clearing existing 'trend' and 'predicted_close' data for {frequency} from {start_time} to {end_time}.")
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    UPDATE {table_name}
                    SET trend = NULL,
                        predicted_close = NULL
                    WHERE timestamp BETWEEN :start_time AND :end_time
                """),
                {'start_time': start_time, 'end_time': end_time}
            )
        logger.info(f"Cleared 'trend' and 'predicted_close' data from {table_name} between {start_time} and {end_time}.")

        # Fetch data within the time range
        logger.debug(f"Fetching data for frequency '{frequency}' from {start_time} to {end_time}.")
        df = fetch_data(engine, table_name, start_time=start_time, end_time=end_time)
        if df.empty:
            logger.info(f"No data found in the specified time range for frequency '{frequency}'. Skipping.")
            return

        logger.info(f"Fetched {len(df)} records for frequency '{frequency}' from {table_name}.")

        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        logger.info("Calculated technical indicators.")

        # Detect trends and perform Test-Time Training (TTT)
        logger.debug(f"Detecting trends and performing TTT for frequency '{frequency}'.")
        df_trends = detect_trends(df, frequency, lower_freq_data=None, perform_ttt=True)

        if df_trends.empty:
            logger.warning(f"No trends detected for frequency '{frequency}'. Skipping update.")
            return

        logger.info(f"Detected trends for frequency '{frequency}'.")

        # Update the 'trend' and 'predicted_close' columns in the database
        logger.debug(f"Updating 'trend' and 'predicted_close' in the database for frequency '{frequency}'.")
        update_trends(engine, table_name, df_trends)

        logger.info(f"Updated 'trend' and 'predicted_close' in {table_name} for frequency '{frequency}'.")

    except Exception as e:
        logger.error(f"Error processing table {table_name} for frequency '{frequency}': {e}")


def main():
    """
    Main function to execute the data processing and model training.
    """
    try:
        logger.debug("Starting data_processor.py")
        engine = get_connection()
        initialize_last_processed_table(engine)
        # Ensure 'trend' and 'predicted_close' columns exist for all tables
        columns_to_ensure = {'trend': 'TEXT', 'predicted_close': 'REAL'}
        for freq, table in TABLE_NAMES.items():
            ensure_columns(engine, table, columns_to_ensure)
        logger.debug("Initialization complete.")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        sys.exit(1)  # Exit if initialization fails

    # Define TTT frequency (in seconds)
    POLL_INTERVAL = 10  # Poll every 10 seconds

    # Determine the number of days to look back
    NUM_DAYS = NUM_DAYS_BACK

    # Initial run: Process the last NUM_DAYS_back days
    for freq, table in TABLE_NAMES.items():
        logger.debug(f"Performing initial processing for table {table} with frequency '{freq}'.")
        process_table(engine, table, freq, num_days=NUM_DAYS)

    # Continuous processing loop
    while True:
        logger.debug("Starting data processing loop.")
        for freq, table in TABLE_NAMES.items():
            logger.debug(f"Processing table '{table}' for frequency '{freq}'.")
            process_table(engine, table, freq, num_days=NUM_DAYS)
            logger.debug(f"Finished processing table '{table}' for frequency '{freq}'.")
        logger.debug(f"Sleeping for {POLL_INTERVAL} seconds before next poll.")
        time.sleep(POLL_INTERVAL)  # Wait for POLL_INTERVAL seconds before next run


if __name__ == '__main__':
    # Ensure model directory exists
    os.makedirs('models', exist_ok=True)
    main()