# data_processor.py

import os
import time
import pandas as pd
import sys
import torch
from sqlalchemy import text
from logger import logger
from config import DATABASE, TABLE_NAMES, NUM_DAYS_BACK
from trend_calculator import detect_trends, calculate_technical_indicators
from database import (
    get_connection,
    initialize_last_processed_table,
    get_last_processed_timestamp,
    update_last_processed_timestamp,
    fetch_data,
    update_trends,
    ensure_columns
)

torch.set_num_threads(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def process_table(engine, table_name, frequency, full_refresh=False):
    """
    Process a table by fetching data, detecting trends, and updating the database.
    If full_refresh is True, it will fetch data from the last NUM_DAYS_BACK days.
    """
    try:
        logger.debug(f"Starting process_table for frequency: {frequency}")

        if full_refresh:
            # Ignore last_processed_timestamp and fetch all data from NUM_DAYS_BACK days
            last_processed_timestamp = None
            logger.info(f"Performing full refresh for {frequency}. Fetching data from last {NUM_DAYS_BACK} days.")
        else:
            last_processed_timestamp = get_last_processed_timestamp(engine, frequency)

        # Determine if model file exists
        model_path = f"models/lstm_model_{frequency}.pt"
        model_exists = os.path.exists(model_path)

        # Decide whether to clear data based on model existence and processing mode
        if not model_exists:
            # If model doesn't exist and initial training is needed, do not change anything with the queries
            logger.debug(f"No existing model for {frequency}. Proceeding without clearing predictions.")
        else:
            if last_processed_timestamp:
                # Remove existing 'trend' and 'predicted_close' data for the data that needs to be reprocessed
                logger.debug(f"Clearing existing 'trend' and 'predicted_close' data for {frequency} starting from {last_processed_timestamp}")
                with engine.begin() as conn:
                    conn.execute(
                        text(f"UPDATE {table_name} SET trend = NULL WHERE timestamp >= :last_processed_timestamp AND trend IS NOT NULL"),
                        {'last_processed_timestamp': last_processed_timestamp}
                    )
                    conn.execute(
                        text(f"UPDATE {table_name} SET predicted_close = NULL WHERE timestamp >= :last_processed_timestamp AND predicted_close IS NOT NULL"),
                        {'last_processed_timestamp': last_processed_timestamp}
                    )
                logger.info(f"Cleared 'trend' and 'predicted_close' data from {table_name} starting from {last_processed_timestamp}.")
            else:
                # No last_processed_timestamp, clear all data (full refresh)
                logger.debug(f"Clearing all 'trend' and 'predicted_close' data for {frequency}")
                with engine.begin() as conn:
                    conn.execute(text(f"UPDATE {table_name} SET trend = NULL WHERE trend IS NOT NULL"))
                    conn.execute(text(f"UPDATE {table_name} SET predicted_close = NULL WHERE predicted_close IS NOT NULL"))
                logger.info(f"Cleared all 'trend' and 'predicted_close' data from {table_name}.")

        # Fetch data
        df = fetch_data(engine, table_name, last_processed_timestamp=last_processed_timestamp)
        if df.empty:
            logger.info(f"No new data to process for {frequency}.")
            return pd.DataFrame()
        logger.info(f"Processing {len(df)} records for table {table_name}.")

        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        logger.info("Technical indicators calculated.")

        # Fetch lower frequency trend data if necessary
        lower_freq_data = None
        if frequency != '1S':
            lower_frequency = get_lower_frequency(frequency)
            if lower_frequency:
                lower_table_name = TABLE_NAMES.get(lower_frequency)
                if lower_table_name:
                    logger.debug(f"Fetching lower frequency trend data from {lower_frequency}")
                    lower_df = fetch_lower_frequency_data(engine, lower_table_name, df['timestamp'].min(), df['timestamp'].max())
                    if not lower_df.empty:
                        lower_freq_data = lower_df
                        logger.info(f"Fetched {len(lower_df)} records of lower frequency data for {frequency}.")
                    else:
                        logger.warning(f"No lower frequency data found for {frequency}.")
                else:
                    logger.warning(f"Lower frequency '{lower_frequency}' not found in TABLE_NAMES.")
            else:
                logger.warning(f"No lower frequency available for {frequency}.")

        # Detect trends
        logger.debug(f"Starting trend detection for {frequency}")
        df = detect_trends(df, frequency, lower_freq_data=lower_freq_data, perform_ttt=True)
        if df is None or df.empty:
            logger.warning(f"No trends detected for {frequency}.")
            return pd.DataFrame()
        logger.info("Trends detected.")

        # Update the 'trend' and 'predicted_close' columns in the database
        logger.debug(f"Updating trends in the database for {frequency}")
        update_trends(engine, table_name, df)

        # Update the last processed timestamp only if not performing full refresh
        if not full_refresh:
            latest_record = df.iloc[-1]
            if 'timestamp' in latest_record:
                new_last_timestamp = latest_record['timestamp']
                logger.info(f"Updated last processed timestamp for {frequency}: {new_last_timestamp}")
                update_last_processed_timestamp(engine, frequency, new_last_timestamp)
            else:
                logger.warning(f"Missing 'timestamp' in the latest record for {frequency}.")

        return df
    except Exception as e:
        logger.error(f"Error processing table {table_name}: {e}")
        return pd.DataFrame()

def get_lower_frequency(frequency):
    """
    Get the lower frequency for the given frequency.
    """
    freq_order = ['1S', '1M', '5M', '15M', '30M', '1H', '1D']
    try:
        idx = freq_order.index(frequency)
        if idx > 0:
            return freq_order[idx - 1]
        else:
            return None
    except ValueError:
        return None

def fetch_lower_frequency_data(engine, table_name, start_time, end_time):
    """
    Fetch lower frequency trend data between start_time and end_time.
    """
    query = f"""
        SELECT timestamp, trend, predicted_close
        FROM {table_name}
        WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, engine)
    return df

def main():
    try:
        logger.debug("Starting data processor")
        engine = get_connection()
        initialize_last_processed_table(engine)
        # Ensure 'trend' and 'predicted_close' columns exist for all tables
        columns_to_ensure = {'trend': 'TEXT', 'predicted_close': 'REAL'}
        for freq, table in TABLE_NAMES.items():
            ensure_columns(engine, table, columns_to_ensure)
        logger.debug("Initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        sys.exit(1)  # Exit if initialization fails

    # Perform a full refresh to fetch data from the last NUM_DAYS_BACK days
    for freq, table in TABLE_NAMES.items():
        logger.debug(f"Performing full refresh for table {table} with frequency {freq}")
        process_table(engine, table, freq, full_refresh=True)

    # Continuous processing loop
    while True:
        logger.debug("Starting data processing loop")
        for freq, table in TABLE_NAMES.items():
            logger.debug(f"Processing table {table} for frequency {freq}")
            process_table(engine, table, freq)
            logger.debug(f"Finished processing table {table} for frequency {freq}")
        time.sleep(60)  # Wait for 60 seconds before next iteration

if __name__ == '__main__':
    main()