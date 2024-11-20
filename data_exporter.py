# data_exporter.py

import os
import pandas as pd
from database import get_connection, fetch_data
from config import TABLE_NAMES, NUM_DAYS_BACK
from logger import logger
import pickle

def export_data():
    """
    Connects to the database and exports data for all frequencies as pickle files.
    Each frequency's data is saved in the 'data' directory with the filename format 'actual_tsla_<frequency>.pkl'.
    """
    try:
        # Establish database connection
        engine = get_connection()
        logger.info("Database connection established for data export.")
    except Exception as e:
        logger.error(f"Failed to establish database connection: {e}")
        return

    # Ensure 'data' directory exists
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Ensured '{data_dir}' directory exists.")

    # Export data for each frequency
    for freq, table in TABLE_NAMES.items():
        try:
            logger.debug(f"Starting data export for frequency: {freq}")
            # Fetch data from the last NUM_DAYS_BACK days
            df = fetch_data(engine, table, last_processed_timestamp=None)
            if df.empty:
                logger.warning(f"No data fetched for frequency {freq}. Skipping export.")
                continue

            # Ensure 'timestamp' is a column, not index
            if df.index.name == 'timestamp':
                df.reset_index(inplace=True)
                logger.debug(f"Reset index to ensure 'timestamp' is a column for frequency '{freq}'.")

            # Save DataFrame as a pickle file
            file_path = os.path.join(data_dir, f"{table}.pkl")
            with open(file_path, 'wb') as file:
                pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Exported data for frequency {freq} to '{file_path}'.")
        except Exception as e:
            logger.error(f"Error exporting data for frequency {freq}: {e}")

    logger.info("Data export completed for all frequencies.")

if __name__ == '__main__':
    export_data()